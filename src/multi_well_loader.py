"""
MULTI-WELL DLIS LOADER
======================
Mevcut true_vdl_loader.py'nin genelleştirilmiş versiyonu.

Farklar:
- Birden fazla kuyudan waveform+scalar yükleyebilir
- Frame/channel isimleri sabit değil, config'ten gelir (auto-detect fallback'li)
- Her kuyuya bir `well_id` eklenir - cross-well split için hayati
- Tüm veri tek bir büyük tabloda birleştirilir, ama well_id korunur

Bu modül mevcut true_vdl_loader.py'yi hiç değiştirmez.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

import dlisio.dlis

from src.utils import (print_info, print_success, print_warning,
                       print_error, print_section, find_files)


# ============================================================================
# TEK KUYU LOADER
# ============================================================================

class WellLoader:
    """Bir kuyudan VDL waveform + scalar data yükler."""

    def __init__(self, well_id: str, well_cfg: dict, dlis_path: Path):
        self.well_id = well_id
        self.cfg = well_cfg
        self.dlis_path = Path(dlis_path)

        # Çıktı holder'ları
        self.depth: Optional[np.ndarray] = None
        self.waveform: Optional[np.ndarray] = None        # (N, 256)
        self.scalar_data: Optional[pd.DataFrame] = None
        self.metadata: dict = {}

    # -------- frame/channel keşfi --------

    @staticmethod
    def _find_frame(lf, target_name: Optional[str]):
        """İsimle veya autodetect ile VDL frame'i bul."""
        if target_name:
            matches = [fr for fr in lf.frames if fr.name == target_name]
            if matches:
                return matches[0]
            print_warning(f"Frame '{target_name}' bulunamadı, autodetect deneniyor...")

        # Autodetect: en fazla channel'a sahip frame genelde ana waveform frame'i olur
        best = None
        best_n = 0
        for fr in lf.frames:
            try:
                n = len(fr.channels)
            except Exception:
                n = 0
            if n > best_n:
                best_n = n
                best = fr
        return best

    @staticmethod
    def _channel_names(frame) -> list:
        return [ch.name for ch in frame.channels]

    @staticmethod
    def _fetch(frame, name: str, curves=None):
        """Channel'ı güvenli çek (yoksa None)."""
        try:
            if curves is None:
                curves = frame.curves()
            return curves[name]
        except (KeyError, IndexError, Exception):
            return None

    @staticmethod
    def _autodetect_waveform_channel(frame, curves, time_samples_expected: int = 256) -> Optional[str]:
        """
        2-boyutlu (N, time) shape'li ve beklenen sample sayısına uyan channel ara.
        Bu, frame isimleri farklı tool'larda değiştiğinde (Volve vb.) iş görür.
        """
        best = None
        for ch in frame.channels:
            name = ch.name
            try:
                arr = curves[name]
            except Exception:
                continue
            if arr.ndim == 2 and arr.shape[1] == time_samples_expected:
                # İsimde WF/WAVE/VDL/CBL geçenleri öncelikli seç
                up = name.upper()
                score = sum(k in up for k in ("WF", "WAVE", "VDL", "CBL"))
                if best is None or score > best[0]:
                    best = (score, name)
        return best[1] if best else None

    # -------- ana yükleme --------

    def load(self) -> "WellLoader":
        print_info(f"DLIS açılıyor: {self.dlis_path.name}")

        with dlisio.dlis.load(str(self.dlis_path)) as batch:
            logical_files = list(batch)
            if not logical_files:
                raise RuntimeError("Logical file bulunamadı.")

            # İlk logical file'la çalış (çoklu pass varsa ileride genişletilebilir)
            lf = logical_files[0]

            # 1) Frame bul
            frame = self._find_frame(lf, self.cfg.get("vdl_frame"))
            if frame is None:
                raise RuntimeError("VDL frame bulunamadı.")
            print_info(f"Frame: {frame.name}  |  channels: {len(frame.channels)}")

            curves = frame.curves()

            # 2) Waveform channel
            wf_name = self.cfg.get("wf_channel")
            if not wf_name or wf_name not in self._channel_names(frame):
                wf_name = self._autodetect_waveform_channel(frame, curves)
                if wf_name is None:
                    raise RuntimeError(
                        f"Waveform channel bulunamadı. Mevcut channels: "
                        f"{self._channel_names(frame)[:20]}"
                    )
                print_warning(f"Waveform channel autodetect: {wf_name}")

            # Seyreltme: farklı örnekleme aralıklı kuyuları aynı fiziksel
            # pencere boyuna getirmek için (ör. 16B-32: 3 in → 6 in)
            dec = int(self.cfg.get("decimate", 1))

            self.waveform = np.asarray(curves[wf_name])[::dec]
            if self.waveform.ndim != 2:
                raise RuntimeError(f"Waveform shape beklenmiyor: {self.waveform.shape}")
            print_success(f"Waveform ({wf_name}): {self.waveform.shape}")

            # 3) Depth
            depth_name = self.cfg.get("depth_channel", "TDEP")
            depth = self._fetch(frame, depth_name, curves)
            if depth is None:
                # Yaygın alternatifler
                for alt in ("MD", "DEPT", "DEPTH"):
                    depth = self._fetch(frame, alt, curves)
                    if depth is not None:
                        depth_name = alt
                        break
            if depth is None:
                raise RuntimeError("Depth channel bulunamadı.")
            self.depth = np.asarray(depth)[::dec]
            print_success(f"Depth ({depth_name}): {self.depth.shape}  "
                          f"[{self.depth.min():.1f} - {self.depth.max():.1f}]")

            # 4) Scalar channels
            scalar = {}
            for ch_name in self.cfg.get("scalar_channels", []):
                arr = self._fetch(frame, ch_name, curves)
                if arr is None:
                    print_warning(f"Scalar channel '{ch_name}' yok, atlandı")
                    continue
                arr = np.asarray(arr, dtype=np.float64)[::dec]
                if arr.ndim != 1:
                    print_warning(f"Scalar '{ch_name}' 1D değil ({arr.shape}), atlandı")
                    continue
                # DLIS kayıp değer kodu (-999.25) → NaN; aksi halde
                # label_3class bu örnekleri Poor olarak etiketler
                arr[arr <= -999.0] = np.nan
                if len(arr) != len(self.depth):
                    print_warning(f"Scalar '{ch_name}' uzunluğu uyumsuz, atlandı")
                    continue
                scalar[ch_name] = arr

            self.scalar_data = pd.DataFrame(scalar)
            print_success(f"Scalar channels: {list(self.scalar_data.columns)}  "
                          f"({len(self.scalar_data)} samples)")

            # 5) Metadata
            self.metadata = {
                "well_id": self.well_id,
                "well_name": self.cfg["name"],
                "site": self.cfg["site"],
                "formation": self.cfg["formation"],
                "frame": frame.name,
                "wf_channel": wf_name,
                "depth_channel": depth_name,
                "decimate": dec,
                "n_samples": int(len(self.depth)),
                "depth_min": float(self.depth.min()),
                "depth_max": float(self.depth.max()),
                "file": str(self.dlis_path.name),
            }

        return self

    # -------- windowing --------

    def make_windows(self, window_size: int, step: int):
        """
        (N, time) waveform'ı → (M, window_size, time) window'larına böl.
        Ayrıca her window için orta-derinlik indeksindeki CBL/BI değerlerini döndür.

        Return: dict with
          - windows  : (M, window_size, time_samples)
          - depths   : (M,) orta-derinlik
          - cbl      : (M,) orta-sample'daki CBL (yoksa NaN)
          - bi       : (M,) orta-sample'daki BI  (yoksa NaN)
          - well_ids : (M,) her window için well_id (string)
          - starts   : (M,) window'un ilk örnek indeksi (purge gap için)
        """
        if self.waveform is None:
            raise RuntimeError("Önce load() çağır.")

        N = len(self.waveform)
        starts = list(range(0, N - window_size + 1, step))
        if not starts:
            print_warning(f"{self.well_id}: window oluşturulamadı (yetersiz sample).")
            return {
                "windows": np.empty((0, window_size, self.waveform.shape[1]), dtype=np.float32),
                "depths": np.empty(0),
                "cbl": np.empty(0), "bi": np.empty(0),
                "well_ids": np.empty(0, dtype=object),
                "starts": np.empty(0, dtype=np.int64),
            }

        wins = np.stack([self.waveform[s:s + window_size] for s in starts]).astype(np.float32)

        mids = np.array([s + window_size // 2 for s in starts])
        depths = self.depth[mids]

        def safe_mid(col):
            if col in self.scalar_data.columns:
                return self.scalar_data[col].to_numpy()[mids]
            return np.full(len(mids), np.nan)

        cbl = safe_mid(self.cfg.get("cbl_channel", "CBL"))
        bi = safe_mid(self.cfg.get("bi_channel", "BI"))

        well_ids = np.array([self.well_id] * len(starts), dtype=object)

        return {
            "windows": wins,
            "depths": depths,
            "cbl": cbl,
            "bi": bi,
            "well_ids": well_ids,
            "starts": np.asarray(starts, dtype=np.int64),
        }


# ============================================================================
# KUYU BULUCU
# ============================================================================

def locate_dlis_for_well(well_id: str, well_cfg: dict, raw_dir: Path) -> Optional[Path]:
    """
    raw_dir içinde well_cfg['file_pattern'] ile eşleşen .dlis dosyası ara.
    Önce well_id alt klasöründe, sonra tüm raw'de.
    """
    pattern = well_cfg.get("file_pattern", "*.dlis")

    # 1) well-specific subdir
    subdir = Path(raw_dir) / well_id
    if subdir.exists():
        hits = find_files(subdir, pattern)
        if hits:
            return hits[0]
        # Any .dlis in subdir as fallback
        hits = find_files(subdir, "*.dlis") + find_files(subdir, "*.DLIS")
        if hits:
            return hits[0]

    # 2) search whole raw dir
    hits = find_files(raw_dir, pattern)
    if hits:
        return hits[0]

    return None


# ============================================================================
# TÜM KUYULARI YÜKLE
# ============================================================================

def load_all_enabled_wells(wells_cfg: dict, raw_dir: Path,
                           window_size: int, step: int) -> dict:
    """
    Tüm aktif kuyuları yükle ve windowing uygula. Concat edilmiş sonucu döndürür.

    Return:
      {
        "windows":  (M_total, window_size, time),
        "depths":   (M_total,),
        "cbl":      (M_total,),
        "bi":       (M_total,),
        "well_ids": (M_total,) dtype=object,
        "per_well_metadata": [ {...}, ... ],
      }
    """
    all_windows, all_depths, all_cbl, all_bi, all_wids = [], [], [], [], []
    metadatas = []

    for well_id, cfg in wells_cfg.items():
        if not cfg.get("enabled", False):
            continue

        print_section(f"Yükleniyor: {cfg['name']}  ({well_id})")
        path = locate_dlis_for_well(well_id, cfg, raw_dir)
        if path is None:
            print_warning(f"DLIS bulunamadı, atlanıyor: {well_id}")
            continue

        try:
            loader = WellLoader(well_id, cfg, path).load()
        except Exception as e:
            print_error(f"Yüklenemedi ({well_id}): {e}")
            continue

        # expected time_samples kontrolü
        if loader.waveform.shape[1] not in (256, 128, 512):
            print_warning(f"Beklenmeyen time-sample sayısı: {loader.waveform.shape[1]}")

        chunk = loader.make_windows(window_size=window_size, step=step)
        print_success(f"{well_id}: {len(chunk['windows'])} window üretildi")

        all_windows.append(chunk["windows"])
        all_depths.append(chunk["depths"])
        all_cbl.append(chunk["cbl"])
        all_bi.append(chunk["bi"])
        all_wids.append(chunk["well_ids"])
        metadatas.append({**loader.metadata, "n_windows": int(len(chunk['windows']))})

    if not all_windows:
        raise RuntimeError("Hiçbir kuyudan veri yüklenemedi. Önce download_data.py çalıştır.")

    # İlk kuyunun time_samples'ını referans al
    ref_time = all_windows[0].shape[2]
    # Tüm kuyuların aynı time boyutunda olduğundan emin ol
    filtered = [(w, d, c, b, i) for w, d, c, b, i in
                zip(all_windows, all_depths, all_cbl, all_bi, all_wids)
                if w.shape[2] == ref_time]
    if len(filtered) != len(all_windows):
        print_warning("Farklı time-sample boyutuna sahip kuyular atlandı.")
    all_windows, all_depths, all_cbl, all_bi, all_wids = zip(*filtered)

    out = {
        "windows":  np.concatenate(all_windows, axis=0),
        "depths":   np.concatenate(all_depths, axis=0),
        "cbl":      np.concatenate(all_cbl, axis=0),
        "bi":       np.concatenate(all_bi, axis=0),
        "well_ids": np.concatenate(all_wids, axis=0),
        "per_well_metadata": metadatas,
    }
    return out
