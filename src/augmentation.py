"""
AUGMENTATION
============
CBL waveform verisi için fiziksel olarak anlamlı 6 augmentation.

Hepsi numpy üzerinde çalışır, batch-friendly.
Her augmentation kendi 'p' parametresiyle olasılığa bağlı uygulanır.

Tasarım prensipleri:
- Fiziksel olarak gerçekçi (CBL tool'un gerçek belirsizliklerini taklit eder)
- Label-preserving (Good window'a noise eklemekle Poor olmaz)
- Boundary-safe (clip, padding ile bounds aşılmaz)
"""

import numpy as np
from typing import Tuple


# ============================================================================
# 1. GAUSSIAN NOISE
# ============================================================================
# Gerçek CBL tool electronics noise'u Gaussian ve SNR ~20-40 dB civarı.
# Her window'a kendi std'sine göre orantılı noise ekleriz.

def gaussian_noise(window: np.ndarray, snr_db_range: Tuple[float, float] = (20.0, 35.0),
                   rng: np.random.Generator = None) -> np.ndarray:
    """
    SNR (signal-to-noise ratio) bazlı Gaussian noise.
    Düşük SNR = daha gürültülü.
    """
    rng = rng or np.random.default_rng()
    snr_db = rng.uniform(*snr_db_range)
    signal_power = np.mean(window ** 2) + 1e-10
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), window.shape).astype(window.dtype)
    return window + noise


# ============================================================================
# 2. TIME SHIFT
# ============================================================================
# Tool firing time uncertainty: her depth sample'da transmitter'ın
# tetiklendiği an birkaç micro-second oynayabilir.
# Window shape (depth, time) - time eksende kaydırma yapılır.

def time_shift(window: np.ndarray, max_shift: int = 5,
               rng: np.random.Generator = None) -> np.ndarray:
    """
    Waveform'u time ekseninde ±max_shift sample kaydır.
    Edge'ler 0 padding ile doldurulur (clip + roll değil çünkü physical olarak doğru değil).
    """
    rng = rng or np.random.default_rng()
    shift = int(rng.integers(-max_shift, max_shift + 1))
    if shift == 0:
        return window

    out = np.zeros_like(window)
    if shift > 0:
        # Sağa kaydır: [_,_,_,a,b,c] → [shift_z, _,_,_,a,b]
        out[:, shift:] = window[:, :-shift]
    else:
        out[:, :shift] = window[:, -shift:]
    return out


# ============================================================================
# 3. AMPLITUDE SCALE
# ============================================================================
# Tool calibration drift: kullanım süresince transmitter çıkışı %20'ye kadar
# farkedebilir. Modeli amplitude'a göre değil, RELATIVE pattern'e göre öğretir.

def amplitude_scale(window: np.ndarray, scale_range: Tuple[float, float] = (0.8, 1.2),
                    rng: np.random.Generator = None) -> np.ndarray:
    """Tüm window'u tek bir scale faktörüyle çarp."""
    rng = rng or np.random.default_rng()
    scale = rng.uniform(*scale_range)
    return window * scale


# ============================================================================
# 4. DEPTH FLIP
# ============================================================================
# CBL bond pattern fiziksel olarak yön-bağımsız: bir window'u depth ekseninde
# ters çevirmek sonik olarak hâlâ aynı bond kalitesini gösterir.
# Bu çok ucuz ve etkili bir augmentation.

def depth_flip(window: np.ndarray) -> np.ndarray:
    """Window'u depth ekseninde (axis=0) ters çevir."""
    return window[::-1, :].copy()


# ============================================================================
# 5. MIXUP (within-class)
# ============================================================================
# İki aynı-sınıf window'u doğrusal interpolate et. Mixup standart bir teknik
# (Zhang et al. 2018). Aynı sınıftan örnekler arası interpolasyon decision
# boundary'i yumuşatır, model regularize olur.
# DİKKAT: sadece aynı sınıflar arasında - cross-class mixup label belirsizliği yaratır.

def mixup_within_class(window: np.ndarray, partner: np.ndarray,
                       alpha: float = 0.2,
                       rng: np.random.Generator = None) -> np.ndarray:
    """
    Beta dağılımından lambda çek, iki window'u karıştır:
      out = lambda * window + (1-lambda) * partner
    alpha küçükse karışım hafif (window'a yakın), büyükse 50/50.
    """
    rng = rng or np.random.default_rng()
    lam = rng.beta(alpha, alpha) if alpha > 0 else 1.0
    # lambda'yı 0.5'in üstünde tut → ana window daha baskın
    lam = max(lam, 1.0 - lam)
    return lam * window + (1.0 - lam) * partner


# ============================================================================
# 6. CUTOUT
# ============================================================================
# Random bir time/depth bandını sıfırla. Dropout'un input-side karşılığı.
# Tool malfunction veya kayıp sample simülasyonu da diyebiliriz.

def cutout(window: np.ndarray, max_depth_cut: int = 15, max_time_cut: int = 30,
           rng: np.random.Generator = None) -> np.ndarray:
    """
    Rastgele bir dikdörtgen bölgeyi 0 yap. Bölge boyutu 1..max_cut arası.
    """
    rng = rng or np.random.default_rng()
    out = window.copy()
    H, W = out.shape

    dh = int(rng.integers(1, max_depth_cut + 1))
    dw = int(rng.integers(1, max_time_cut + 1))
    h0 = int(rng.integers(0, max(H - dh + 1, 1)))
    w0 = int(rng.integers(0, max(W - dw + 1, 1)))
    out[h0:h0 + dh, w0:w0 + dw] = 0.0
    return out


# ============================================================================
# AUGMENTATION PIPELINE
# ============================================================================

DEFAULT_AUG_CONFIG = {
    "noise":      {"p": 0.5, "snr_db_range": (20.0, 35.0)},
    "time_shift": {"p": 0.4, "max_shift": 5},
    "amp_scale":  {"p": 0.5, "scale_range": (0.8, 1.2)},
    "depth_flip": {"p": 0.3},
    "mixup":      {"p": 0.3, "alpha": 0.2},
    "cutout":     {"p": 0.3, "max_depth_cut": 15, "max_time_cut": 30},
}


def apply_augmentations(window: np.ndarray,
                        partner: np.ndarray = None,
                        config: dict = None,
                        rng: np.random.Generator = None) -> np.ndarray:
    """
    Bir window'a augmentation pipeline'ını uygula.
    Her augmentation kendi 'p' olasılığına göre uygulanır.

    Sıra önemli: önce fiziksel modifikasyonlar (mixup, flip), sonra
    sinyal-seviye modifikasyonları (noise, shift, scale), en son cutout.
    """
    rng = rng or np.random.default_rng()
    cfg = config or DEFAULT_AUG_CONFIG
    out = window.astype(np.float32, copy=True)

    # 1. Mixup (partner varsa)
    mixup_cfg = cfg.get("mixup", {})
    if partner is not None and mixup_cfg.get("p", 0) > 0 and rng.random() < mixup_cfg["p"]:
        out = mixup_within_class(out, partner.astype(np.float32),
                                 alpha=mixup_cfg.get("alpha", 0.2), rng=rng)

    # 2. Depth flip
    flip_cfg = cfg.get("depth_flip", {})
    if flip_cfg.get("p", 0) > 0 and rng.random() < flip_cfg["p"]:
        out = depth_flip(out)

    # 3. Amplitude scale
    amp_cfg = cfg.get("amp_scale", {})
    if amp_cfg.get("p", 0) > 0 and rng.random() < amp_cfg["p"]:
        out = amplitude_scale(out, scale_range=amp_cfg.get("scale_range", (0.8, 1.2)), rng=rng)

    # 4. Time shift
    ts_cfg = cfg.get("time_shift", {})
    if ts_cfg.get("p", 0) > 0 and rng.random() < ts_cfg["p"]:
        out = time_shift(out, max_shift=ts_cfg.get("max_shift", 5), rng=rng)

    # 5. Gaussian noise
    noise_cfg = cfg.get("noise", {})
    if noise_cfg.get("p", 0) > 0 and rng.random() < noise_cfg["p"]:
        out = gaussian_noise(out, snr_db_range=noise_cfg.get("snr_db_range", (20.0, 35.0)), rng=rng)

    # 6. Cutout (en son - diğer augmentation'ların etkisini gizlememek için)
    cut_cfg = cfg.get("cutout", {})
    if cut_cfg.get("p", 0) > 0 and rng.random() < cut_cfg["p"]:
        out = cutout(out, max_depth_cut=cut_cfg.get("max_depth_cut", 15),
                     max_time_cut=cut_cfg.get("max_time_cut", 30), rng=rng)

    return out


# ============================================================================
# OFFLINE AUGMENTATION (her sınıf için orantılı çoğaltma)
# ============================================================================

def compute_proportional_targets(class_counts: dict,
                                 multiplier: float = 5.0,
                                 min_class_target: int = 500) -> dict:
    """
    Her sınıf için hedef örnek sayısı hesapla.
    Strateji: orantılı çoğaltma, azınlık sınıfa daha fazla boost.

    Yöntem:
    - max_class = en büyük sınıfın boyutu
    - max sınıfın hedefi: max_class * multiplier ** 0.4  (hafif boost)
    - min sınıfın hedefi: max(min_class_target, max_target / 4)
    - ortadakiler: log-doğrusal interpolasyon

    Bu, azınlık sınıfa orantısız fazla çoğaltma yapıp aynı 32 örnekten
    binlerce neredeyse-kopya üretmeyi engeller.
    """
    if not class_counts:
        return {}

    sorted_classes = sorted(class_counts.items(), key=lambda x: -x[1])
    max_count = sorted_classes[0][1]
    target_max = int(max_count * multiplier ** 0.4)   # ~max * 1.9 for mult=5

    # Min sınıfın hedefi: en azından min_class_target, en fazla max'in 1/3'ü
    target_min_floor = min(target_max // 3, min_class_target)

    targets = {}
    for cls, cnt in class_counts.items():
        if cnt == max_count:
            targets[cls] = target_max
        elif cnt <= max_count * 0.1:  # küçük azınlık (Partial gibi)
            # Azınlık için: hedefi target_min_floor, ama orijinalin en az 10x'i
            targets[cls] = max(target_min_floor, cnt * 10)
        else:
            # Orta sınıflar: orantılı boost
            ratio = cnt / max_count
            target = int(target_max * (ratio ** 0.7))
            targets[cls] = max(target, cnt)  # orijinalden az olmasın

    return targets


def offline_augment_dataset(windows: np.ndarray, labels: np.ndarray,
                            class_targets: dict,
                            aug_config: dict = None,
                            seed: int = 42,
                            verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sınıf bazında hedef sayıya kadar çoğaltma yap.

    Return:
      windows_aug: (M, H, W) - orijinaller + augment edilmiş
      labels_aug: (M,)
      is_augmented: (M,) bool - True ise augment, False ise orijinal
    """
    rng = np.random.default_rng(seed)
    cfg = aug_config or DEFAULT_AUG_CONFIG

    all_windows = []
    all_labels = []
    all_flags = []   # False = orijinal, True = augment

    for cls in sorted(class_targets.keys()):
        mask = labels == cls
        orig_windows = windows[mask]
        orig_count = len(orig_windows)
        target = class_targets[cls]

        if orig_count == 0:
            continue

        # Önce orijinalleri al
        all_windows.append(orig_windows)
        all_labels.append(np.full(orig_count, cls, dtype=labels.dtype))
        all_flags.append(np.zeros(orig_count, dtype=bool))

        # Eksik kalan kadar augment üret
        n_needed = max(target - orig_count, 0)
        if n_needed > 0:
            aug_windows = np.empty((n_needed, *orig_windows.shape[1:]), dtype=np.float32)
            for i in range(n_needed):
                idx = int(rng.integers(0, orig_count))
                # Mixup için aynı sınıftan farklı bir partner
                if orig_count > 1:
                    partner_idx = int(rng.integers(0, orig_count))
                    while partner_idx == idx:
                        partner_idx = int(rng.integers(0, orig_count))
                    partner = orig_windows[partner_idx]
                else:
                    partner = None

                aug_windows[i] = apply_augmentations(
                    orig_windows[idx], partner=partner, config=cfg, rng=rng
                )

            all_windows.append(aug_windows)
            all_labels.append(np.full(n_needed, cls, dtype=labels.dtype))
            all_flags.append(np.ones(n_needed, dtype=bool))

            if verbose:
                print(f"  class {cls}: {orig_count} → {orig_count + n_needed}  "
                      f"(+{n_needed} augment)")
        else:
            if verbose:
                print(f"  class {cls}: {orig_count} (hedef {target}, augment gerekmedi)")

    return (np.concatenate(all_windows, axis=0),
            np.concatenate(all_labels, axis=0),
            np.concatenate(all_flags, axis=0))
