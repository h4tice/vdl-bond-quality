"""
REVISION-2: BUILD PER-WELL BUNDLES
==================================
Her aktif kuyu için ayrı bir .npz üretir (cross-well testinde kuyular ayrı
tutulmalı). Önceki bundle'lardan farkları:

- DLIS kayıp değer kodu (-999.25) NaN yapılır → bu window'lar etiketsiz kalır
  (eski bundle'da Poor olarak etiketleniyordu)
- Her window için başlangıç örnek indeksi (starts) saklanır → purge gap
  derinlik birimi yerine örnek sayısıyla, kesin olarak uygulanır
- 16B-32 doğru frame/kanallardan (30B, WF1, DCBL/DBI) ve 2'ye seyreltilerek
  okunur

KULLANIM:
    python scripts/rev2_build_data.py
    python scripts/rev2_build_data.py --window 100 --step 5

ÇIKTI:
    data/processed/rev2_<well_id>.npz
    data/processed/rev2_summary.json
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from config import (WELLS, DATA_RAW_DIR, DATA_PROCESSED_DIR,
                    LABEL_THRESHOLDS, CLASS_NAMES_3)
from src.multi_well_loader import WellLoader, locate_dlis_for_well
from src.labeling import label_3class, label_distribution
from src.utils import print_header, print_section, print_info, print_success, print_warning


def count_runs(labels_sorted, cls):
    """Derinliğe göre sıralı etiketlerde cls sınıfının ardışık bölge sayısı."""
    is_c = labels_sorted == cls
    return int(np.sum(is_c[1:] & ~is_c[:-1]) + int(is_c[0])) if len(is_c) else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--step", type=int, default=5)
    args = parser.parse_args()

    print_header("REVISION-2: PER-WELL BUNDLES")
    summary = {"window_size": args.window, "step": args.step,
               "label_thresholds": LABEL_THRESHOLDS, "wells": {}}

    for well_id, cfg in WELLS.items():
        if not cfg.get("enabled", False):
            continue
        print_section(f"{cfg['name']} ({well_id})")
        path = locate_dlis_for_well(well_id, cfg, DATA_RAW_DIR)
        if path is None:
            print_warning(f"DLIS bulunamadı, atlanıyor: {well_id}")
            continue

        loader = WellLoader(well_id, cfg, path).load()
        chunk = loader.make_windows(window_size=args.window, step=args.step)
        labels = label_3class(chunk["cbl"], chunk["bi"], LABEL_THRESHOLDS)
        dist = label_distribution(labels, CLASS_NAMES_3)
        print_info(json.dumps(dist, ensure_ascii=False))

        order = np.argsort(chunk["starts"])
        runs = {name: count_runs(labels[order], i) for i, name in enumerate(CLASS_NAMES_3)}
        print_info(f"Ardışık bölge sayısı: {runs}")

        out = DATA_PROCESSED_DIR / f"rev2_{well_id}.npz"
        np.savez_compressed(
            out,
            windows=chunk["windows"].astype(np.float32),
            depths=chunk["depths"].astype(np.float64),
            starts=chunk["starts"],
            cbl=chunk["cbl"].astype(np.float64),
            bi=chunk["bi"].astype(np.float64),
            labels_3=labels.astype(np.int64),
        )
        print_success(f"{out.name}: {chunk['windows'].shape}  "
                      f"({out.stat().st_size / 1e6:.1f} MB)")

        spacing = float(np.median(np.abs(np.diff(loader.depth))))
        summary["wells"][well_id] = {
            **loader.metadata,
            "n_windows": int(len(labels)),
            "depth_spacing_units": spacing,
            "label_distribution": dist,
            "contiguous_class_zones": runs,
        }

    with open(DATA_PROCESSED_DIR / "rev2_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=float)
    print_success("rev2_summary.json yazıldı")


if __name__ == "__main__":
    main()
