"""
LABELING (extended)
==================
Eski labeling.py ile aynı, ek olarak filter_valid çok-argüman destekli.
"""

import numpy as np


def label_3class(cbl: np.ndarray, bi: np.ndarray, thresholds: dict) -> np.ndarray:
    cbl = np.asarray(cbl, dtype=float)
    bi = np.asarray(bi, dtype=float)
    labels = np.full(len(cbl), -1, dtype=np.int64)
    valid = np.isfinite(cbl) & np.isfinite(bi)

    good = valid & (cbl < thresholds["cbl_good"]) & (bi > thresholds["bi_good"])
    poor = valid & ((cbl > thresholds["cbl_poor"]) | (bi < thresholds["bi_poor"])) & (~good)
    partial = valid & (~good) & (~poor)

    labels[good] = 0
    labels[partial] = 1
    labels[poor] = 2
    return labels


def label_binary(labels_3: np.ndarray) -> np.ndarray:
    out = labels_3.copy()
    mask_valid = out >= 0
    out[mask_valid & (out > 0)] = 1
    return out


def label_distribution(labels: np.ndarray, class_names: list) -> dict:
    out = {"total": int(len(labels)), "invalid": int((labels == -1).sum())}
    for i, name in enumerate(class_names):
        cnt = int((labels == i).sum())
        pct = 100 * cnt / len(labels) if len(labels) else 0.0
        out[name] = {"count": cnt, "percent": round(pct, 2)}
    return out


def filter_valid(labels: np.ndarray, *arrays):
    """Label geçerli (>=0) olan örnekleri filtrele. Çok-argüman."""
    mask = labels >= 0
    return (labels[mask],) + tuple(a[mask] for a in arrays)
