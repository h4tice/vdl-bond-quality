"""
PURGED BLOCKED CROSS-VALIDATION
===============================
Örtüşen window'lar için sızıntısız derinlik-blok çapraz doğrulama
(Hakem-2 madde 1, Hakem-1 madde 2).

Önceki stratified_depth_split bant içinde rastgele bölüyordu; %95 örtüşen
komşu window'lar farklı kümelere düşüyordu. blocked_purge_split ise purge
gap'i derinlik biriminde (0.1 in) 100 alıyordu ≈ 1.7 örnek, pencere boyunun
çok altında.

Burada:
  1. Window'lar başlangıç örneğine (starts) göre sıralanır ve n_blocks
     ardışık, eşit sayılı bloğa bölünür.
  2. Bloklar her seed için rastgele n_folds kata dağıtılır (tekrarlı CV).
  3. Her katta: test = katın blokları; kalan her bloğun rastgele bir
     ucundan bitişik val_frac'lık dilim validation, geri kalanı train.
     (Sınıflar derinliğe göre güçlü ayrıştığı için tek bloktan alınan
     validation çoğu zaman tek sınıflı oluyor; dilimler train dağılımını
     temsil eder.)
  4. Purge: test ya da validation window'larından herhangi biriyle tek bir
     derinlik satırı bile paylaşan train window'ları atılır
     (|start_i - start_j| < window_size). Aynısı validation ile test arasında
     da uygulanır. Böylece kümeler arasında ortak ham veri kalmaz.
"""

import numpy as np


def _overlaps(starts_a, starts_b, window_size):
    """starts_a'daki her window, starts_b'deki herhangi bir window ile örtüşüyor mu?"""
    if len(starts_b) == 0:
        return np.zeros(len(starts_a), dtype=bool)
    sb = np.sort(starts_b)
    pos = np.searchsorted(sb, starts_a)
    left = sb[np.clip(pos - 1, 0, len(sb) - 1)]
    right = sb[np.clip(pos, 0, len(sb) - 1)]
    dist = np.minimum(np.abs(starts_a - left), np.abs(starts_a - right))
    return dist < window_size


def min_gap(starts_a, starts_b):
    """İki küme arasındaki en küçük başlangıç farkı (örnek)."""
    if len(starts_a) == 0 or len(starts_b) == 0:
        return np.inf
    sb = np.sort(starts_b)
    pos = np.searchsorted(sb, starts_a)
    left = sb[np.clip(pos - 1, 0, len(sb) - 1)]
    right = sb[np.clip(pos, 0, len(sb) - 1)]
    return int(np.min(np.minimum(np.abs(starts_a - left), np.abs(starts_a - right))))


def make_blocks(starts, n_blocks):
    """Derinlik sırasına göre ardışık, eşit sayılı bloklar. Return: block_id (N,)."""
    order = np.argsort(starts)
    block_id = np.empty(len(starts), dtype=int)
    for b, chunk in enumerate(np.array_split(order, n_blocks)):
        block_id[chunk] = b
    return block_id


def purged_blocked_kfold(starts, labels, window_size, n_blocks=10, n_folds=5,
                         val_frac=0.15, seed=42):
    """
    Return: list of dict(fold, train, val, test, diagnostics).
    Her window tam olarak bir katta test edilir (out-of-fold tahmin için).
    """
    starts = np.asarray(starts, dtype=np.int64)
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    block_id = make_blocks(starts, n_blocks)

    perm = rng.permutation(n_blocks)
    fold_blocks = np.array_split(perm, n_folds)

    folds = []
    for f, test_blocks in enumerate(fold_blocks):
        rest = [b for b in perm if b not in set(test_blocks)]
        test = np.where(np.isin(block_id, test_blocks))[0]

        train_parts, val_parts = [], []
        for b in rest:
            idx = np.where(block_id == b)[0]
            idx = idx[np.argsort(starts[idx])]
            n_val = int(round(len(idx) * val_frac))
            if rng.random() < 0.5:
                val_parts.append(idx[:n_val]); train_parts.append(idx[n_val:])
            else:
                val_parts.append(idx[-n_val:]); train_parts.append(idx[:-n_val])
        val = np.sort(np.concatenate(val_parts))
        train = np.sort(np.concatenate(train_parts))

        # Purge
        val = val[~_overlaps(starts[val], starts[test], window_size)]
        held_out = np.concatenate([starts[test], starts[val]])
        n_train_before = len(train)
        train = train[~_overlaps(starts[train], held_out, window_size)]

        diag = {
            "fold": f,
            "test_blocks": sorted(int(b) for b in test_blocks),
            "n_train": int(len(train)), "n_val": int(len(val)), "n_test": int(len(test)),
            "n_purged_train": int(n_train_before - len(train)),
            "min_gap_train_test": min_gap(starts[train], starts[test]),
            "min_gap_train_val": min_gap(starts[train], starts[val]),
            "min_gap_val_test": min_gap(starts[val], starts[test]),
            "test_class_counts": np.bincount(labels[test], minlength=3).tolist(),
            "train_class_counts": np.bincount(labels[train], minlength=3).tolist(),
            "val_class_counts": np.bincount(labels[val], minlength=3).tolist(),
        }
        assert diag["min_gap_train_test"] >= window_size
        assert diag["min_gap_train_val"] >= window_size
        assert diag["min_gap_val_test"] >= window_size
        folds.append({"fold": f, "train": train, "val": val, "test": test,
                      "diagnostics": diag})
    return folds


def purged_train_val(starts, window_size, n_blocks=10, val_frac=0.15, seed=42):
    """
    Tüm kuyu eğitime girdiğinde (cross-well): her bloğun rastgele bir ucundan
    bitişik val_frac'lık dilim validation, örtüşen train window'ları atılır.
    """
    starts = np.asarray(starts, dtype=np.int64)
    rng = np.random.default_rng(seed)
    block_id = make_blocks(starts, n_blocks)
    train_parts, val_parts = [], []
    for b in range(n_blocks):
        idx = np.where(block_id == b)[0]
        idx = idx[np.argsort(starts[idx])]
        n_val = int(round(len(idx) * val_frac))
        if rng.random() < 0.5:
            val_parts.append(idx[:n_val]); train_parts.append(idx[n_val:])
        else:
            val_parts.append(idx[-n_val:]); train_parts.append(idx[:-n_val])
    val = np.sort(np.concatenate(val_parts))
    train = np.sort(np.concatenate(train_parts))
    train = train[~_overlaps(starts[train], starts[val], window_size)]
    return train, val


def overlap_fraction(starts_train, starts_test, window_size):
    """Test window'larının kaçta kaçı en az bir train window'u ile satır paylaşıyor."""
    return float(_overlaps(np.asarray(starts_test), np.asarray(starts_train),
                           window_size).mean())
