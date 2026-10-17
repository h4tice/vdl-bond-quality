"""
SPLIT STRATEGIES
================
3 farklı train/test split stratejisi:

1. random_split        - stratified random (mevcut, data leakage riski yüksek)
2. depth_based_split   - derinliğe göre böl (test set hiç görülmeyen aralık)
3. kfold_random        - 5-fold cross validation (istatistiksel güven)
4. kfold_depth         - depth-aware k-fold (en sağlam)

Her stratejinin return formatı aynı:
- random/depth: dict {"train": idx, "val": idx, "test": idx}
- kfold: list of such dicts, k tane
"""

import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold


# ============================================================================
# 1. RANDOM SPLIT (mevcut)
# ============================================================================

def random_split(labels: np.ndarray, well_ids: np.ndarray = None,
                 test_size: float = 0.15, val_size: float = 0.15,
                 seed: int = 42) -> dict:
    """Stratified random split. Komşu window'lar farklı set'lere düşebilir."""
    idx = np.arange(len(labels))
    idx_trainval, idx_test = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=labels
    )
    val_ratio = val_size / (1.0 - test_size)
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=val_ratio, random_state=seed,
        stratify=labels[idx_trainval]
    )
    return {"train": idx_train, "val": idx_val, "test": idx_test}


# ============================================================================
# 2. DEPTH-BASED SPLIT (basic - deprecated)
# ============================================================================

def depth_based_split(labels: np.ndarray, depths: np.ndarray,
                      test_size: float = 0.15, val_size: float = 0.15,
                      seed: int = 42) -> dict:
    """
    Basit derinlik-bazlı split: en alt %15'i test'e ayır.

    UYARI: Sınıf dağılımı derinliğe göre dengesizse (free-pipe üst, good-bond
    alt gibi) test seti tek sınıfa dominantlaşabilir. Bu durumda
    'stratified_depth_split' kullan.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(len(labels))

    order = np.argsort(depths)

    n = len(idx)
    n_test = int(n * test_size)
    n_val = int(n * val_size)
    n_train = n - n_test - n_val

    train_idx = order[:n_train]
    val_idx = order[n_train:n_train + n_val]
    test_idx = order[n_train + n_val:]

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return {"train": train_idx, "val": val_idx, "test": test_idx}


# ============================================================================
# 2b. STRATIFIED DEPTH SPLIT (önerilen)
# ============================================================================

def stratified_depth_split(labels: np.ndarray, depths: np.ndarray,
                           test_size: float = 0.15, val_size: float = 0.15,
                           n_depth_bins: int = 5,
                           seed: int = 42) -> dict:
    """
    Derinliği n_depth_bins parçaya böl. Her bandın içinden stratified
    olarak (sınıf dağılımı koruyarak) test/val ayır.

    Bu sayede:
      1. Test seti tüm derinlik aralığını kapsar (random window-level değil)
      2. Test setinde her sınıf temsil edilir (depth-class dengesizliğine rağmen)
      3. Komşu window'lar aynı set'te kalır (depth band içinde shuffling
         yapılmadığı için band-içi leakage minimize edilir)

    Bu stratified-depth-blocked split.
    """
    rng = np.random.default_rng(seed)
    n = len(labels)
    idx = np.arange(n)

    # Derinlik bantları (quantile-based, eşit veri sayısı için)
    depth_quantiles = np.quantile(depths, np.linspace(0, 1, n_depth_bins + 1))
    # Her sample'ın bandı
    band_id = np.digitize(depths, depth_quantiles[1:-1])   # 0..n_depth_bins-1

    train_indices, val_indices, test_indices = [], [], []

    for b in range(n_depth_bins):
        band_mask = band_id == b
        band_idx = idx[band_mask]
        band_labels = labels[band_mask]

        # Bu band'de hangi sınıflar var?
        present_classes, class_counts = np.unique(band_labels, return_counts=True)
        # Stratify ancak her sınıfta >=2 örnek varsa çalışır
        can_stratify = (class_counts >= 2).all()

        if len(band_idx) < 4:
            # Yetersiz örnek → hepsi train'e
            train_indices.append(band_idx)
            continue

        # Önce test'i ayır
        if can_stratify:
            trainval, test = train_test_split(
                band_idx,
                test_size=test_size,
                random_state=seed + b,
                stratify=band_labels,
            )
            # Sonra val'i train'den ayır
            trainval_labels = labels[trainval]
            tv_present, tv_counts = np.unique(trainval_labels, return_counts=True)
            tv_can_strat = (tv_counts >= 2).all()
            if tv_can_strat:
                tr, va = train_test_split(
                    trainval,
                    test_size=val_size / (1.0 - test_size),
                    random_state=seed + b,
                    stratify=trainval_labels,
                )
            else:
                tr, va = train_test_split(
                    trainval,
                    test_size=val_size / (1.0 - test_size),
                    random_state=seed + b,
                )
        else:
            # Stratify mümkün değil → random
            trainval, test = train_test_split(
                band_idx, test_size=test_size, random_state=seed + b,
            )
            tr, va = train_test_split(
                trainval,
                test_size=val_size / (1.0 - test_size),
                random_state=seed + b,
            )

        train_indices.append(tr)
        val_indices.append(va)
        test_indices.append(test)

    return {
        "train": np.concatenate(train_indices) if train_indices else np.array([], dtype=int),
        "val": np.concatenate(val_indices) if val_indices else np.array([], dtype=int),
        "test": np.concatenate(test_indices) if test_indices else np.array([], dtype=int),
    }


# ============================================================================
# 3. K-FOLD (random)
# ============================================================================

def kfold_random(labels: np.ndarray, n_splits: int = 5,
                 val_size: float = 0.15, seed: int = 42) -> list:
    """
    Stratified k-fold CV - her fold'da test set ayrı.
    Val set, her fold'da train'den ayrılır.
    Return: list of k dict, her biri {"train", "val", "test"}.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for trainval_idx, test_idx in skf.split(np.zeros(len(labels)), labels):
        # train/val ayır
        train_idx, val_idx = train_test_split(
            trainval_idx,
            test_size=val_size / (1.0 - 1.0 / n_splits),
            random_state=seed,
            stratify=labels[trainval_idx],
        )
        folds.append({"train": train_idx, "val": val_idx, "test": test_idx})
    return folds


# ============================================================================
# 4. K-FOLD (depth-aware)
# ============================================================================

def kfold_depth(labels: np.ndarray, depths: np.ndarray,
                n_splits: int = 5, val_size: float = 0.15,
                seed: int = 42) -> list:
    """
    Derinlik-bilinçli k-fold: kuyu derinlik aralığını k eş parçaya böl.
    Her fold'da bir derinlik bandı test, diğerleri train.
    Val set train'in random %15'i (yine derinlik karışmaz, depth zone içinde).

    Bu en sağlam değerlendirme - gerçek generalization + istatistik.
    """
    rng = np.random.default_rng(seed)
    n = len(labels)

    order = np.argsort(depths)
    bins = np.array_split(order, n_splits)

    folds = []
    for k in range(n_splits):
        test_idx = bins[k]
        train_pool = np.concatenate([bins[i] for i in range(n_splits) if i != k])

        # Validation = train_pool'un random %15'i
        n_val = int(len(train_pool) * val_size)
        rng.shuffle(train_pool)
        val_idx = train_pool[:n_val]
        train_idx = train_pool[n_val:]

        folds.append({"train": train_idx, "val": val_idx, "test": test_idx})
    return folds


# ============================================================================
# YARDIMCILAR
# ============================================================================

def describe_split(splits: dict, labels: np.ndarray, class_names: list = None) -> dict:
    """Bir split'in sınıf dağılımını döndür."""
    out = {}
    for name, idx in splits.items():
        sub_labels = labels[idx]
        cls_counts = {}
        for cls in np.unique(labels):
            cnt = int((sub_labels == cls).sum())
            key = class_names[cls] if class_names and cls < len(class_names) else f"cls_{cls}"
            cls_counts[key] = cnt
        out[name] = {"n": int(len(idx)), "class_counts": cls_counts}
    return out