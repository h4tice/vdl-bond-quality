"""
REVISION-2: CROSS-VALIDATED EXPERIMENTS (FORGE 78B-32)
======================================================
Hakem cevapları için ana deney matrisi. Kaldığı yerden devam eder: her koşu
kendi dosyasına yazılır, var olanlar atlanır.

Protokoller:
  blocked     Sızıntısız purged blocked K-fold (10 blok, 5 kat), önerilen protokol
  random      Tabakalı rastgele 70/15/15 (makaledeki random split; sızıntı referansı)
  stratdepth  Makaledeki stratified-depth split (Hakem-2 madde 1'in eleştirdiği)
  lc          Öğrenme eğrisi: blocked katlarında train'in %25 / %50'si (Hakem-1 madde 3)

Her (protokol, seed, kat) için 4 DL modeli × augmentation {0,1} ve
4 klasik baseline koşar. Normalizasyon her zaman yalnızca train'den.

KULLANIM:
    python scripts/rev2_run_cv.py                          # blocked + random + stratdepth
    python scripts/rev2_run_cv.py --protocols blocked
    python scripts/rev2_run_cv.py --protocols lc --seeds 42 123 456

ÇIKTI:
    outputs/revision2/runs/<protokol>/<yöntem>_aug<a>_s<seed>_f<kat>.npz
    outputs/revision2/runs/<protokol>/splits_s<seed>.json
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from config import (DATA_PROCESSED_DIR, OUTPUT_DIR, TEST_SIZE, VAL_SIZE,
                    NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
                    EARLY_STOPPING_PATIENCE, AUG_CONFIG,
                    AUGMENTATION_MULTIPLIER, AUGMENTATION_MIN_CLASS_TARGET)
from src.blocked_cv import purged_blocked_kfold, overlap_fraction, min_gap
from src.splits import random_split, stratified_depth_split
from src.rev2_engine import (fit_predict, metrics, train_only_normalize, augment_train,
                             engineered_features, cbl_proxy_features)
from src.utils import print_header, print_section, print_info, print_success, get_device

REV2_DIR = OUTPUT_DIR / "revision2"
RUNS_DIR = REV2_DIR / "runs"

MODELS = ["attention", "cnn", "resnet1d", "lstm"]
SEEDS = [42, 123, 456, 789, 2024]
N_BLOCKS, N_FOLDS = 10, 5
LC_FRACTIONS = [0.25, 0.5]


def load_well(well_id="forge_78b32"):
    d = np.load(DATA_PROCESSED_DIR / f"rev2_{well_id}.npz")
    valid = d["labels_3"] >= 0
    return {k: d[k][valid] for k in ("windows", "depths", "starts", "cbl", "bi", "labels_3")}


def build_splits(protocol, data, seed, window_size):
    """Return: list of dict(fold, train, val, test) + tanı bilgisi."""
    L, starts = data["labels_3"], data["starts"]
    if protocol in ("blocked", "lc"):
        return purged_blocked_kfold(starts, L, window_size, n_blocks=N_BLOCKS,
                                    n_folds=N_FOLDS, seed=seed)
    if protocol == "random":
        s = random_split(L, test_size=TEST_SIZE, val_size=VAL_SIZE, seed=seed)
    elif protocol == "stratdepth":
        s = stratified_depth_split(L, data["depths"], test_size=TEST_SIZE,
                                   val_size=VAL_SIZE, n_depth_bins=5, seed=seed)
    else:
        raise ValueError(protocol)
    diag = {
        "fold": 0,
        "n_train": len(s["train"]), "n_val": len(s["val"]), "n_test": len(s["test"]),
        "min_gap_train_test": min_gap(starts[s["train"]], starts[s["test"]]),
        "frac_test_sharing_rows_with_train": overlap_fraction(
            starts[s["train"]], starts[s["test"]], window_size),
        "test_class_counts": np.bincount(L[s["test"]], minlength=3).tolist(),
    }
    return [{"fold": 0, **s, "diagnostics": diag}]


def run_baselines(feats_eng, feats_proxy, L, split, seed):
    tr, te = split["train"], split["test"]
    candidates = {
        "majority": (DummyClassifier(strategy="most_frequent"), feats_eng),
        "engineered_lr": (make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=seed)), feats_eng),
        "engineered_rf": (RandomForestClassifier(
            n_estimators=300, max_depth=10, class_weight="balanced",
            random_state=seed, n_jobs=-1), feats_eng),
        # Etiket döngüselliği testi: orta derinlikteki 2 dalga formu skaleri
        # üzerinde 3 derinlikli karar ağacı ≈ eşik kuralı
        "cbl_proxy_tree": (DecisionTreeClassifier(
            max_depth=3, class_weight="balanced", random_state=seed), feats_proxy),
    }
    out = {}
    for name, (clf, X) in candidates.items():
        clf.fit(X[tr], L[tr])
        # predict_proba sütunları train'de görülen sınıflara göre → 3 sınıfa genişlet
        full = np.zeros((len(te), 3))
        full[:, clf.classes_] = clf.predict_proba(X[te])
        out[name] = full
    return out


def save_run(path, test_idx, y_true, probs, info, extra=None):
    m = metrics(y_true, probs.argmax(1))
    np.savez_compressed(path, test_idx=test_idx, y_true=y_true, probs=probs.astype(np.float32),
                        info=json.dumps({**info, **(extra or {}), "metrics": m}, default=float))
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--protocols", nargs="+", default=["blocked", "random", "stratdepth"])
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--aug", nargs="+", type=int, default=[0, 1])
    p.add_argument("--no-baselines", action="store_true")
    args = p.parse_args()

    device = get_device()
    print_header("REVISION-2 CV EXPERIMENTS")
    print_info(f"Device: {device} | protocols={args.protocols} | seeds={args.seeds}")

    data = load_well()
    W, L, starts = data["windows"], data["labels_3"], data["starts"]
    window_size = W.shape[1]
    print_info(f"Windows: {W.shape}, sınıflar: {np.bincount(L).tolist()}")
    feats_eng = engineered_features(W)
    feats_proxy = cbl_proxy_features(W)

    for protocol in args.protocols:
        out_dir = RUNS_DIR / protocol
        out_dir.mkdir(parents=True, exist_ok=True)
        aug_states = [1] if protocol == "lc" else args.aug
        fractions = LC_FRACTIONS if protocol == "lc" else [1.0]

        for seed in args.seeds:
            splits = build_splits(protocol, data, seed, window_size)
            with open(out_dir / f"splits_s{seed}.json", "w") as f:
                json.dump([s["diagnostics"] for s in splits], f, indent=2, default=float)

            for split in splits:
                fold = split["fold"]
                print_section(f"{protocol} | seed {seed} | fold {fold} | "
                              f"train {len(split['train'])} val {len(split['val'])} "
                              f"test {len(split['test'])}")

                if not args.no_baselines and protocol != "lc":
                    bpath = out_dir / f"majority_aug0_s{seed}_f{fold}.npz"
                    if not bpath.exists():
                        for name, pr in run_baselines(feats_eng, feats_proxy, L, split, seed).items():
                            m = save_run(out_dir / f"{name}_aug0_s{seed}_f{fold}.npz",
                                         split["test"], L[split["test"]], pr,
                                         {"method": name, "family": "baseline"})
                            print(f"    {name:16s} f1_macro={m['f1_macro']:.4f} "
                                  f"f1_w={m['f1_weighted']:.4f}")

                Wtr, Wva, Wte = train_only_normalize(W[split["train"]], W[split["val"]],
                                                     W[split["test"]])
                Ltr, Lva = L[split["train"]], L[split["val"]]

                for frac in fractions:
                    tag = "" if frac == 1.0 else f"_frac{frac}"
                    if frac < 1.0:
                        rng = np.random.default_rng(seed + fold)
                        sub = np.sort(rng.choice(len(Ltr), int(len(Ltr) * frac), replace=False))
                        Wtr_f, Ltr_f = Wtr[sub], Ltr[sub]
                    else:
                        Wtr_f, Ltr_f = Wtr, Ltr

                    for aug in aug_states:
                        if aug:
                            Wtr_a, Ltr_a, is_aug = augment_train(
                                Wtr_f, Ltr_f, AUG_CONFIG, AUGMENTATION_MULTIPLIER,
                                AUGMENTATION_MIN_CLASS_TARGET, seed)
                        else:
                            Wtr_a, Ltr_a = Wtr_f, Ltr_f

                        for model_name in args.models:
                            path = out_dir / f"{model_name}{tag}_aug{aug}_s{seed}_f{fold}.npz"
                            if path.exists():
                                continue
                            probs, info = fit_predict(
                                model_name, Wtr_a, Ltr_a, Wva, Lva, {"test": Wte},
                                seed=seed, device=device, num_epochs=NUM_EPOCHS,
                                batch_size=BATCH_SIZE, lr=LEARNING_RATE,
                                weight_decay=WEIGHT_DECAY, patience=EARLY_STOPPING_PATIENCE)
                            m = save_run(path, split["test"], L[split["test"]], probs["test"],
                                         info, {"method": model_name, "family": "deep_learning",
                                                "aug": aug, "train_fraction": frac,
                                                "n_train_original": int(len(Ltr_f)),
                                                "n_train_after_aug": int(len(Ltr_a))})
                            print(f"    {model_name:9s}{tag} aug={aug} f1_macro={m['f1_macro']:.4f} "
                                  f"f1_w={m['f1_weighted']:.4f} bal={m['balanced_accuracy']:.4f} "
                                  f"({info['epochs_ran']} ep, {info['train_time_s']:.0f}s)")
                        if aug:
                            del Wtr_a
                torch.cuda.empty_cache()

    print_success("Tamamlandı.")


if __name__ == "__main__":
    main()
