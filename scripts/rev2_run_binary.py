"""
REVISION-2: BINARY TASK (Good vs. Not-Good)
===========================================
Partial sınıfı üç sınıflı görevde öğrenilemediği için (Hakem-1 madde 4,
Hakem-2 madde 3) ek analiz: Partial ve Poor birleştirilip "Not-Good" yapılır
ve modeller doğrudan iki sınıfla eğitilir.

Üç sınıflı deneyle birebir karşılaştırılabilirlik için:
  - aynı purged blocked katlar ve seed'ler (katlar etiketten bağımsız)
  - aynı train-only normalizasyon
  - augmentation üç sınıflı etiketlerle yapılır (aynı sentetik pencereler,
    mixup yalnızca aynı üç-sınıf içinde), ardından etiketler ikiye indirilir
Yani iki deney arasındaki tek fark eğitimdeki etiket uzayıdır.

KULLANIM:
    python scripts/rev2_run_binary.py

ÇIKTI:
    outputs/revision2/runs/binary/<yöntem>_aug<a>_s<seed>_f<kat>.npz
      (y_true: ikili, y_true3: üç sınıflı orijinal etiket, probs: (N, 2))
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
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_recall_fscore_support, confusion_matrix)

from config import (NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
                    EARLY_STOPPING_PATIENCE, AUG_CONFIG,
                    AUGMENTATION_MULTIPLIER, AUGMENTATION_MIN_CLASS_TARGET)
from src.labeling import label_binary
from src.rev2_engine import (fit_predict, train_only_normalize, augment_train,
                             engineered_features, cbl_proxy_features)
from src.utils import print_header, print_section, print_info, print_success, get_device
from scripts.rev2_run_cv import RUNS_DIR, MODELS, SEEDS, load_well, build_splits


def binary_metrics(y_true, y_pred, y_true3=None):
    """Good = 0, Not-Good = 1 (pozitif sınıf)."""
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1],
                                                 zero_division=0)
    out = {
        "n": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, labels=[0, 1], average="weighted",
                                zero_division=0),
        "precision_good": p[0], "recall_good": r[0], "f1_good": f[0],
        "precision_notgood": p[1], "recall_notgood": r[1], "f1_notgood": f[1],
        "cm": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    if y_true3 is not None:
        # Orijinal Partial / Poor pencerelerinin ne kadarı Not-Good bulundu
        for c, name in [(1, "partial"), (2, "poor")]:
            m = y_true3 == c
            out[f"notgood_rate_{name}"] = float(y_pred[m].mean()) if m.any() else np.nan
            out[f"n_{name}"] = int(m.sum())
    return out


def run_baselines(feats_eng, feats_proxy, L, split, seed):
    tr, te = split["train"], split["test"]
    candidates = {
        "majority": (DummyClassifier(strategy="most_frequent"), feats_eng),
        "engineered_lr": (make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=seed)), feats_eng),
        "engineered_rf": (RandomForestClassifier(
            n_estimators=300, max_depth=10, class_weight="balanced",
            random_state=seed, n_jobs=-1), feats_eng),
        "cbl_proxy_tree": (DecisionTreeClassifier(
            max_depth=3, class_weight="balanced", random_state=seed), feats_proxy),
    }
    out = {}
    for name, (clf, X) in candidates.items():
        clf.fit(X[tr], L[tr])
        full = np.zeros((len(te), 2))
        full[:, clf.classes_] = clf.predict_proba(X[te])
        out[name] = full
    return out


def save_run(path, test_idx, y_true, y_true3, probs, info, extra=None):
    m = binary_metrics(y_true, probs.argmax(1), y_true3)
    np.savez_compressed(path, test_idx=test_idx, y_true=y_true, y_true3=y_true3,
                        probs=probs.astype(np.float32),
                        info=json.dumps({**info, **(extra or {}), "metrics": m}, default=float))
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--aug", nargs="+", type=int, default=[0, 1])
    args = p.parse_args()

    device = get_device()
    print_header("REVISION-2 BINARY TASK (Good vs. Not-Good)")
    data = load_well()
    W, L3 = data["windows"], data["labels_3"]
    L2 = label_binary(L3)
    window_size = W.shape[1]
    print_info(f"Windows: {W.shape}, ikili sınıflar: {np.bincount(L2).tolist()}")
    feats_eng = engineered_features(W)
    feats_proxy = cbl_proxy_features(W)
    out_dir = RUNS_DIR / "binary"
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        for split in build_splits("blocked", data, seed, window_size):
            fold = split["fold"]
            tr, va, te = split["train"], split["val"], split["test"]
            print_section(f"binary | seed {seed} | fold {fold} | "
                          f"train {len(tr)} val {len(va)} test {len(te)}")

            if not (out_dir / f"majority_aug0_s{seed}_f{fold}.npz").exists():
                for name, pr in run_baselines(feats_eng, feats_proxy, L2, split, seed).items():
                    m = save_run(out_dir / f"{name}_aug0_s{seed}_f{fold}.npz",
                                 te, L2[te], L3[te], pr, {"method": name, "family": "baseline"})
                    print(f"    {name:16s} f1_macro={m['f1_macro']:.4f} "
                          f"bal={m['balanced_accuracy']:.4f}")

            Wtr, Wva, Wte = train_only_normalize(W[tr], W[va], W[te])
            for aug in args.aug:
                if aug:
                    Wtr_a, L3tr_a, _ = augment_train(Wtr, L3[tr], AUG_CONFIG,
                                                     AUGMENTATION_MULTIPLIER,
                                                     AUGMENTATION_MIN_CLASS_TARGET, seed)
                else:
                    Wtr_a, L3tr_a = Wtr, L3[tr]
                L2tr_a = label_binary(L3tr_a)

                for model_name in args.models:
                    path = out_dir / f"{model_name}_aug{aug}_s{seed}_f{fold}.npz"
                    if path.exists():
                        continue
                    probs, info = fit_predict(
                        model_name, Wtr_a, L2tr_a, Wva, L2[va], {"test": Wte},
                        seed=seed, device=device, num_classes=2, num_epochs=NUM_EPOCHS,
                        batch_size=BATCH_SIZE, lr=LEARNING_RATE,
                        weight_decay=WEIGHT_DECAY, patience=EARLY_STOPPING_PATIENCE)
                    m = save_run(path, te, L2[te], L3[te], probs["test"], info,
                                 {"method": model_name, "family": "deep_learning", "aug": aug,
                                  "n_train_after_aug": int(len(L2tr_a))})
                    print(f"    {model_name:9s} aug={aug} f1_macro={m['f1_macro']:.4f} "
                          f"bal={m['balanced_accuracy']:.4f} f1_ng={m['f1_notgood']:.4f} "
                          f"({info['epochs_ran']} ep, {info['train_time_s']:.0f}s)")
                if aug:
                    del Wtr_a
            torch.cuda.empty_cache()

    print_success("Tamamlandı.")


if __name__ == "__main__":
    main()
