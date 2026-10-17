"""
REVISION-2: CROSS-WELL TEST (78B-32 -> 16B(78)-32)
=================================================
Hakem-1 madde 1: farklı veri kümesinde genelleme.

Modeller 78B-32'nin tamamında eğitilir (validation: blok uçlarından purge'lü
dilimler), hiç görülmemiş 16B(78)-32 kuyusunda test edilir. 16B-32 farklı
casing (11.75" / 7"), farklı araç (IBC + sonic) ve farklı CBL türevi (DCBL,
attenuation'dan) kullanır; etiketlerin ~%99'u Poor. Bu nedenle test esas
olarak alan kayması altında Poor tespitini / yanlış "Good" alarmını ölçer.

Normalizasyon iki şekilde raporlanır:
  own     16B kendi global mean/std'si ile (etiket kullanmaz; kuyu bazlı)
  source  78B train istatistikleri ile (ham aktarım)

KULLANIM:
    python scripts/rev2_run_crosswell.py

ÇIKTI:
    outputs/revision2/runs/crosswell/<yöntem>_aug<a>_s<seed>_<norm>.npz
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from config import (DATA_PROCESSED_DIR, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE,
                    WEIGHT_DECAY, EARLY_STOPPING_PATIENCE, AUG_CONFIG,
                    AUGMENTATION_MULTIPLIER, AUGMENTATION_MIN_CLASS_TARGET)
from src.blocked_cv import purged_train_val
from src.rev2_engine import (fit_predict, train_only_normalize, augment_train,
                             engineered_features, cbl_proxy_features)
from src.utils import print_header, print_info, print_success, get_device
from scripts.rev2_run_cv import RUNS_DIR, MODELS, SEEDS, load_well, save_run


def zscore(W):
    return (W - W.mean()) / (W.std() + 1e-8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--aug", nargs="+", type=int, default=[0, 1])
    args = p.parse_args()

    device = get_device()
    print_header("REVISION-2 CROSS-WELL: 78B-32 -> 16B(78)-32")
    src = load_well("forge_78b32")
    tgt = load_well("forge_16b32")
    print_info(f"Kaynak 78B-32: {src['windows'].shape} {np.bincount(src['labels_3'], minlength=3).tolist()}")
    print_info(f"Hedef  16B-32: {tgt['windows'].shape} {np.bincount(tgt['labels_3'], minlength=3).tolist()}")

    out_dir = RUNS_DIR / "crosswell"
    out_dir.mkdir(parents=True, exist_ok=True)
    W, L = src["windows"], src["labels_3"]
    Wt, Lt = tgt["windows"], tgt["labels_3"]
    test_idx = np.arange(len(Lt))

    # Klasik baseline'lar: öznitelikler kuyu bazlı z-score'lanmış window'lardan
    fe_s, fp_s = engineered_features(zscore(W)), cbl_proxy_features(zscore(W))
    fe_t, fp_t = engineered_features(zscore(Wt)), cbl_proxy_features(zscore(Wt))

    for seed in args.seeds:
        tr, va = purged_train_val(src["starts"], W.shape[1], seed=seed)
        for name, clf, Xs, Xt in (
            ("engineered_lr", make_pipeline(StandardScaler(), LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=seed)), fe_s, fe_t),
            ("engineered_rf", RandomForestClassifier(
                n_estimators=300, max_depth=10, class_weight="balanced",
                random_state=seed, n_jobs=-1), fe_s, fe_t),
            ("cbl_proxy_tree", DecisionTreeClassifier(
                max_depth=3, class_weight="balanced", random_state=seed), fp_s, fp_t),
        ):
            path = out_dir / f"{name}_aug0_s{seed}_own.npz"
            if path.exists():
                continue
            clf.fit(Xs[tr], L[tr])
            pr = np.zeros((len(Lt), 3))
            pr[:, clf.classes_] = clf.predict_proba(Xt)
            m = save_run(path, test_idx, Lt, pr, {"method": name, "family": "baseline",
                                                  "norm": "own", "seed": seed})
            print(f"  {name:16s} s{seed}: recall_poor={m['recall_poor']:.4f}")

        Wtr_n, Wva_n, Wt_src = train_only_normalize(W[tr], W[va], Wt)
        Wt_own = zscore(Wt)
        for aug in args.aug:
            if aug:
                Wtr_a, Ltr_a, _ = augment_train(Wtr_n, L[tr], AUG_CONFIG, AUGMENTATION_MULTIPLIER,
                                                AUGMENTATION_MIN_CLASS_TARGET, seed)
            else:
                Wtr_a, Ltr_a = Wtr_n, L[tr]
            for model_name in args.models:
                paths = {n: out_dir / f"{model_name}_aug{aug}_s{seed}_{n}.npz"
                         for n in ("own", "source")}
                if all(pth.exists() for pth in paths.values()):
                    continue
                probs, info = fit_predict(
                    model_name, Wtr_a, Ltr_a, Wva_n, L[va],
                    {"own": Wt_own, "source": Wt_src}, seed=seed, device=device,
                    num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE,
                    weight_decay=WEIGHT_DECAY, patience=EARLY_STOPPING_PATIENCE)
                for n, pth in paths.items():
                    m = save_run(pth, test_idx, Lt, probs[n], info,
                                 {"method": model_name, "family": "deep_learning",
                                  "aug": aug, "norm": n, "seed": seed})
                    pred_dist = np.bincount(probs[n].argmax(1), minlength=3).tolist()
                    print(f"  {model_name:9s} aug={aug} s{seed} {n:6s}: "
                          f"recall_poor={m['recall_poor']:.4f} pred_dist={pred_dist}")

    print_success("Cross-well tamamlandı.")


if __name__ == "__main__":
    main()
