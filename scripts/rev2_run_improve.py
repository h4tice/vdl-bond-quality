"""
REVISION-2: PRE-SPECIFIED IMPROVEMENT EXPERIMENT
================================================
Sızıntısız blok CV sonuçları görüldükten sonra yapılacak iyileştirmeler,
test verisine göre seçim yapılmasın diye ÖNCEDEN sabitlendi:

  modeller : attention, attention_pe (öğrenilebilir konum kodlaması),
             cnn, resnet1d, lstm
  kayıp    : ağırlıklı CE, ağırlıklı focal loss (gamma = 2)
  augment. : açık (makaledeki önerilen hat)

Aynı purged blocked katlar ve seed'ler kullanılır. Her koşu validation
olasılıklarını da saklar; karar eşiği ayarı, ensemble ve model seçimi
rev2_improve_analysis.py'de YALNIZCA validation verisiyle yapılır.

KULLANIM:
    python scripts/rev2_run_improve.py

ÇIKTI:
    outputs/revision2/runs/improve/<model>_<loss>_aug1_s<seed>_f<kat>.npz
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from config import (NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
                    EARLY_STOPPING_PATIENCE, AUG_CONFIG,
                    AUGMENTATION_MULTIPLIER, AUGMENTATION_MIN_CLASS_TARGET)
from src.rev2_engine import fit_predict, metrics, train_only_normalize, augment_train
from src.utils import print_header, print_section, print_info, print_success, get_device
from scripts.rev2_run_cv import RUNS_DIR, SEEDS, load_well, build_splits

MODELS = ["attention", "attention_pe", "cnn", "resnet1d", "lstm"]
LOSSES = ["ce", "focal"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--losses", nargs="+", default=LOSSES)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = p.parse_args()

    device = get_device()
    print_header("REVISION-2 IMPROVEMENT EXPERIMENT")
    data = load_well()
    W, L = data["windows"], data["labels_3"]
    out_dir = RUNS_DIR / "improve"
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        for split in build_splits("blocked", data, seed, W.shape[1]):
            fold = split["fold"]
            todo = [(m, l) for m in args.models for l in args.losses
                    if not (out_dir / f"{m}_{l}_aug1_s{seed}_f{fold}.npz").exists()]
            if not todo:
                continue
            print_section(f"seed {seed} | fold {fold}")
            Wtr, Wva, Wte = train_only_normalize(W[split["train"]], W[split["val"]],
                                                 W[split["test"]])
            Wtr_a, Ltr_a, _ = augment_train(Wtr, L[split["train"]], AUG_CONFIG,
                                            AUGMENTATION_MULTIPLIER,
                                            AUGMENTATION_MIN_CLASS_TARGET, seed)
            for model_name, loss in todo:
                probs, info = fit_predict(
                    model_name, Wtr_a, Ltr_a, Wva, L[split["val"]],
                    {"val": Wva, "test": Wte}, seed=seed, device=device,
                    num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE,
                    weight_decay=WEIGHT_DECAY, patience=EARLY_STOPPING_PATIENCE, loss=loss)
                y_te = L[split["test"]]
                m = metrics(y_te, probs["test"].argmax(1))
                np.savez_compressed(
                    out_dir / f"{model_name}_{loss}_aug1_s{seed}_f{fold}.npz",
                    test_idx=split["test"], y_true=y_te, probs=probs["test"].astype(np.float32),
                    val_idx=split["val"], y_val=L[split["val"]],
                    val_probs=probs["val"].astype(np.float32),
                    info=json.dumps({**info, "method": f"{model_name}_{loss}",
                                     "family": "deep_learning", "aug": 1, "loss": loss,
                                     "metrics": m}, default=float))
                print(f"    {model_name:12s} {loss:5s} f1_macro={m['f1_macro']:.4f} "
                      f"f1_partial={m['f1_partial']:.4f} ({info['epochs_ran']} ep, "
                      f"{info['train_time_s']:.0f}s)")
            del Wtr_a
            torch.cuda.empty_cache()
    print_success("Tamamlandı.")


if __name__ == "__main__":
    main()
