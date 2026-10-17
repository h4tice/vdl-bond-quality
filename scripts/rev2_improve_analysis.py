"""
REVISION-2: IMPROVEMENT ANALYSIS (validation-only selection)
============================================================
rev2_run_improve.py çıktılarından, test verisine hiç bakmadan karar veren
son-işlemler:

  raw        her konfigürasyon, argmax
  tuned      sınıf olasılık çarpanları validation macro-F1'ine göre ayarlanır
  ensemble   konfigürasyonların olasılık ortalaması (ayarlı)
  selected   her katta validation macro-F1'i en yüksek konfigürasyon (ayarlı)

Ayrıca 3 sınıflı olasılıklardan binary (Good / Not-Good) metrikler.

KULLANIM:
    python scripts/rev2_improve_analysis.py
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score, balanced_accuracy_score

from src.rev2_engine import metrics
from src.utils import print_header, print_success
from scripts.rev2_run_cv import RUNS_DIR
from scripts.rev2_analysis import TAB_DIR, mean_std, pretty_table, save_table

W_PARTIAL = np.logspace(-1, 2, 25)
W_POOR = np.logspace(-1, 1, 17)
KEY = ["f1_macro", "balanced_accuracy", "f1_weighted", "f1_partial", "recall_partial",
       "precision_partial", "bin_f1_notgood", "bin_balanced_accuracy"]


def load_improve():
    rows = []
    for f in sorted((RUNS_DIR / "improve").glob("*.npz")):
        z = np.load(f)
        info = json.loads(str(z["info"]))
        stem = f.stem.split("_")
        rows.append({"config": "_".join(stem[:-3]), "seed": int(stem[-2][1:]),
                     "fold": int(stem[-1][1:]), "test_idx": z["test_idx"],
                     "y_true": z["y_true"], "probs": z["probs"], "y_val": z["y_val"],
                     "val_probs": z["val_probs"], "n_params": info["n_params"]})
    return pd.DataFrame(rows)


def val_macro_f1(y, pred):
    present = np.unique(y)
    return f1_score(y, pred, labels=present, average="macro", zero_division=0)


def tune_weights(y_val, p_val):
    """Validation macro-F1'ini maksimize eden sınıf çarpanları (Good = 1)."""
    best, best_w = -1.0, np.ones(3)
    wp_grid = W_PARTIAL if (y_val == 1).any() else [1.0]   # val'de Partial yoksa ayarlanamaz
    for wp in wp_grid:
        for wr in W_POOR:
            w = np.array([1.0, wp, wr])
            s = val_macro_f1(y_val, (p_val * w).argmax(1))
            if s > best + 1e-9:
                best, best_w = s, w
    return best_w, best


def score(y, probs):
    m = metrics(y, probs.argmax(1))
    yb = (y != 0).astype(int)
    pb = (probs[:, 1] + probs[:, 2] > probs[:, 0]).astype(int)
    m["bin_f1_notgood"] = f1_score(yb, pb, zero_division=0)
    m["bin_balanced_accuracy"] = balanced_accuracy_score(yb, pb)
    return m


def pooled_rows(fold_preds, label):
    """fold_preds: {(seed, fold): (y_true, probs)} → seed başına havuzlanmış metrikler."""
    rows = []
    for seed in sorted({s for s, _ in fold_preds}):
        keys = sorted(k for k in fold_preds if k[0] == seed)
        y = np.concatenate([fold_preds[k][0] for k in keys])
        p = np.concatenate([fold_preds[k][1] for k in keys])
        rows.append({"variant": label, "seed": seed, **score(y, p)})
    return rows


def main():
    print_header("REVISION-2 IMPROVEMENT ANALYSIS")
    df = load_improve()
    configs = sorted(df.config.unique())
    rows = []
    tuned_cache = {}     # (config, seed, fold) -> (val_score, y_te, tuned test probs, tuned val probs)

    for cfg in configs:
        raw, tuned = {}, {}
        for _, r in df[df.config == cfg].iterrows():
            k = (r.seed, r.fold)
            raw[k] = (r.y_true, r.probs)
            w, s = tune_weights(r.y_val, r.val_probs)
            tuned[k] = (r.y_true, r.probs * w / (r.probs * w).sum(1, keepdims=True))
            tuned_cache[(cfg, *k)] = (s, r.y_true, tuned[k][1])
        rows += pooled_rows(raw, f"{cfg} | raw")
        rows += pooled_rows(tuned, f"{cfg} | tuned")

    # Ensemble'lar (validation'da ayarlı)
    groups = {
        "ensemble: 4 original models (CE)": [c for c in configs if c.endswith("_ce") and
                                             not c.startswith("attention_pe")],
        "ensemble: all configurations": configs,
    }
    for label, members in groups.items():
        ens = {}
        for (seed, fold), g in df[df.config.isin(members)].groupby(["seed", "fold"]):
            pv = np.mean(np.stack(g.val_probs.values), axis=0)
            pt = np.mean(np.stack(g.probs.values), axis=0)
            w, _ = tune_weights(g.y_val.iloc[0], pv)
            ens[(seed, fold)] = (g.y_true.iloc[0], pt * w / (pt * w).sum(1, keepdims=True))
        rows += pooled_rows(ens, label)

    # Validation ile konfigürasyon seçimi
    sel, chosen = {}, []
    for (seed, fold), g in df.groupby(["seed", "fold"]):
        best = max(g.config, key=lambda c: tuned_cache[(c, seed, fold)][0])
        _, y, p = tuned_cache[(best, seed, fold)]
        sel[(seed, fold)] = (y, p)
        chosen.append({"seed": seed, "fold": fold, "selected": best})
    rows += pooled_rows(sel, "validation-selected configuration")
    pd.DataFrame(chosen).to_csv(TAB_DIR / "T9b_selected_configs.csv", index=False)

    res = pd.DataFrame(rows)
    agg = mean_std(res, ["variant"], KEY).sort_values("f1_macro_mean", ascending=False)
    save_table(pretty_table(agg, ["variant"], KEY), "T9_improvement")

    # Referans: önceden önerilen hat (attention, CE, aug, raw) ile eşleştirilmiş test
    ref = res[res.variant == "attention_ce | raw"].set_index("seed")
    tests = []
    for v in agg.variant:
        if v == "attention_ce | raw":
            continue
        cur = res[res.variant == v].set_index("seed")
        j = ref[["f1_macro"]].join(cur[["f1_macro"]], lsuffix="_ref", rsuffix="_v").dropna()
        p = wilcoxon(j.f1_macro_v, j.f1_macro_ref).pvalue if len(j) >= 5 and not np.allclose(
            j.f1_macro_v, j.f1_macro_ref) else np.nan
        tests.append({"variant": v, "mean_diff_vs_attention_ce_raw": (j.f1_macro_v - j.f1_macro_ref).mean(),
                      "seeds_better": int((j.f1_macro_v > j.f1_macro_ref).sum()), "p": p})
    save_table(pd.DataFrame(tests).round(4), "T9c_improvement_vs_reference")
    print_success("Tamam.")


if __name__ == "__main__":
    main()
