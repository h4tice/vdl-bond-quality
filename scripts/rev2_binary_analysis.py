"""
REVISION-2: BINARY TASK ANALYSIS (T10)
======================================
rev2_run_binary.py çıktılarını üç sınıflı blocked koşularıyla karşılaştırır.

  binary-trained   modeller doğrudan Good / Not-Good ile eğitildi
  3-class→binary   üç sınıflı modellerin tahmini sonradan birleştirildi
                   (argmax ≠ Good → Not-Good), aynı kat ve seed'ler

Metrikler her seed için havuzlanmış out-of-fold tahminlerden hesaplanır
(T1 ile aynı), ortalama ± std (5 seed). Eşleştirilmiş Wilcoxon + Holm:
seed düzeyinde macro-F1 (n=5) ve kat düzeyinde balanced accuracy (n=25).

KULLANIM:
    python scripts/rev2_binary_analysis.py

ÇIKTI:
    outputs/revision2/tables/T10_binary.{csv,md}
    outputs/revision2/tables/T10b_binary_tests.{csv,md}
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from scripts.rev2_run_cv import RUNS_DIR, REV2_DIR
from scripts.rev2_run_binary import binary_metrics
from scripts.rev2_analysis import LABEL, holm, fmt
from src.labeling import label_binary

TABLE_DIR = REV2_DIR / "tables"
LABEL = {**LABEL, "majority": "Majority", "engineered_lr": "Engineered + LR",
         "engineered_rf": "Engineered + RF", "cbl_proxy_tree": "CBL-proxy tree"}
COLS = ["f1_macro", "balanced_accuracy", "f1_notgood", "recall_notgood", "recall_good",
        "notgood_rate_partial", "notgood_rate_poor"]


def load(proto, binary):
    rows = []
    for f in sorted((RUNS_DIR / proto).glob("*_aug*_s*_f*.npz")):
        stem = f.stem.split("_")
        if stem[-4].startswith("frac"):
            continue
        z = np.load(f)
        if binary:
            y2, y3, pred = z["y_true"], z["y_true3"], z["probs"].argmax(1)
        else:
            y3 = z["y_true"]
            y2, pred = label_binary(y3), (z["probs"].argmax(1) != 0).astype(int)
        rows.append({"method": "_".join(stem[:-3]), "aug": int(stem[-3][3:]),
                     "seed": int(stem[-2][1:]), "fold": int(stem[-1][1:]),
                     "y2": y2, "y3": y3, "pred": pred,
                     "fold_bal": binary_metrics(y2, pred)["balanced_accuracy"]
                     if len(np.unique(y2)) > 1 else np.nan})
    df = pd.DataFrame(rows)
    df["task"] = "binary-trained" if binary else "3-class→binary"
    return df


def per_seed(df):
    out = []
    for (task, m, aug, seed), g in df.groupby(["task", "method", "aug", "seed"]):
        r = binary_metrics(np.concatenate(g.y2.values), np.concatenate(g.pred.values),
                           np.concatenate(g.y3.values))
        out.append({"task": task, "method": m, "aug": aug, "seed": seed,
                    **{c: r[c] for c in COLS}})
    return pd.DataFrame(out)


def save(t, name):
    t.to_csv(TABLE_DIR / f"{name}.csv", index=False)
    (TABLE_DIR / f"{name}.md").write_text(t.to_markdown(index=False), encoding="utf-8")
    print(f"\n== {name}\n{t.to_markdown(index=False)}")


def main():
    runs = pd.concat([load("binary", True), load("blocked", False)], ignore_index=True)
    seeds = per_seed(runs)

    agg = seeds.groupby(["task", "method", "aug"])[COLS].agg(["mean", "std"])
    rows = []
    for (task, m, aug), r in agg.iterrows():
        rows.append({"task": task, "method": LABEL.get(m, m), "aug": aug,
                     **{c: fmt(r[(c, "mean")], r[(c, "std")]) for c in COLS},
                     "_sort": r[("f1_macro", "mean")]})
    t = pd.DataFrame(rows).sort_values(["task", "_sort"], ascending=[False, False])
    save(t.drop(columns="_sort"), "T10_binary")

    # --- paired tests -------------------------------------------------------
    tests = []

    def pair(a, b, label, metric, level):
        key = ["seed"] if level == "seed" else ["seed", "fold"]
        src = seeds if level == "seed" else runs.rename(columns={"fold_bal": metric})
        A = src.query(a)[key + [metric]]
        B = src.query(b)[key + [metric]]
        j = A.merge(B, on=key, suffixes=("_a", "_b")).dropna()
        d = j[f"{metric}_a"] - j[f"{metric}_b"]
        p = wilcoxon(d).pvalue if (d != 0).any() else 1.0
        tests.append({"comparison": label, "metric": metric, "level": level, "n_pairs": len(j),
                      "mean_a": j[f"{metric}_a"].mean(), "mean_b": j[f"{metric}_b"].mean(),
                      "mean_diff": d.mean(), "wins_a": int((d > 0).sum()), "p": p})

    B_, C_ = "task=='binary-trained'", "task=='3-class→binary'"
    dl = ["attention", "cnn", "resnet1d", "lstm"]
    for metric, level in [("f1_macro", "seed"), ("balanced_accuracy", "fold")]:
        for m in dl:
            pair(f"{B_} and method=='{m}' and aug==1", f"{C_} and method=='{m}' and aug==1",
                 f"{LABEL[m]} (aug): binary-trained vs 3-class→binary", metric, level)
        for m in ["cnn", "resnet1d", "lstm"]:
            pair(f"{B_} and method=='attention' and aug==1", f"{B_} and method=='{m}' and aug==1",
                 f"Binary: AttentionNet vs {LABEL[m]} (aug)", metric, level)
        for m in dl:
            pair(f"{B_} and method=='{m}' and aug==1", f"{B_} and method=='{m}' and aug==0",
                 f"Binary: {LABEL[m]} aug vs no-aug", metric, level)
        for m in dl:
            pair(f"{B_} and method=='{m}' and aug==1", f"{B_} and method=='engineered_lr'",
                 f"Binary: {LABEL[m]} (aug) vs Engineered + LR", metric, level)
            pair(f"{B_} and method=='{m}' and aug==1", f"{B_} and method=='cbl_proxy_tree'",
                 f"Binary: {LABEL[m]} (aug) vs CBL-proxy tree", metric, level)
    tt = pd.DataFrame(tests)
    tt["p_holm"] = np.nan
    for metric in tt.metric.unique():
        mask = tt.metric == metric
        tt.loc[mask, "p_holm"] = holm(tt.loc[mask, "p"].values)
    save(tt.round(4), "T10b_binary_tests")


if __name__ == "__main__":
    main()
