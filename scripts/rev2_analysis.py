"""
REVISION-2: ANALYSIS — TABLES, STATISTICS, FIGURES
==================================================
outputs/revision2/runs/ altındaki tüm koşuları toplar ve hakem cevapları ile
revize makale için tablo/figür üretir.

  T1  blocked CV ana sonuçlar (havuzlanmış OOF, seed'ler üzerinden ort ± std)
  T2  split protokolü karşılaştırması (random / stratdepth / blocked)
  T3  eşleştirilmiş Wilcoxon testleri (Holm düzeltmeli)
  T4  cross-well 78B → 16B
  T5  öğrenme eğrisi
  T6  etiket döngüselliği: eşiğe yakınlığa göre hata, CBL ile korelasyon
  T7  augmentation kopya analizi (Partial)
  T8  veri ve split tanıları

KULLANIM:
    python scripts/rev2_analysis.py
"""

import sys
import json
from pathlib import Path
from itertools import combinations

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, spearmanr
from sklearn.metrics import cohen_kappa_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (LABEL_THRESHOLDS, AUG_CONFIG, AUGMENTATION_MULTIPLIER,
                    AUGMENTATION_MIN_CLASS_TARGET)
from src.rev2_engine import metrics, train_only_normalize, augment_train
from src.blocked_cv import purged_blocked_kfold
from src.utils import print_header, print_success, print_info
from scripts.rev2_run_cv import REV2_DIR, RUNS_DIR, load_well

TAB_DIR = REV2_DIR / "tables"
FIG_DIR = REV2_DIR / "figures"
DL = ["attention", "cnn", "resnet1d", "lstm"]
BASELINES = ["majority", "cbl_proxy_tree", "engineered_lr", "engineered_rf"]
LABEL = {"attention": "AttentionNet", "cnn": "2D-CNN", "resnet1d": "ResNet1D",
         "lstm": "Bi-LSTM", "majority": "Majority", "cbl_proxy_tree": "CBL-proxy tree",
         "engineered_lr": "Engineered + LR", "engineered_rf": "Engineered + RF"}
# Referans paletin ilk slotları (dataviz palette.md, light)
COLOR = {"attention": "#2a78d6", "cnn": "#eb6834", "resnet1d": "#1baf7a",
         "lstm": "#eda100", "baseline": "#8a8985"}
PROTO_COLOR = {"random": "#2a78d6", "stratdepth": "#eb6834", "blocked": "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
KEY = ["f1_macro", "balanced_accuracy", "f1_weighted", "accuracy",
       "precision_partial", "recall_partial", "f1_partial", "recall_poor", "recall_good"]


def style_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


# ============================================================================
# YÜKLEME
# ============================================================================

def load_runs():
    rows = []
    for proto_dir in sorted(RUNS_DIR.iterdir()):
        if not proto_dir.is_dir():
            continue
        for f in sorted(proto_dir.glob("*.npz")):
            z = np.load(f, allow_pickle=False)
            info = json.loads(str(z["info"]))
            stem = f.stem.split("_")
            # <method>[_fracX]_aug<a>_s<seed>_<f<fold>|norm>
            last = stem[-1]
            seed = int(stem[-2][1:])
            aug = int(stem[-3][3:])
            method_parts = stem[:-3]
            frac = 1.0
            if method_parts[-1].startswith("frac"):
                frac = float(method_parts[-1][4:])
                method_parts = method_parts[:-1]
            rows.append({
                "protocol": proto_dir.name, "method": "_".join(method_parts),
                "family": info.get("family"), "aug": aug, "seed": seed,
                "fold": int(last[1:]) if last.startswith("f") else 0,
                "norm": last if not last.startswith("f") else None,
                "frac": frac, "n_params": info.get("n_params"),
                "epochs_ran": info.get("epochs_ran"),
                "test_idx": z["test_idx"], "y_true": z["y_true"], "probs": z["probs"],
                **{k: v for k, v in info["metrics"].items() if k != "cm"},
            })
    return pd.DataFrame(rows)


def pooled(df, group_cols):
    """Seed başına katları birleştir (out-of-fold), metrik hesapla."""
    out = []
    for keys, g in df.groupby(group_cols + ["seed"], dropna=False):
        y = np.concatenate(g["y_true"].values)
        p = np.concatenate(g["probs"].values).argmax(1)
        m = metrics(y, p)
        m.update(dict(zip(group_cols + ["seed"], keys)))
        out.append(m)
    return pd.DataFrame(out)


def mean_std(df, group_cols, cols):
    agg = df.groupby(group_cols, dropna=False)[cols].agg(["mean", "std"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    return agg.reset_index()


def fmt(m, s, d=3):
    return f"{m:.{d}f} ± {s:.{d}f}" if pd.notna(s) else f"{m:.{d}f}"


def pretty_table(agg, group_cols, cols):
    t = agg[group_cols].copy()
    for c in cols:
        t[c] = [fmt(m, s) for m, s in zip(agg[f"{c}_mean"], agg[f"{c}_std"])]
    return t


def save_table(df, name):
    df.to_csv(TAB_DIR / f"{name}.csv", index=False)
    with open(TAB_DIR / f"{name}.md", "w", encoding="utf-8") as f:
        f.write(df.to_markdown(index=False))
    print_success(f"{name}")
    print(df.to_string(index=False))


# ============================================================================
# TABLOLAR
# ============================================================================

def t1_main(df):
    b = df[(df.protocol == "blocked") & (df.frac == 1.0)]
    pool = pooled(b, ["method", "aug"])
    agg = mean_std(pool, ["method", "aug"], KEY)
    fold = mean_std(b, ["method", "aug"], ["f1_weighted", "balanced_accuracy"])
    fold.columns = ["method", "aug", "fold_f1w_mean", "fold_f1w_std", "fold_bal_mean", "fold_bal_std"]
    agg = agg.merge(fold, on=["method", "aug"])
    params = b.groupby("method")["n_params"].first()
    agg["params_k"] = agg["method"].map(params).fillna(0) / 1e3
    agg = agg.sort_values("f1_macro_mean", ascending=False)
    t = pretty_table(agg, ["method", "aug"], KEY)
    t["per_fold_f1_weighted"] = [fmt(m, s) for m, s in zip(agg.fold_f1w_mean, agg.fold_f1w_std)]
    t.insert(2, "params_k", agg["params_k"].round(1).values)
    t["method"] = t["method"].map(LABEL)
    save_table(t, "T1_blocked_cv_main")
    return pool, agg


def t2_protocols(df):
    d = df[(df.family == "deep_learning") & (df.frac == 1.0) &
           df.protocol.isin(["random", "stratdepth", "blocked"])]
    pool = pooled(d, ["protocol", "method", "aug"])
    agg = mean_std(pool, ["protocol", "method", "aug"], ["f1_weighted", "f1_macro", "balanced_accuracy"])

    t = pretty_table(agg, ["protocol", "method", "aug"], ["f1_weighted", "f1_macro", "balanced_accuracy"])
    t["method"] = t["method"].map(LABEL)
    save_table(t, "T2_protocol_comparison")
    return agg


def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(running, 1.0)
    return adj


def t3_stats(df, pool):
    """
    Eşleştirilmiş Wilcoxon: (seed, kat) çiftleri üzerinden kat-düzeyi
    balanced accuracy ve weighted-F1; seed düzeyinde havuzlanmış macro-F1.
    """
    b = df[(df.protocol == "blocked") & (df.frac == 1.0)]
    rows = []

    def pair(a_sel, b_sel, label, metric, level):
        if level == "fold":
            A = b[a_sel(b)].set_index(["seed", "fold"])[metric]
            B = b[b_sel(b)].set_index(["seed", "fold"])[metric]
        else:
            A = pool[a_sel(pool)].set_index("seed")[metric]
            B = pool[b_sel(pool)].set_index("seed")[metric]
        j = A.to_frame("a").join(B.to_frame("b"), how="inner").dropna()
        if len(j) < 5 or np.allclose(j.a, j.b):
            p = np.nan
        else:
            p = wilcoxon(j.a, j.b).pvalue
        rows.append({"comparison": label, "metric": metric, "level": level, "n_pairs": len(j),
                     "mean_a": j.a.mean(), "mean_b": j.b.mean(),
                     "mean_diff": (j.a - j.b).mean(),
                     "wins_a": int((j.a > j.b).sum()), "p": p})

    sel = lambda m, a: (lambda d: (d.method == m) & (d.aug == a))
    for metric, level in (("f1_weighted", "fold"), ("balanced_accuracy", "fold"),
                          ("f1_macro", "seed")):
        for m in ["cnn", "resnet1d", "lstm"]:
            pair(sel("attention", 1), sel(m, 1), f"AttentionNet vs {LABEL[m]} (aug)", metric, level)
        for m in DL:
            pair(sel(m, 1), sel(m, 0), f"{LABEL[m]}: aug vs no-aug", metric, level)
        for m in DL:
            pair(sel(m, 1), sel("engineered_rf", 0), f"{LABEL[m]} (aug) vs Engineered + RF", metric, level)
    t = pd.DataFrame(rows)
    for metric in t.metric.unique():
        mask = (t.metric == metric) & t.p.notna()
        t.loc[mask, "p_holm"] = holm(t.loc[mask, "p"].values)
    save_table(t.round(4), "T3_paired_tests")


def t4_crosswell(df):
    c = df[df.protocol == "crosswell"]
    if c.empty:
        print_info("cross-well sonuçları yok, T4 atlandı")
        return
    rows = []
    for (m, aug, norm), g in c.groupby(["method", "aug", "norm"]):
        pred = np.stack([p.argmax(1) for p in g.probs])          # (seeds, N)
        y = g.y_true.iloc[0]
        dist = np.array([np.bincount(p, minlength=3) / len(p) for p in pred])
        rows.append({
            "method": LABEL[m], "aug": aug, "norm": norm, "n_seeds": len(g),
            "recall_poor": fmt(g.recall_poor.mean(), g.recall_poor.std()),
            "pred_good_%": fmt(100 * dist[:, 0].mean(), 100 * dist[:, 0].std(), 1),
            "pred_partial_%": fmt(100 * dist[:, 1].mean(), 100 * dist[:, 1].std(), 1),
            "pred_poor_%": fmt(100 * dist[:, 2].mean(), 100 * dist[:, 2].std(), 1),
            "nonpoor_correct": f"{np.mean([(p[y != 2] == y[y != 2]).sum() for p in pred]):.1f}/{(y != 2).sum()}",
        })
    save_table(pd.DataFrame(rows), "T4_crosswell_78B_to_16B")


def t5_learning_curve(df):
    full = df[(df.protocol == "blocked") & (df.aug == 1) & (df.family == "deep_learning")]
    lc = df[df.protocol == "lc"]
    if lc.empty:
        print_info("öğrenme eğrisi sonuçları yok, T5 atlandı")
        return None
    seeds = sorted(lc.seed.unique())
    both = pd.concat([lc, full[full.seed.isin(seeds)]])
    pool = pooled(both, ["method", "frac"])
    agg = mean_std(pool, ["method", "frac"], ["f1_macro", "balanced_accuracy", "f1_weighted"])
    t = pretty_table(agg, ["method", "frac"], ["f1_macro", "balanced_accuracy", "f1_weighted"])
    t["method"] = t["method"].map(LABEL)
    save_table(t, "T5_learning_curve")
    return agg


def t6_label_circularity(df, data):
    """
    Hakem-2 madde 4. Havuzlanmış OOF tahminlerinde hataların etiket eşiklerine
    yakınlıkla ilişkisi; P(Poor) ile CBL arasındaki sıra korelasyonu; DL ile
    eşik-benzeri CBL-proxy ağacının uyumu.
    """
    th = LABEL_THRESHOLDS
    cbl, bi = data["cbl"], data["bi"]
    d_cbl = np.minimum(np.abs(cbl - th["cbl_good"]) / th["cbl_good"],
                       np.abs(cbl - th["cbl_poor"]) / th["cbl_poor"])
    d_bi = np.minimum(np.abs(bi - th["bi_good"]) / th["bi_good"],
                      np.abs(bi - th["bi_poor"]) / th["bi_poor"])
    margin = np.minimum(d_cbl, d_bi)
    bins = [0, 0.1, 0.25, 0.5, np.inf]
    bin_lab = ["<10%", "10–25%", "25–50%", ">50%"]
    which = np.digitize(margin, bins[1:-1])

    b = df[(df.protocol == "blocked") & (df.frac == 1.0)]
    rows, oof = [], {}
    for (m, aug), g in b.groupby(["method", "aug"]):
        if m == "majority":
            continue
        for seed, gs in g.groupby("seed"):
            idx = np.concatenate(gs.test_idx.values)
            if len(idx) != len(data["labels_3"]):     # seed'in tüm katları bitmemiş
                continue
            pr = np.zeros((len(data["labels_3"]), 3))
            pr[idx] = np.concatenate(gs.probs.values)
            oof[(m, aug, seed)] = pr
            err = pr.argmax(1) != data["labels_3"]
            row = {"method": LABEL[m], "aug": aug, "seed": seed,
                   "spearman_Ppoor_CBL": spearmanr(pr[:, 2], cbl).statistic,
                   "spearman_Pgood_BI": spearmanr(pr[:, 0], bi).statistic}
            for k, lab in enumerate(bin_lab):
                row[f"err_margin_{lab}"] = err[which == k].mean()
            rows.append(row)
    if not rows:
        print_info("Tamamlanmış seed yok, T6 atlandı")
        return None
    r = pd.DataFrame(rows)
    cols = [c for c in r.columns if c not in ("method", "aug", "seed")]
    agg = mean_std(r, ["method", "aug"], cols)
    t = pretty_table(agg, ["method", "aug"], cols)
    counts = {lab: int((which == k).sum()) for k, lab in enumerate(bin_lab)}
    save_table(t, "T6_label_circularity")
    print_info(f"Eşik yakınlığı kutularındaki window sayısı: {counts}")

    # DL ↔ CBL-proxy ağacı uyumu (Cohen kappa)
    kap = []
    for m in DL:
        for seed in sorted(b.seed.unique()):
            a, t_ = oof.get((m, 1, seed)), oof.get(("cbl_proxy_tree", 0, seed))
            if a is not None and t_ is not None:
                kap.append({"method": LABEL[m], "seed": seed,
                            "kappa_vs_proxy_tree": cohen_kappa_score(a.argmax(1), t_.argmax(1)),
                            "kappa_vs_labels": cohen_kappa_score(a.argmax(1), data["labels_3"])})
    if kap:
        k = pd.DataFrame(kap)
        save_table(pretty_table(mean_std(k, ["method"], ["kappa_vs_proxy_tree", "kappa_vs_labels"]),
                                ["method"], ["kappa_vs_proxy_tree", "kappa_vs_labels"]),
                   "T6b_agreement_with_proxy")
    return {"margin": margin, "which": which, "bin_lab": bin_lab, "counts": counts, "oof": oof}


def t7_augmentation_duplicates(data, seed=42):
    """
    Hakem-1 madde 4: Partial için augmentation'ın ürettiği örnekler kaynak
    window'larına ne kadar benziyor? Her sentetik örneğin en yakın orijinal
    Partial train window'u ile Pearson korelasyonu, orijinallerin kendi
    aralarındaki en yakın komşu korelasyonu ile karşılaştırılır.
    """
    W, L = data["windows"], data["labels_3"]
    rows = []
    for split in purged_blocked_kfold(data["starts"], L, W.shape[1], seed=seed):
        Wtr, = train_only_normalize(W[split["train"]])
        Ltr = L[split["train"]]
        Wa, La, is_aug = augment_train(Wtr, Ltr, AUG_CONFIG, AUGMENTATION_MULTIPLIER,
                                       AUGMENTATION_MIN_CLASS_TARGET, seed)
        orig = Wa[(La == 1) & ~is_aug].reshape((La[(La == 1) & ~is_aug]).size, -1)
        syn = Wa[(La == 1) & is_aug].reshape((La[(La == 1) & is_aug]).size, -1)
        z = lambda X: (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-8)
        zo, zs = z(orig), z(syn)
        c_oo = zo @ zo.T / zo.shape[1]
        np.fill_diagonal(c_oo, -np.inf)
        c_so = zs @ zo.T / zo.shape[1]
        starts_tr = np.sort(data["starts"][split["train"]][Ltr == 1])
        zones = int(1 + np.sum(np.diff(starts_tr) > 5)) if len(starts_tr) else 0
        rows.append({"fold": split["fold"], "n_partial_original": len(orig),
                     "n_partial_independent_zones": zones,
                     "n_partial_synthetic": len(syn),
                     "synthetic_per_original": len(syn) / max(len(orig), 1),
                     "nn_corr_original_original": float(np.median(c_oo.max(1))),
                     "nn_corr_synthetic_original": float(np.median(c_so.max(1))),
                     "frac_synthetic_corr_gt_0.95": float((c_so.max(1) > 0.95).mean())})
    save_table(pd.DataFrame(rows).round(3), "T7_augmentation_duplicates")


def t8_diagnostics(data):
    rows = []
    for proto in ("blocked", "random", "stratdepth"):
        p = RUNS_DIR / proto
        for f in sorted(p.glob("splits_s*.json")):
            for d in json.load(open(f)):
                rows.append({"protocol": proto, "seed": int(f.stem[8:]), **{
                    k: d.get(k) for k in ("fold", "n_train", "n_val", "n_test", "n_purged_train",
                                          "min_gap_train_test",
                                          "frac_test_sharing_rows_with_train")},
                             "test_class_counts": str(d.get("test_class_counts"))})
    if rows:
        save_table(pd.DataFrame(rows), "T8_split_diagnostics")


# ============================================================================
# FİGÜRLER
# ============================================================================

def fig_protocols(agg):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    protos = ["random", "stratdepth", "blocked"]
    names = {"random": "Random split", "stratdepth": "Stratified-depth (original)",
             "blocked": "Purged blocked CV (revised)"}
    for ax, metric, title in zip(axes, ["f1_weighted", "f1_macro"], ["Weighted F1", "Macro F1"]):
        a = agg[agg.aug == 0]
        x = np.arange(len(DL))
        w = 0.26
        for i, pr in enumerate(protos):
            s = a[a.protocol == pr].set_index("method").reindex(DL)
            ax.bar(x + (i - 1) * w, s[f"{metric}_mean"], w - 0.02, yerr=s[f"{metric}_std"],
                   color=PROTO_COLOR[pr], label=names[pr], capsize=2,
                   error_kw={"elinewidth": 0.8, "ecolor": INK2})
        ax.set_xticks(x, [LABEL[m] for m in DL])
        ax.set_title(f"{title} (no augmentation, 5 seeds)", color=INK, fontsize=10)
        ax.set_ylim(0, 1.0)
        style_axes(ax)
    axes[0].set_ylabel("Score", color=INK2)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_protocol_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def fig_confusions(df):
    b = df[(df.protocol == "blocked") & (df.frac == 1.0)]
    methods = [(m, 1) for m in DL] + [("engineered_rf", 0), ("cbl_proxy_tree", 0)]
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    names = ["Good", "Partial", "Poor"]
    for ax, (m, aug) in zip(axes.flat, methods):
        g = b[(b.method == m) & (b.aug == aug)]
        if g.empty:
            ax.axis("off"); continue
        cms = []
        for _, gs in g.groupby("seed"):
            y = np.concatenate(gs.y_true.values)
            p = np.concatenate(gs.probs.values).argmax(1)
            cms.append(np.array(metrics(y, p)["cm"]))
        cm = np.mean(cms, axis=0)
        rec = cm / cm.sum(1, keepdims=True)
        ax.imshow(rec, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i, j]:.0f}\n({100 * rec[i, j]:.0f}%)", ha="center", va="center",
                        fontsize=8, color="white" if rec[i, j] > 0.6 else INK)
        ax.set_xticks(range(3), names, fontsize=8)
        ax.set_yticks(range(3), names, fontsize=8)
        ax.set_xlabel("Predicted", fontsize=8, color=INK2)
        ax.set_ylabel("True", fontsize=8, color=INK2)
        ax.set_title(LABEL[m] + (" + aug" if aug else ""), fontsize=10, color=INK)
    plt.suptitle("Pooled out-of-fold confusion matrices, purged blocked CV (mean over 5 seeds)",
                 fontsize=11, color=INK)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_oof_confusion.png", dpi=300, bbox_inches="tight")
    plt.close()


def fig_learning_curve(agg):
    if agg is None:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in DL:
        s = agg[agg.method == m].sort_values("frac")
        ax.errorbar(100 * s.frac, s.f1_macro_mean, yerr=s.f1_macro_std, color=COLOR[m],
                    lw=2, marker="o", ms=6, capsize=2, label=LABEL[m])
    ax.set_xlabel("Training data used (% of training folds)", color=INK2)
    ax.set_ylabel("Macro F1 (pooled OOF)", color=INK2)
    ax.set_xticks([25, 50, 100])
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_learning_curve.png", dpi=300, bbox_inches="tight")
    plt.close()


def fig_depth_track(df, data, circ):
    """Derinlik boyunca etiket ve OOF tahminleri (en iyi DL + RF), blok sınırları ile."""
    if circ is None:
        return
    order = np.argsort(data["depths"])
    depth_ft = data["depths"][order] / 120.0
    tracks = [("Labels (CBL/BI)", data["labels_3"][order])]
    for m, aug in (("attention", 1), ("cnn", 1), ("engineered_rf", 0)):
        pr = circ["oof"].get((m, aug, 42))
        if pr is not None:
            tracks.append((LABEL[m] + (" + aug" if aug else ""), pr.argmax(1)[order]))
    cls_color = np.array(["#2a78d6", "#eb6834", "#1baf7a"])  # Good, Partial, Poor
    fig, axes = plt.subplots(1, len(tracks), figsize=(1.6 * len(tracks) + 1, 7), sharey=True)
    for ax, (title, lab) in zip(axes, tracks):
        ax.scatter(np.zeros_like(depth_ft), depth_ft, c=cls_color[lab], marker="_", s=400, lw=1.2)
        ax.set_xticks([])
        ax.set_title(title, fontsize=8, color=INK)
        for s in ("top", "right", "bottom"):
            ax.spines[s].set_visible(False)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Depth (ft)", color=INK2)
    handles = [plt.Line2D([], [], color=c, lw=4) for c in cls_color]
    fig.legend(handles, ["Good", "Partial", "Poor"], loc="lower center", ncol=3, frameon=False, fontsize=8)
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    plt.savefig(FIG_DIR / "fig_depth_track_oof.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    print_header("REVISION-2 ANALYSIS")
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = load_well()
    df = load_runs()
    print_info(f"Toplam koşu: {len(df)}  |  protokoller: {df.protocol.value_counts().to_dict()}")

    pool, _ = t1_main(df)
    agg2 = t2_protocols(df)
    t3_stats(df, pool)
    t4_crosswell(df)
    lc = t5_learning_curve(df)
    circ = t6_label_circularity(df, data)
    t7_augmentation_duplicates(data)
    t8_diagnostics(data)

    fig_protocols(agg2)
    fig_confusions(df)
    fig_learning_curve(lc)
    fig_depth_track(df, data, circ)
    print_success(f"Tablolar: {TAB_DIR}\nFigürler: {FIG_DIR}")


if __name__ == "__main__":
    main()
