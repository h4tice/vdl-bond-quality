"""
REVISION-2 TRAINING / EVALUATION ENGINE
=======================================
Tüm revizyon deneyleri tek bir eğitim fonksiyonundan geçer; makaledeki
protokolün aynısı (AdamW 1e-3 / wd 1e-4, batch 16, ağırlıklı CE,
ReduceLROnPlateau factor 0.5 patience 4, early stopping patience 10,
en fazla 50 epoch).

Farklar (hakem düzeltmeleri):
- z-score istatistikleri yalnızca train kümesinden (Hakem-2 madde 5)
- Test tahminleri olasılık olarak döner → out-of-fold havuzlama, macro-F1,
  sınıf bazında metrikler (Hakem-2 madde 3)
- Veri GPU'da tutulur, batch'ler indeksle çekilir (aynı protokol, daha hızlı)
"""

import time
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_recall_fscore_support, confusion_matrix)

from src.models_zoo import build_model, count_parameters
from src.augmentation import compute_proportional_targets, offline_augment_dataset


CLASSES = [0, 1, 2]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_only_normalize(W_train, *others):
    """Train kümesinin global mean/std'si ile tüm kümeleri normalize et."""
    mu = float(W_train.mean())
    sigma = float(W_train.std()) + 1e-8
    return [(W_train - mu) / sigma] + [(W - mu) / sigma for W in others]


def augment_train(W_train, L_train, aug_config, multiplier, min_target, seed):
    counts = {int(k): int(v) for k, v in zip(*np.unique(L_train, return_counts=True))}
    targets = compute_proportional_targets(counts, multiplier=multiplier,
                                           min_class_target=min_target)
    W_aug, L_aug, is_aug = offline_augment_dataset(
        W_train, L_train, class_targets=targets, aug_config=aug_config,
        seed=seed, verbose=False)
    return W_aug, L_aug, is_aug


def class_weights(L, num_classes=3):
    L = np.asarray(L, dtype=np.int64)
    present = np.unique(L)
    cw = np.ones(num_classes, dtype=np.float32)
    if len(present) >= 2:
        for c, w in zip(present, compute_class_weight("balanced", classes=present, y=L)):
            cw[int(c)] = float(w)
    return cw


class FocalLoss(nn.Module):
    """Sınıf ağırlıklı focal loss (Lin et al. 2017); kolay örneklerin katkısını azaltır."""

    def __init__(self, weight, gamma=2.0):
        super().__init__()
        self.weight, self.gamma = weight, gamma

    def forward(self, logits, target):
        logp = torch.log_softmax(logits, dim=1)
        logp_t = logp.gather(1, target[:, None]).squeeze(1)
        w = self.weight[target]
        loss = -w * (1 - logp_t.exp()) ** self.gamma * logp_t
        return loss.sum() / w.sum()


@torch.no_grad()
def predict_proba(model, X, batch_size=64):
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        out.append(torch.softmax(model(X[i:i + batch_size]), dim=1))
    return torch.cat(out).cpu().numpy()


def fit_predict(model_name, W_train, L_train, W_val, L_val, test_sets, *,
                seed, device, num_classes=3, num_epochs=50, batch_size=16,
                lr=1e-3, weight_decay=1e-4, patience=10, sched_patience=4,
                loss="ce"):
    """
    Bir modeli eğit (en iyi val loss'ta durdur), test kümeleri için olasılık döndür.
    test_sets: {ad: W_test}
    Return: (dict ad → (N, C) olasılık, info dict)
    """
    seed_everything(seed)
    to_t = lambda a: torch.as_tensor(a[:, None], dtype=torch.float32, device=device)
    Xtr, Xva = to_t(W_train), to_t(W_val)
    ytr = torch.as_tensor(L_train, dtype=torch.long, device=device)
    yva = torch.as_tensor(L_val, dtype=torch.long, device=device)

    cw = class_weights(L_train, num_classes)
    model = build_model(model_name, num_classes=num_classes).to(device)
    cw_t = torch.tensor(cw, device=device)
    crit = FocalLoss(cw_t) if loss == "focal" else nn.CrossEntropyLoss(weight=cw_t)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5,
                                                       patience=sched_patience)
    gen = torch.Generator(device="cpu").manual_seed(seed)

    best_val, best_state, best_epoch, wait = float("inf"), None, 0, 0
    history = []
    t0 = time.time()
    n = len(Xtr)
    for epoch in range(1, num_epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=gen).to(device)
        tr_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) < 2:          # BatchNorm tek örnekli batch kabul etmez
                continue
            opt.zero_grad()
            loss = crit(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(idx)

        model.eval()
        with torch.no_grad():
            va_loss = 0.0
            for i in range(0, len(Xva), 64):
                va_loss += crit(model(Xva[i:i + 64]), yva[i:i + 64]).item() * len(Xva[i:i + 64])
            va_loss /= max(len(Xva), 1)
        sched.step(va_loss)
        history.append((tr_loss / n, va_loss))

        if va_loss < best_val:
            best_val, best_epoch, wait = va_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    probs = {name: predict_proba(model, to_t(W)) for name, W in test_sets.items()}
    info = {
        "epochs_ran": len(history), "best_epoch": best_epoch,
        "best_val_loss": float(best_val), "train_time_s": time.time() - t0,
        "n_params": count_parameters(model), "class_weights": cw.tolist(),
    }
    del Xtr, Xva, model
    torch.cuda.empty_cache()
    return probs, info


# ============================================================================
# METRİKLER
# ============================================================================

def metrics(y_true, y_pred, classes=CLASSES):
    """Hakem-2 madde 3: macro-F1, balanced accuracy, sınıf bazında metrikler."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=classes,
                                                 zero_division=0)
    present = [c for c in classes if (y_true == c).any()]
    out = {
        "n": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred) if len(present) > 1 else np.nan,
        "f1_weighted": f1_score(y_true, y_pred, labels=classes, average="weighted", zero_division=0),
        # macro-F1 yalnızca testte bulunan sınıflar üzerinden (tek katlı raporlar için);
        # havuzlanmış OOF'ta üç sınıf da mevcut olduğundan klasik macro-F1'e eşit
        "f1_macro": f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0),
    }
    for c, name in zip(classes, ["good", "partial", "poor"]):
        out[f"precision_{name}"] = p[c]
        out[f"recall_{name}"] = r[c]
        out[f"f1_{name}"] = f[c]
        out[f"support_{name}"] = int(s[c])
    out["cm"] = confusion_matrix(y_true, y_pred, labels=classes).tolist()
    return out


# ============================================================================
# KLASİK BASELINE'LAR İÇİN ÖZNİTELİKLER
# ============================================================================

def engineered_features(W, early=64):
    """
    Pencere başına 12 akustik öznitelik (vektörize).
    W: (N, H, T)
    """
    A = np.abs(W)
    E2 = W ** 2
    feats = [
        np.sqrt(E2.mean(axis=(1, 2))),                 # global RMS
        A.max(axis=(1, 2)),                            # tepe genlik
        A.mean(axis=(1, 2)),                           # ortalama |genlik|
        W.std(axis=(1, 2)),                            # std
        A[:, :, :early].max(axis=(1, 2)),              # erken varış tepe
        np.sqrt(E2[:, :, :early].mean(axis=(1, 2))),   # erken enerji
        np.sqrt(E2[:, :, early:150].mean(axis=(1, 2))),  # orta enerji
        np.sqrt(E2[:, :, 150:].mean(axis=(1, 2))),     # geç enerji
    ]
    feats.append(feats[7] / (feats[5] + 1e-6))         # geç/erken oranı
    feats.append(np.abs(np.diff(W, axis=2)).mean(axis=(1, 2)))  # zamansal pürüzlülük
    feats.append(W.mean(axis=2).std(axis=1))           # derinlik boyunca değişkenlik
    feats.append(np.percentile(A.reshape(len(W), -1), 90, axis=1))
    return np.stack(feats, axis=1)


def cbl_proxy_features(W, early=64, center_rows=5):
    """
    Etiket döngüselliği testi (Hakem-2 madde 4): etiketin hesaplandığı orta
    derinlikte, dalga formundan doğrudan okunan iki skaler.
      - erken varış tepe genliği  (CBL genliğinin vekili)
      - geç/erken enerji oranı    (zayıflama / BI vekili)
    """
    H = W.shape[1]
    c = W[:, H // 2 - center_rows // 2: H // 2 + center_rows // 2 + 1, :]
    peak_early = np.abs(c[:, :, :early]).max(axis=(1, 2))
    e_early = np.sqrt((c[:, :, :early] ** 2).mean(axis=(1, 2)))
    e_late = np.sqrt((c[:, :, early:] ** 2).mean(axis=(1, 2)))
    return np.stack([peak_early, e_late / (e_early + 1e-6)], axis=1)
