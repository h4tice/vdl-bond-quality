"""
MODELS ZOO
==========
Makale karşılaştırması için 4 farklı mimari:

1. VDLCNN (2D CNN)       - mevcut baseline, waveform'u (depth, time) görüntü olarak işler
2. ResNet1D              - 1D residual CNN, time-domain sequence olarak işler
3. WaveformLSTM          - LSTM, time-domain sequence olarak
4. AttentionNet          - self-attention ile interpretability sağlar

Tüm modeller aynı (B, 1, 140, 256) input alır, (B, num_classes) output verir.
Bu sayede aynı pipeline'da rahatça swap edilebilir.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# MODEL 1: VDLCNN (2D)  -- mevcut baseline
# ============================================================================

class VDLCNN(nn.Module):
    """2D CNN classifier - mevcut baseline."""
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (B, 1, H, W)
        x = self.features(x)
        return self.classifier(x)


# ============================================================================
# MODEL 2: ResNet-1D
# ============================================================================
# Her depth-sample'ı bir time-series (256 sample) olarak işle.
# Window içindeki depth boyutu (140) batch dimension'a göç eder → ortalama alınır.
# Daha doğrusu: önce her depth ayrı işlenir, sonra depth-wise pooling.

class ResidualBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet1D(nn.Module):
    """
    ResNet-1D classifier.
    Input (B, 1, 140, 256) → her depth row'u ayrı 1D conv, sonra global pool.
    """
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(
            ResidualBlock1D(32, 32),
            ResidualBlock1D(32, 32),
        )
        self.layer2 = nn.Sequential(
            ResidualBlock1D(32, 64, stride=2),
            ResidualBlock1D(64, 64),
        )
        self.layer3 = nn.Sequential(
            ResidualBlock1D(64, 128, stride=2),
            ResidualBlock1D(128, 128),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: (B, 1, H, W) where H=depth, W=time
        B, C, H, W = x.shape
        # Her depth row'unu ayrı işle: birleştir (B*H, 1, W)
        x = x.permute(0, 2, 1, 3).reshape(B * H, C, W)
        # 1D ResNet
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).squeeze(-1)   # (B*H, 128)
        # Depth-wise aggregate: B*H → B (mean pool over depth)
        x = x.view(B, H, -1).mean(dim=1)   # (B, 128)
        return self.classifier(x)


# ============================================================================
# MODEL 3: Waveform LSTM
# ============================================================================
# Depth boyutunu sequence olarak işle (100 step).
# Her depth-row'unun 256 time sample'ı feature olarak verilir.
# Bu hem fizik olarak doğru (depth boyunca bond pattern'i öğren) hem 100x
# daha hızlı (eski versiyonda B*H=1600 sequence × 256 step vardı).

class WaveformLSTM(nn.Module):
    def __init__(self, num_classes: int = 3,
                 hidden_size: int = 128, num_layers: int = 2,
                 bidirectional: bool = True,
                 time_compress: int = 32):
        """
        time_compress: 256 time sample'ı önce 1D conv ile bu boyuta sıkıştır.
                       LSTM input_size = time_compress (256 yerine 32).
        """
        super().__init__()
        # 1) Önce time eksenini sıkıştır (256 → time_compress)
        self.time_encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),    # 256→128
            nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),   # 128→64
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.AdaptiveAvgPool1d(time_compress),                     # →time_compress
        )
        # 2) LSTM: input = 32 features, sequence = H (depth) = 100
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        d = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(d, 64), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: (B, 1, H, W) where H=depth, W=time (256)
        B, C, H, W = x.shape

        # 1) Her depth-row'u time_encoder'a ver, time-axis sıkıştır
        x = x.permute(0, 2, 1, 3).reshape(B * H, C, W)   # (B*H, 1, 256)
        feat = self.time_encoder(x)                       # (B*H, 32, time_compress)
        # Flatten feature dimension (avg over time_compress)
        feat = feat.mean(dim=-1)                          # (B*H, 32)
        feat = feat.view(B, H, -1)                        # (B, H=100, 32)

        # 2) LSTM over depth dimension (H=100 sequence, 32 features)
        out, (h_n, _) = self.lstm(feat)

        # 3) Son hidden state'i al (bidirectional ise iki yönü birleştir)
        if self.lstm.bidirectional:
            last = torch.cat([h_n[-2], h_n[-1]], dim=-1)   # (B, 2*hidden)
        else:
            last = h_n[-1]                                  # (B, hidden)

        return self.classifier(last)


# ============================================================================
# MODEL 4: Attention Net
# ============================================================================
# Hafif self-attention. Her depth row'u embed et, attention ile ağırlıkla, sınıflandır.
# Attention weights interpretability sağlar - hangi derinlik aralığı kritik?

class AttentionNet(nn.Module):
    def __init__(self, num_classes: int = 3, embed_dim: int = 64,
                 num_heads: int = 4, positional: bool = False, max_len: int = 512):
        super().__init__()
        # positional=True: öğrenilebilir derinlik konum kodlaması. Kodlama yoksa
        # self-attention token sırasına duyarsızdır (derinlik sırası görülmez).
        self.pos_embed = None
        if positional:
            self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        # Önce her depth row'unu küçük CNN ile embed
        self.depth_embed = nn.Sequential(
            nn.Conv1d(1, 32, 7, stride=2, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, embed_dim, 5, stride=2, padding=2),
            nn.BatchNorm1d(embed_dim), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        # Multi-head self-attention over depth tokens
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=0.2, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        # Classification token (BERT tarzı)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(embed_dim, 64), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x, return_attention: bool = False):
        # x: (B, 1, H, W)
        B, C, H, W = x.shape
        # Her depth row'u embed et
        x = x.permute(0, 2, 1, 3).reshape(B * H, C, W)
        emb = self.depth_embed(x).squeeze(-1)   # (B*H, embed_dim)
        emb = emb.view(B, H, -1)                 # (B, H, embed_dim)
        if self.pos_embed is not None:
            emb = emb + self.pos_embed[:, :H]

        # CLS token ekle
        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, emb], dim=1)        # (B, H+1, embed_dim)

        # Self-attention
        attn_out, attn_weights = self.attn(seq, seq, seq, need_weights=True,
                                            average_attn_weights=True)
        seq = self.norm1(seq + attn_out)

        # FFN
        ffn_out = self.ffn(seq)
        seq = self.norm2(seq + ffn_out)

        # CLS token output → classifier
        cls_repr = seq[:, 0]                      # (B, embed_dim)
        logits = self.classifier(cls_repr)

        if return_attention:
            # attention weights of CLS over depth tokens
            return logits, attn_weights[:, 0, 1:]  # (B, H)
        return logits


# ============================================================================
# MODEL REGISTRY
# ============================================================================

MODEL_REGISTRY = {
    "cnn":      VDLCNN,
    "resnet1d": ResNet1D,
    "lstm":     WaveformLSTM,
    "attention": AttentionNet,
    "attention_pe": lambda num_classes: AttentionNet(num_classes=num_classes, positional=True,
                                                     max_len=100),
}


def build_model(name: str, num_classes: int) -> nn.Module:
    """name → model instance."""
    name = name.lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Bilinmeyen model: {name}. "
                         f"Mevcut: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](num_classes=num_classes)


def count_parameters(model: nn.Module) -> int:
    """Eğitilebilir parametre sayısı."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(name: str, num_classes: int = 3) -> dict:
    """Model özet bilgileri."""
    m = build_model(name, num_classes)
    return {
        "name": name,
        "num_classes": num_classes,
        "n_params": count_parameters(m),
        "n_params_M": round(count_parameters(m) / 1e6, 3),
    }