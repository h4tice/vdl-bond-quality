"""
MULTI-WELL VDL PIPELINE - CONFIGURATION
========================================
Tüm parametreler tek yerden yönetilir. Mevcut main_vdl_training.py ile çakışmaması
için bu config ayrı bir namespace altında.
"""

from pathlib import Path

# ============================================================================
# PROJECT ROOT
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.resolve()

# ============================================================================
# DATA DIRECTORIES
# ============================================================================

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
RESULTS_DIR = OUTPUT_DIR / "results"
IMAGES_DIR = OUTPUT_DIR / "images"
COMPARISON_DIR = OUTPUT_DIR / "comparison"

# Create all dirs
for d in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODEL_DIR, RESULTS_DIR, IMAGES_DIR, COMPARISON_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================================
# WELL REGISTRY
# ============================================================================
# Her kuyu için: download URL, dosya adı, hangi frame'de VDL var, label stratejisi.
#
# ÖNEMLİ: GDR doğrudan bir .dlis URL'si vermediği için, arşiv .zip içinde geliyor.
# Kullanıcı arşivi manuel indirip data/raw/ altına çıkarıyor. Script dosyayı
# bulup otomatik işliyor.
#
# Alternatif olarak, 78B-32 mevcut projeden taşınabilir (already_downloaded=True).

WELLS = {
    # =========================================================================
    # FORGE 78B-32 (mevcut proje - zaten var)
    # =========================================================================
    "forge_78b32": {
        "name": "FORGE 78B-32",
        "site": "FORGE",
        "file_pattern": "*78B*32*CBL*.dlis",       # dosya adı eşleme paterni
        "source_url": "https://gdr.openei.org/submissions/1330",
        "doi": "10.15121/1814488",
        "formation": "crystalline granite (Roosevelt Hot Springs)",
        "vdl_frame": "60B",
        "wf_channel": "WF1",
        "array_channel": "SWFA1",
        "depth_channel": "TDEP",
        "scalar_channels": ["TDEP", "BI", "CBL", "DT", "GOBO"],
        "enabled": True,
    },

    # =========================================================================
    # FORGE 16A(78)-32 - Injection well
    # DEVRE DIŞI: GDR'de CBL waveform DLIS'i paylaşılmamış, sadece LAS var
    # =========================================================================
    "forge_16a32": {
        "name": "FORGE 16A(78)-32",
        "site": "FORGE",
        "file_pattern": "*16A*CBL*.dlis",
        "source_url": "https://gdr.openei.org/submissions/1292",
        "doi": "10.15121/1777912",
        "formation": "crystalline granite (Roosevelt Hot Springs)",
        "vdl_frame": "60B",
        "wf_channel": "WF1",
        "array_channel": "SWFA1",
        "depth_channel": "TDEP",
        "scalar_channels": ["TDEP", "BI", "CBL", "DT", "GOBO"],
        "enabled": False,   # CBL waveform DLIS bulunamadı
    },

    # =========================================================================
    # FORGE 16B(78)-32 - Production well
    # Sonic WF1 30B frame'inde (3 in aralık), CBL/BI attenuation'dan türetilmiş
    # DCBL/DBI olarak geliyor. 11.75" casing (78B-32: 7"). 78B ile aynı fiziksel
    # pencere boyu için 2'ye seyreltilir (6 in).
    # =========================================================================
    "forge_16b32": {
        "name": "FORGE 16B(78)-32",
        "site": "FORGE",
        "file_pattern": "*16B*IBC*CBL*.dlis",
        "source_url": "https://gdr.openei.org/submissions/1531",
        "doi": "10.15121/2001059",
        "formation": "crystalline granite (Roosevelt Hot Springs)",
        "vdl_frame": "30B",
        "wf_channel": "WF1",
        "array_channel": "SWFA1",
        "depth_channel": "TDEP",
        "scalar_channels": ["TDEP", "DBI", "DCBL", "TT1"],
        "cbl_channel": "DCBL",
        "bi_channel": "DBI",
        "decimate": 2,
        "enabled": True,
    },

    # =========================================================================
    # FORGE 56-32 - Seismic monitoring well
    # DEVRE DIŞI: through-bit sonic DLIS'inde CBL kanalları doğrulanmadı
    # =========================================================================
    "forge_56_32": {
        "name": "FORGE 56-32",
        "site": "FORGE",
        "file_pattern": "*56-32*CBL*.dlis",
        "source_url": "https://gdr.openei.org/submissions/1295",
        "doi": "10.15121/1777170",
        "formation": "crystalline granite (Roosevelt Hot Springs)",
        "vdl_frame": "60B",
        "wf_channel": "WF1",
        "array_channel": "SWFA1",
        "depth_channel": "TDEP",
        "scalar_channels": ["TDEP", "BI", "CBL", "DT", "GOBO"],
        "enabled": False,   # CBL waveform kanalları doğrulanmadı
    },

    # =========================================================================
    # EQUINOR VOLVE - 15/9-F wells (Kuzey Denizi)
    # Farklı jeoloji (Jurassic sandstone), generalization testi için ideal.
    # DSLT tool kullanılıyor, frame/channel isimleri farklı olabilir.
    # =========================================================================
    "volve_f11a": {
        "name": "Volve 15/9-F-11A",
        "site": "VOLVE",
        "file_pattern": "*F-11A*CBL*.dlis",
        "source_url": "https://data.equinor.com/dataset/Volve",
        "formation": "Jurassic Hugin sandstone (North Sea)",
        "vdl_frame": None,          # auto-detect - farklı tool, farklı frame isimleri
        "wf_channel": None,         # auto-detect: WF/WFDL/CBLW benzeri
        "array_channel": None,
        "depth_channel": "TDEP",    # or MD / DEPT
        "scalar_channels": ["TDEP", "CBL", "BI", "TTBK"],
        "enabled": False,           # varsayılan: kapalı (opsiyonel)
    },
}

# ============================================================================
# WINDOWING PARAMETERS
# ============================================================================

WINDOW_SIZE = 140           # her window'daki derinlik sample sayısı
STEP = 10                   # window adım boyutu
TIME_SAMPLES = 256          # her waveform'daki zaman sample sayısı

# ============================================================================
# LABELING THRESHOLDS
# ============================================================================
# CBL (Cement Bond Log) amplitüd (mV) ve BI (Bond Index) için eşikler.
# Klasik endüstri standardı:
#   Good   : CBL < 10 mV ve BI > 0.8
#   Poor   : CBL > 20 mV veya BI < 0.4  (free pipe veya bad bond)
#   Partial: arada kalan her şey
# Bu eşikler her kuyu için fiziksel olarak biraz farklı olabilir; ileride
# kuyu-bazlı normalizasyon için 'adaptive_labeling.py' eklenebilir.

LABEL_THRESHOLDS_STANDARD = {
    "cbl_good": 10.0,
    "cbl_poor": 20.0,
    "bi_good": 0.8,
    "bi_poor": 0.4,
}

# Mevcut projedeki (save_all_training_images.py'da görülen) daha gevşek eşikler:
LABEL_THRESHOLDS_LEGACY = {
    "cbl_good": 5.0,
    "cbl_poor": 20.0,
    "bi_good": 0.8,
    "bi_poor": 0.4,
}

# Aktif kullanılan eşik seti
LABEL_THRESHOLDS = LABEL_THRESHOLDS_STANDARD

# ============================================================================
# CLASS DEFINITIONS
# ============================================================================

CLASS_NAMES_3 = ["Good", "Partial", "Poor"]
CLASS_NAMES_2 = ["Good", "Poor"]            # binary: Partial + Poor merged

CLASS_TO_IDX_3 = {name: i for i, name in enumerate(CLASS_NAMES_3)}
CLASS_TO_IDX_2 = {name: i for i, name in enumerate(CLASS_NAMES_2)}

# ============================================================================
# TRAINING PARAMETERS
# ============================================================================

SEED = 42
NUM_EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 10

# Train/val/test split ratios (stratified)
TEST_SIZE = 0.15
VAL_SIZE = 0.15    # within remaining 85%

# ============================================================================
# EXPERIMENT MODES
# ============================================================================

MODES = {
    "pooled": {
        # Tüm kuyulardaki window'ları birleştir, rastgele böl
        "description": "Pooled: tüm kuyuları karıştır, rastgele train/val/test split",
    },
    "cross_well": {
        # Bazı kuyular train, bazıları test (generalization testi)
        "description": "Cross-well: bir kuyuda eğit, farklı kuyuda test et",
        "train_wells": ["forge_78b32"],     # seismic monitoring well
        "test_wells": ["forge_16b32"],      # production well - hiç görmediği kuyu
    },
    "single_well_baseline": {
        # Mevcut projeyle karşılaştırma için: sadece 78B-32 üzerinde eğit
        "description": "Baseline: yalnız 78B-32 (mevcut projeyle karşılaştırma)",
        "wells": ["forge_78b32"],
    },
}

# ============================================================================
# DOWNLOAD CANDIDATES
# ============================================================================
# GDR submission sayfaları - direkt arşiv indirme için fallback liste.
# NOT: GDR bazen dinamik URL üretiyor; scripts/download_data.py bunları
# deneyecek, başarısız olursa kullanıcıya manuel talimat gösterecek.

GDR_DOWNLOAD_HINTS = {
    "forge_16a32": {
        "submission_id": 1292,
        "target_file_hint": "Schlumberger Logs CBL",
    },
    "forge_16b32": {
        "submission_id": 1531,
        "target_file_hint": "cement bond log CBL",
    },
    "forge_56_32": {
        "submission_id": 1295,
        "target_file_hint": "CBL sonic",
    },
}

# ============================================================================
# LOGGING
# ============================================================================

VERBOSE = True
# ============================================================================
# AUGMENTATION
# ============================================================================

AUGMENTATION_MULTIPLIER = 5.0
AUGMENTATION_MIN_CLASS_TARGET = 500

AUG_CONFIG = {
    "noise":      {"p": 0.5, "snr_db_range": (20.0, 35.0)},
    "time_shift": {"p": 0.4, "max_shift": 5},
    "amp_scale":  {"p": 0.5, "scale_range": (0.8, 1.2)},
    "depth_flip": {"p": 0.3},
    "mixup":      {"p": 0.3, "alpha": 0.2},
    "cutout":     {"p": 0.3, "max_depth_cut": 15, "max_time_cut": 30},
}