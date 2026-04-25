# src/config.py
"""Central configuration — all hyper-parameters and paths defined once.

Import this module in every other src file so constants are never
duplicated across files.

Code Conventions
----------------
- All constants : UPPER_SNAKE_CASE
- No logic here : only assignments and imports
"""

import pathlib

# ── Image settings ────────────────────────────────────────────────────────────
IMG_SIZE   = 64
CHANNELS   = 1

# ── Model settings ────────────────────────────────────────────────────────────
LATENT_DIM = 32

# ── Training settings ─────────────────────────────────────────────────────────
BATCH_SIZE  = 128
LR          = 1e-3
VAE_LR      = 2e-4
NOISE_STD   = 0.15
DROPOUT     = 0.4
SEED        = 42

# ── VAE-specific settings ─────────────────────────────────────────────────────
FREE_BITS     = 0.1   # minimum KL nats per latent dimension
WARMUP_EPOCHS = 15    # epochs to linearly ramp beta 0 → target

# ── Per-region KL weight ──────────────────────────────────────────────────────
REGION_BETA = {
    "AbdomenCT": 1.0,
    "BreastMRI": 0.5,
    "CXR":       0.5,
    "ChestCT":   1.0,
    "Hand":      0.5,
    "HeadCT":    0.5,
}

# ── Per-region max epochs ─────────────────────────────────────────────────────
REGION_EPOCHS = {
    "AbdomenCT": 30,
    "BreastMRI": 50,   # complex MRI texture needs more epochs
    "CXR":       30,
    "ChestCT":   30,
    "Hand":      30,
    "HeadCT":    30,
}

# ── Dropout per region ────────────────────────────────────────────────────────
HIGH_DROPOUT_REGIONS = {"BreastMRI", "Hand", "HeadCT", "CXR"}

# ── Region class names ────────────────────────────────────────────────────────
REGION_CLASSES = [
    "AbdomenCT",
    "BreastMRI",
    "ChestCT",
    "CXR",
    "Hand",
    "HeadCT",
]

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = pathlib.Path("data/raw")
MODELS_DIR = pathlib.Path("models")

# ── tf.data ───────────────────────────────────────────────────────────────────
import tensorflow as tf  # noqa: E402
AUTOTUNE = tf.data.AUTOTUNE