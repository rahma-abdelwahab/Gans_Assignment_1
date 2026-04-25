# src/train.py
"""Training loops and Keras callbacks for AE and VAE models.

Separated from model files so training logic can be changed without
touching model architecture.

Code Conventions
----------------
- Constants : imported from src.config (UPPER_SNAKE_CASE)
- Functions : snake_case with full docstrings (Args / Returns)
- No raw NumPy loops over datasets — tf.data handles all iteration
"""

import time

import tensorflow as tf
from tensorflow import keras

from src.ae_model import build_ae
from src.config import MODELS_DIR, REGION_BETA, REGION_EPOCHS
from src.vae_model import VAEModel, build_vae

DEFAULT_EPOCHS = 30


def get_callbacks(region: str, model_type: str) -> list:
    """Return standard Keras callbacks for one training run.

    Monitors val_loss for AE and val_total_loss for VAE.
    mode='min' is set explicitly to support custom VAE metric names.

    Args:
        region:     Anatomical region name.
        model_type: 'ae' or 'vae'.

    Returns:
        List of keras.callbacks.Callback instances.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    monitor = "val_loss" if model_type == "ae" else "val_total_loss"

    return [
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode="min",
            patience=5,
            restore_best_weights=True,
            verbose=0,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=0,
        ),
    ]


def train_ae_for_region(
    region: str,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int = DEFAULT_EPOCHS,
) -> tuple:
    """Build and train a ConvAE for one anatomical region.

    Saves weights to models/<region>_ae.weights.h5 after training.

    Args:
        region:   Anatomical region class name.
        train_ds: Training tf.data.Dataset yielding (img, img) pairs.
        val_ds:   Validation tf.data.Dataset yielding (img, img) pairs.
        epochs:   Maximum number of training epochs.

    Returns:
        Tuple (ae_model, history, elapsed_seconds).
    """
    print(f"\n{'─' * 60}")
    print(f"[AE] Training region: {region}")
    print(f"{'─' * 60}")

    ae      = build_ae(region_name=region)
    t0      = time.time()
    history = ae.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=get_callbacks(region, "ae"),
        verbose=1,
    )
    elapsed = time.time() - t0

    weight_path = str(MODELS_DIR / f"{region}_ae.weights.h5")
    ae.save_weights(weight_path)
    print(f"  -> Done in {elapsed:.1f}s  |  saved: {weight_path}")
    return ae, history, elapsed


def train_vae_for_region(
    region: str,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    beta: float = 1.0,
    epochs: int = DEFAULT_EPOCHS,
) -> tuple:
    """Build and train a ConvVAE for one anatomical region.

    Uses a LambdaCallback to increment the KL annealing epoch counter
    at the end of each training epoch.

    Saves encoder and decoder weights to models/ after training.

    Args:
        region:   Anatomical region class name.
        train_ds: Training tf.data.Dataset yielding (img, img) pairs.
        val_ds:   Validation tf.data.Dataset yielding (img, img) pairs.
        beta:     KL divergence weight (beta-VAE formulation).
        epochs:   Maximum number of training epochs.

    Returns:
        Tuple (vae_model, history, elapsed_seconds).
    """
    print(f"\n{'─' * 60}")
    print(f"[VAE beta={beta}] Training region: {region}")
    print(f"{'─' * 60}")

    vae = build_vae(region_name=region, beta=beta)

    # LambdaCallback increments KL annealing counter each epoch
    kl_annealing_cb = keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: vae.current_epoch.assign(
            float(epoch + 1)
        )
    )

    t0      = time.time()
    history = vae.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=get_callbacks(region, "vae") + [kl_annealing_cb],
        verbose=1,
    )
    elapsed = time.time() - t0

    enc_path = str(MODELS_DIR / f"{region}_vae_enc.weights.h5")
    dec_path = str(MODELS_DIR / f"{region}_vae_dec.weights.h5")
    vae.encoder.save_weights(enc_path)
    vae.decoder.save_weights(dec_path)
    print(f"  -> Done in {elapsed:.1f}s")
    return vae, history, elapsed


def train_all_regions(
    region_datasets: dict,
    epochs_map: dict = None,
    beta_map: dict   = None,
) -> tuple:
    """Train one AE and one VAE for every anatomical region.

    Args:
        region_datasets: Dict mapping region -> (train_ds, val_ds, paths).
        epochs_map:      Optional dict region -> max epochs.
                         Defaults to config.REGION_EPOCHS.
        beta_map:        Optional dict region -> KL beta.
                         Defaults to config.REGION_BETA.

    Returns:
        Tuple (ae_results, vae_results) where each maps
        region -> (model, history, elapsed_seconds).
    """
    if epochs_map is None:
        epochs_map = REGION_EPOCHS
    if beta_map is None:
        beta_map = REGION_BETA

    ae_results  = {}
    vae_results = {}

    for region, (train_ds, val_ds, _) in region_datasets.items():
        n_epochs = epochs_map.get(region, DEFAULT_EPOCHS)
        vae_beta = beta_map.get(region, 1.0)

        ae_model,  ae_hist,  ae_t = train_ae_for_region(
            region, train_ds, val_ds, epochs=n_epochs
        )
        vae_model, vae_hist, vae_t = train_vae_for_region(
            region, train_ds, val_ds, beta=vae_beta, epochs=n_epochs
        )

        ae_results[region]  = (ae_model,  ae_hist,  ae_t)
        vae_results[region] = (vae_model, vae_hist, vae_t)

    print("\nAll regions trained successfully.")
    return ae_results, vae_results