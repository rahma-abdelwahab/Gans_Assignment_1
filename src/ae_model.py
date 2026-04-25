# src/ae_model.py
"""Convolutional Autoencoder (AE) model definition.

Separated from vae_model.py so each model can be imported,
tested, and modified independently.

Code Conventions
----------------
- Constants : imported from src.config (UPPER_SNAKE_CASE)
- Classes   : PascalCase
- Functions : snake_case with full docstrings (Args / Returns)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

from src.config import (
    CHANNELS,
    HIGH_DROPOUT_REGIONS,
    IMG_SIZE,
    LATENT_DIM,
    LR,
)


def build_ae(
    latent_dim: int  = LATENT_DIM,
    img_size: int    = IMG_SIZE,
    region_name: str = "ae",
) -> Model:
    """Build and compile a regularised Convolutional Autoencoder (AE).

    Architecture (64x64 input)
    --------------------------
    Encoder:
        RandomFlip + RandomRotation (augmentation)
        Conv2D(32,s=2)  -> BN -> LeakyReLU
        Conv2D(64,s=2)  -> BN -> LeakyReLU
        Conv2D(128,s=2) -> BN -> LeakyReLU
        Flatten -> Dense(256) -> Dropout -> Dense(latent_dim)

    Decoder:
        Dense(256) -> Dropout -> Dense(8*8*128) -> Reshape
        Conv2DT(128,s=2) -> BN -> LeakyReLU
        Conv2DT(64,s=2)  -> BN -> LeakyReLU
        Conv2DT(32,s=2)  -> BN -> LeakyReLU
        Conv2D(1, sigmoid)

    Args:
        latent_dim:  Size of the bottleneck vector.
        img_size:    Spatial side length of the square input image.
        region_name: Included in the Keras model name.

    Returns:
        Compiled Keras Model (Adam + MSE loss).
    """
    reg          = keras.regularizers.l2(1e-4)
    dropout_rate = 0.4 if region_name in HIGH_DROPOUT_REGIONS else 0.2

    # ── Input + augmentation ──────────────────────────────────────────────
    inp = keras.Input(
        shape=(img_size, img_size, CHANNELS), name="ae_input"
    )
    x = layers.RandomFlip("horizontal", name="aug_flip")(inp)
    x = layers.RandomRotation(0.05, name="aug_rot")(x)

    # ── Encoder ───────────────────────────────────────────────────────────
    x = layers.Conv2D(
        32, 3, strides=2, padding="same",
        use_bias=False, name="enc_conv1"
    )(x)
    x = layers.BatchNormalization(name="enc_bn1")(x)
    x = layers.LeakyReLU(0.2, name="enc_lrelu1")(x)

    x = layers.Conv2D(
        64, 3, strides=2, padding="same",
        use_bias=False, name="enc_conv2"
    )(x)
    x = layers.BatchNormalization(name="enc_bn2")(x)
    x = layers.LeakyReLU(0.2, name="enc_lrelu2")(x)

    x = layers.Conv2D(
        128, 3, strides=2, padding="same",
        use_bias=False, name="enc_conv3"
    )(x)
    x = layers.BatchNormalization(name="enc_bn3")(x)
    x = layers.LeakyReLU(0.2, name="enc_lrelu3")(x)

    pre_flat_shape = x.shape[1:]
    flat_dim = (
        pre_flat_shape[0] * pre_flat_shape[1] * pre_flat_shape[2]
    )

    x = layers.Flatten(name="enc_flatten")(x)
    x = layers.Dense(
        256, activation="relu",
        kernel_regularizer=reg, name="enc_fc"
    )(x)
    x = layers.Dropout(dropout_rate, name="enc_dropout")(x)
    z = layers.Dense(latent_dim, name="latent_z")(x)

    # ── Decoder ───────────────────────────────────────────────────────────
    x = layers.Dense(
        256, activation="relu",
        kernel_regularizer=reg, name="dec_fc"
    )(z)
    x = layers.Dropout(dropout_rate, name="dec_dropout")(x)
    x = layers.Dense(flat_dim, activation="relu", name="dec_dense")(x)
    x = layers.Reshape(pre_flat_shape, name="dec_reshape")(x)

    x = layers.Conv2DTranspose(
        128, 3, strides=2, padding="same",
        use_bias=False, name="dec_convT1"
    )(x)
    x = layers.BatchNormalization(name="dec_bn1")(x)
    x = layers.LeakyReLU(0.2, name="dec_lrelu1")(x)

    x = layers.Conv2DTranspose(
        64, 3, strides=2, padding="same",
        use_bias=False, name="dec_convT2"
    )(x)
    x = layers.BatchNormalization(name="dec_bn2")(x)
    x = layers.LeakyReLU(0.2, name="dec_lrelu2")(x)

    x = layers.Conv2DTranspose(
        32, 3, strides=2, padding="same",
        use_bias=False, name="dec_convT3"
    )(x)
    x = layers.BatchNormalization(name="dec_bn3")(x)
    x = layers.LeakyReLU(0.2, name="dec_lrelu3")(x)

    out = layers.Conv2D(
        CHANNELS, 3, padding="same",
        activation="sigmoid", name="ae_output"
    )(x)

    ae = Model(inp, out, name=f"AE_{region_name}")
    ae.compile(
        optimizer=keras.optimizers.Adam(LR),
        loss="mse",
    )
    return ae


def build_ae_encoder_submodel(ae_model: Model) -> Model:
    """Extract the encoder half of a trained AE as a standalone Model.

    Args:
        ae_model: A trained AE Keras model with a layer named 'latent_z'.

    Returns:
        Keras Model: image -> latent vector z, shape (B, latent_dim).
    """
    latent_out = ae_model.get_layer("latent_z").output
    return Model(ae_model.input, latent_out, name="ae_encoder_sub")


def build_ae_decoder_submodel(
    ae_model: Model,
    latent_dim: int = LATENT_DIM,
) -> Model:
    """Extract the decoder half of a trained AE as a standalone Model.

    Args:
        ae_model:   A trained AE Keras model.
        latent_dim: Size of the latent vector input.

    Returns:
        Keras Model: latent vector z -> reconstructed image.
    """
    latent_inp   = keras.Input(shape=(latent_dim,))
    x            = latent_inp
    after_latent = False
    for layer in ae_model.layers:
        if after_latent:
            x = layer(x)
        if layer.name == "latent_z":
            after_latent = True
    return Model(latent_inp, x, name="ae_decoder_sub")