# src/vae_model.py
"""Variational Autoencoder (VAE) model definition.

Separated from ae_model.py so each model is independently testable.
Contains SamplingLayer, VAEModel, and build_vae factory.

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
    DROPOUT,
    FREE_BITS,
    IMG_SIZE,
    LATENT_DIM,
    REGION_BETA,
    VAE_LR,
    WARMUP_EPOCHS,
)


class SamplingLayer(layers.Layer):
    """Reparameterisation trick: z = mu + eps * exp(0.5 * log_var).

    Clamps mu to [-3, 3] and log_var to [-4, 4] to prevent
    numerical explosion during early training (NaN prevention).

    Inputs
    ------
    [mu, log_var] : list of two tensors, each shape (batch, latent_dim).

    Returns
    -------
    z : sampled latent vector, shape (batch, latent_dim).
    """

    def call(self, inputs):
        mu, log_var = inputs
        mu      = tf.clip_by_value(mu,      -3.0,  3.0)
        log_var = tf.clip_by_value(log_var, -4.0,  4.0)
        batch   = tf.shape(mu)[0]
        dim     = tf.shape(mu)[1]
        eps     = tf.random.normal(shape=(batch, dim))
        return mu + tf.exp(0.5 * log_var) * eps


class VAEModel(Model):
    """Variational Autoencoder with KL annealing and free bits.

    ELBO Loss:
        total = recon_loss + beta(t) * kl_loss

        recon_loss = mean over batch of summed per-pixel MSE
        kl_loss    = mean over batch of sum_d max(KL_d, free_bits)
        beta(t)    = beta_target * min(epoch / warmup_epochs, 1.0)

    Design decisions:
        KL annealing  — prevents posterior collapse early in training
        Free bits     — prevents any single latent dim collapsing to 0
        Grad clipping — prevents exploding gradients (NaN protection)

    Args:
        encoder:       Keras Model returning (mu, log_var, z).
        decoder:       Keras Model mapping z -> reconstruction.
        beta:          Target KL weight after warmup.
        warmup_epochs: Epochs to linearly ramp beta from 0 to target.
        free_bits:     Minimum KL nats per latent dimension.
    """

    def __init__(
        self,
        encoder: Model,
        decoder: Model,
        beta: float        = 1.0,
        warmup_epochs: int = WARMUP_EPOCHS,
        free_bits: float   = FREE_BITS,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder       = encoder
        self.decoder       = decoder
        self.beta_target   = beta
        self.warmup_epochs = warmup_epochs
        self.free_bits     = free_bits
        self.current_epoch = tf.Variable(0.0, trainable=False)

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.recon_loss_tracker = keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker    = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.recon_loss_tracker,
            self.kl_loss_tracker,
        ]

    def _current_beta(self) -> tf.Tensor:
        """Linearly annealed beta: 0 -> beta_target over warmup_epochs.

        Returns:
            Scalar tensor — current effective KL weight.
        """
        progress = tf.minimum(
            self.current_epoch / self.warmup_epochs, 1.0
        )
        return self.beta_target * progress

    def _compute_losses(self, x: tf.Tensor, training: bool):
        """Shared forward pass for train_step and test_step.

        Args:
            x:        Input image batch, shape (B, H, W, 1).
            training: Forwarded to sub-models.

        Returns:
            total_loss, recon_loss, kl_loss — scalar tensors.
        """
        mu, log_var, z = self.encoder(x, training=training)
        mu      = tf.clip_by_value(mu,      -3.0,  3.0)
        log_var = tf.clip_by_value(log_var, -4.0,  4.0)
        x_hat   = self.decoder(z, training=training)

        # MSE reconstruction: sum over pixels, average over batch
        recon = tf.reduce_mean(
            tf.reduce_sum(tf.square(x - x_hat), axis=(1, 2, 3))
        )

        # KL per dimension with free bits floor
        kl_per_dim = -0.5 * (
            1.0 + log_var - tf.square(mu) - tf.exp(log_var)
        )
        kl_per_dim = tf.maximum(kl_per_dim, self.free_bits)
        kl = tf.reduce_mean(tf.reduce_sum(kl_per_dim, axis=1))

        beta  = self._current_beta()
        total = recon + beta * kl
        return total, recon, kl

    def train_step(self, data):
        x, _ = data
        with tf.GradientTape() as tape:
            total, recon, kl = self._compute_losses(x, training=True)
        grads = tape.gradient(total, self.trainable_weights)
        # Gradient clipping — prevents NaN from exploding gradients
        grads, _ = tf.clip_by_global_norm(grads, clip_norm=1.0)
        self.optimizer.apply_gradients(
            zip(grads, self.trainable_weights)
        )
        self.total_loss_tracker.update_state(total)
        self.recon_loss_tracker.update_state(recon)
        self.kl_loss_tracker.update_state(kl)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, _ = data
        total, recon, kl = self._compute_losses(x, training=False)
        self.total_loss_tracker.update_state(total)
        self.recon_loss_tracker.update_state(recon)
        self.kl_loss_tracker.update_state(kl)
        return {m.name: m.result() for m in self.metrics}


def build_vae(
    latent_dim: int    = LATENT_DIM,
    img_size: int      = IMG_SIZE,
    beta: float        = 1.0,
    warmup_epochs: int = WARMUP_EPOCHS,
    region_name: str   = "vae",
) -> VAEModel:
    """Build and compile a regularised Convolutional VAE.

    Architecture (64x64 input)
    --------------------------
    Encoder:
        RandomFlip + RandomRotation
        Conv2D(32,s=2)  -> BN -> LeakyReLU
        Conv2D(64,s=2)  -> BN -> LeakyReLU
        Conv2D(128,s=2) -> BN -> LeakyReLU
        Flatten -> Dense(256) -> Dropout
        Dense(latent_dim) [mu]      small weight init
        Dense(latent_dim) [log_var] small weight init
        SamplingLayer              [z]

    Decoder:
        Dense(256) -> Dropout -> Dense(8*8*128) -> Reshape
        Conv2DT(128,s=2) -> BN -> LeakyReLU
        Conv2DT(64,s=2)  -> BN -> LeakyReLU
        Conv2DT(32,s=2)  -> BN -> LeakyReLU
        Conv2D(1, sigmoid)

    Args:
        latent_dim:    Size of the bottleneck latent space.
        img_size:      Spatial side length of the square input image.
        beta:          Target KL loss weight.
        warmup_epochs: Epochs to warm beta from 0 to target.
        region_name:   Included in the Keras model name.

    Returns:
        Compiled VAEModel (Adam LR=2e-4, custom ELBO loss).
    """
    reg     = keras.regularizers.l2(1e-4)
    spatial = img_size // 8

    # ── Encoder ───────────────────────────────────────────────────────────
    enc_inp = keras.Input(
        shape=(img_size, img_size, CHANNELS), name="vae_enc_input"
    )
    x = layers.RandomFlip("horizontal", name="vae_aug_flip")(enc_inp)
    x = layers.RandomRotation(0.05, name="vae_aug_rot")(x)

    x = layers.Conv2D(
        32, 3, strides=2, padding="same",
        use_bias=False, name="venc_conv1"
    )(x)
    x = layers.BatchNormalization(name="venc_bn1")(x)
    x = layers.LeakyReLU(0.2, name="venc_lrelu1")(x)

    x = layers.Conv2D(
        64, 3, strides=2, padding="same",
        use_bias=False, name="venc_conv2"
    )(x)
    x = layers.BatchNormalization(name="venc_bn2")(x)
    x = layers.LeakyReLU(0.2, name="venc_lrelu2")(x)

    x = layers.Conv2D(
        128, 3, strides=2, padding="same",
        use_bias=False, name="venc_conv3"
    )(x)
    x = layers.BatchNormalization(name="venc_bn3")(x)
    x = layers.LeakyReLU(0.2, name="venc_lrelu3")(x)

    x = layers.Flatten(name="venc_flatten")(x)
    x = layers.Dense(
        256, activation="relu",
        kernel_regularizer=reg, name="venc_fc"
    )(x)
    x = layers.Dropout(DROPOUT, name="venc_dropout")(x)

    # Small weight init — keeps mu/log_var near 0 at start → prevents NaN
    mu = layers.Dense(
        latent_dim,
        kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
        bias_initializer="zeros",
        name="mu",
    )(x)
    log_var = layers.Dense(
        latent_dim,
        kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
        bias_initializer="zeros",
        name="log_var",
    )(x)

    z       = SamplingLayer(name="z")([mu, log_var])
    encoder = Model(
        enc_inp, [mu, log_var, z], name=f"VAE_enc_{region_name}"
    )

    # ── Decoder ───────────────────────────────────────────────────────────
    dec_inp = keras.Input(shape=(latent_dim,), name="vae_dec_input")
    x = layers.Dense(
        256, activation="relu",
        kernel_regularizer=reg, name="vdec_fc"
    )(dec_inp)
    x = layers.Dropout(DROPOUT, name="vdec_dropout")(x)
    x = layers.Dense(
        spatial * spatial * 128,
        activation="relu", name="vdec_dense"
    )(x)
    x = layers.Reshape((spatial, spatial, 128), name="vdec_reshape")(x)

    x = layers.Conv2DTranspose(
        128, 3, strides=2, padding="same",
        use_bias=False, name="vdec_convT1"
    )(x)
    x = layers.BatchNormalization(name="vdec_bn1")(x)
    x = layers.LeakyReLU(0.2, name="vdec_lrelu1")(x)

    x = layers.Conv2DTranspose(
        64, 3, strides=2, padding="same",
        use_bias=False, name="vdec_convT2"
    )(x)
    x = layers.BatchNormalization(name="vdec_bn2")(x)
    x = layers.LeakyReLU(0.2, name="vdec_lrelu2")(x)

    x = layers.Conv2DTranspose(
        32, 3, strides=2, padding="same",
        use_bias=False, name="vdec_convT3"
    )(x)
    x = layers.BatchNormalization(name="vdec_bn3")(x)
    x = layers.LeakyReLU(0.2, name="vdec_lrelu3")(x)

    dec_out = layers.Conv2D(
        CHANNELS, 3, padding="same",
        activation="sigmoid", name="vdec_output"
    )(x)
    decoder = Model(
        dec_inp, dec_out, name=f"VAE_dec_{region_name}"
    )

    vae = VAEModel(
        encoder, decoder,
        beta=beta,
        warmup_epochs=warmup_epochs,
        name=f"VAE_{region_name}_b{beta}",
    )
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=VAE_LR))
    return vae