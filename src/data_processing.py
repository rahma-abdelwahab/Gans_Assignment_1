# src/data_processing.py
"""tf.data pipeline helpers for the Medical MNIST dataset.

Code Conventions
----------------
- Constants : imported from src.config (UPPER_SNAKE_CASE)
- Functions : snake_case with full docstrings (Args / Returns)
- Data      : tf.data.Dataset exclusively — no manual NumPy loops
"""

import pathlib
import random

import tensorflow as tf

from src.config import (
    AUTOTUNE,
    BATCH_SIZE,
    CHANNELS,
    IMG_SIZE,
    NOISE_STD,
    SEED,
)


def load_and_preprocess(path: tf.Tensor):
    """Read one image file, decode, resize, convert to grayscale in [0, 1].

    Args:
        path: tf.string scalar — full path to an image file.

    Returns:
        Tuple (img, img) where img has shape (IMG_SIZE, IMG_SIZE, 1)
        and dtype float32 with values in [0.0, 1.0].
    """
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.rgb_to_grayscale(img)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    img.set_shape([IMG_SIZE, IMG_SIZE, CHANNELS])
    return img, img


def add_noise(img: tf.Tensor, _target: tf.Tensor):
    """Corrupt a clean image with Gaussian noise for denoising experiments.

    Args:
        img:     Clean image tensor of shape (IMG_SIZE, IMG_SIZE, 1).
        _target: Ignored second element from the (img, img) pair.

    Returns:
        Tuple (noisy_img, clean_img), both shape (IMG_SIZE, IMG_SIZE, 1).
    """
    noise = tf.random.normal(
        shape=tf.shape(img), mean=0.0, stddev=NOISE_STD
    )
    noisy = tf.clip_by_value(img + noise, 0.0, 1.0)
    return noisy, img


def collect_paths(region_dir: pathlib.Path) -> list:
    """Gather all image file paths recursively from a region directory.

    Args:
        region_dir: pathlib.Path pointing to one class folder.

    Returns:
        Sorted, deduplicated list of absolute file path strings.
    """
    exts  = ["jpg", "jpeg", "png", "bmp"]
    paths = []
    for ext in exts:
        paths += list(region_dir.rglob(f"*.{ext}"))
        paths += list(region_dir.rglob(f"*.{ext.upper()}"))
    return sorted(set(str(p) for p in paths))


def make_dataset(
    paths: list,
    training: bool  = True,
    denoising: bool = False,
) -> tf.data.Dataset:
    """Build a batched, prefetched tf.data.Dataset from a list of file paths.

    Args:
        paths:     List of image file path strings.
        training:  If True, shuffle the dataset before batching.
        denoising: If True, map add_noise so pairs become (noisy, clean).

    Returns:
        tf.data.Dataset yielding (input_batch, target_batch) tensors of
        shape (BATCH_SIZE, IMG_SIZE, IMG_SIZE, 1).
    """
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
    if denoising:
        ds = ds.map(add_noise, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.shuffle(buffer_size=min(len(paths), 3000), seed=SEED)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds


def build_region_datasets(
    data_dir: pathlib.Path,
    region: str,
    val_split: float = 0.1,
) -> tuple:
    """Return (train_ds, val_ds, all_paths) for one anatomical region.

    Args:
        data_dir:  Root directory containing class-named sub-folders.
        region:    One of REGION_CLASSES.
        val_split: Fraction of data reserved for validation.

    Returns:
        train_ds   : tf.data.Dataset for training (shuffled, batched).
        val_ds     : tf.data.Dataset for validation (ordered, batched).
        all_paths  : Complete sorted list of file path strings.
    """
    region_dir = data_dir / region
    all_paths  = collect_paths(region_dir)

    rng = random.Random(SEED)
    rng.shuffle(all_paths)

    split    = int((1.0 - val_split) * len(all_paths))
    train_ds = make_dataset(all_paths[:split], training=True)
    val_ds   = make_dataset(all_paths[split:], training=False)

    print(
        f"  [{region}]  total={len(all_paths)}  "
        f"train={split}  val={len(all_paths) - split}"
    )
    return train_ds, val_ds, all_paths


def build_all_region_datasets(data_dir: pathlib.Path) -> dict:
    """Build train/val datasets for every anatomical region.

    Args:
        data_dir: Root directory containing class-named sub-folders.

    Returns:
        Dict mapping region name -> (train_ds, val_ds, all_paths).
    """
    print("Building region datasets:")
    region_datasets = {}
    from src.config import REGION_CLASSES
    for region in REGION_CLASSES:
        train_ds, val_ds, all_paths = build_region_datasets(
            data_dir, region
        )
        region_datasets[region] = (train_ds, val_ds, all_paths)
    return region_datasets