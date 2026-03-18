"""
train.py
Training script for both the Residual U-Net baseline and the 3D U-Net cine-loop model.

Usage:
    # Baseline (static frames):
    python src/train.py --mode baseline --img_type raw

    # Cine-loop (3D U-Net):
    python src/train.py --mode cine
"""

import os
import argparse
import math
import random
import glob
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from models import (
    build_residual_unet,
    build_3d_unet,
    dice_coef,
    dice_loss,
    hybrid_loss,
)

# ---------------------------
# REPRODUCIBILITY
# ---------------------------
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ---------------------------
# SHARED HYPERPARAMETERS
# ---------------------------
CROP_H = 256
CROP_W = 384
BATCH_SIZE = 16
LR = 3e-5
EPOCHS = 50
THRESH = 0.5
AUTOTUNE = tf.data.AUTOTUNE


# ============================================================
# BASELINE — DATA PIPELINE (static frames)
# ============================================================

def read_resize_baseline(img_path, mask_path, training=False):
    """Load, decode, resize, normalize image+mask. Optional augmentation for training."""
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.resize(img, [CROP_H, CROP_W], method='bilinear')
    img = tf.cast(img, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, [CROP_H, CROP_W], method='nearest')
    mask = tf.cast(mask > 127, tf.float32)

    if training:
        img = tf.image.random_flip_left_right(img)
        mask = tf.image.random_flip_left_right(mask)
        img = tf.image.random_flip_up_down(img)
        mask = tf.image.random_flip_up_down(mask)
        img = tf.clip_by_value(tf.image.random_brightness(img, 0.05), 0.0, 1.0)

    img.set_shape([CROP_H, CROP_W, 1])
    mask.set_shape([CROP_H, CROP_W, 1])
    return img, mask


def build_baseline_dataset(img_paths, mask_paths, batch_size, shuffle, training):
    """Build a tf.data.Dataset for baseline static-frame training."""
    ds = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
    if shuffle:
        ds = ds.shuffle(len(img_paths), seed=SEED)
    ds = ds.map(
        lambda i, m: tf.py_function(
            read_resize_baseline, [i, m, training], [tf.float32, tf.float32]
        ),
        num_parallel_calls=AUTOTUNE,
    )
    ds = ds.map(lambda i, m: (
        tf.ensure_shape(i, [CROP_H, CROP_W, 1]),
        tf.ensure_shape(m, [CROP_H, CROP_W, 1]),
    ))
    if training:
        ds = ds.repeat()
    return ds.batch(batch_size).prefetch(AUTOTUNE)


def get_baseline_paths(work_dir, split_csv, img_type, split):
    """Return (img_paths, mask_paths) for a given split and input variant."""
    import pandas as pd
    df = pd.read_csv(split_csv)
    df_split = df[df['split'] == split]

    folder_map = {
        'raw': 'resized',
        'seg': 'mask_combined',
        'imp': 'imposed',
        'skl': 'skeleton',
    }
    img_folder = os.path.join(work_dir, folder_map[img_type])
    mask_folder = os.path.join(work_dir, 'masks')

    img_paths, mask_paths = [], []
    for _, row in df_split.iterrows():
        img_name = os.path.basename(row['raw_image'])
        mask_name = os.path.basename(row['mask_image'])
        ip = os.path.join(img_folder, img_name)
        mp = os.path.join(mask_folder, mask_name)
        if os.path.exists(ip) and os.path.exists(mp):
            img_paths.append(ip)
            mask_paths.append(mp)
    return img_paths, mask_paths


# ============================================================
# CINE-LOOP — DATA PIPELINE (3D U-Net)
# ============================================================

SEQ_LEN = 16


def load_npz_as_arrays(path):
    """Load a preprocessed .npz file and return (X, Y) arrays (grayscale, resized)."""
    data = np.load(path)
    X = data["X"].astype(np.float32)   # (T, H, W, C)
    Y = (data["Y"] > 0.5).astype(np.uint8)

    # Force grayscale
    if X.shape[-1] != 1:
        X = X[..., :1]

    # Resize frames
    X_out = np.zeros((X.shape[0], CROP_H, CROP_W, 1), dtype=np.float32)
    for i in range(X.shape[0]):
        X_out[i] = tf.image.resize(X[i], (CROP_H, CROP_W), method='bilinear').numpy()
    return X_out, Y


def pad_or_crop_seq(X, Y, seq_len, deterministic=False):
    """Pad or randomly crop a sequence to exactly seq_len frames."""
    T = X.shape[0]
    if T >= seq_len:
        start = 0 if (deterministic or T == seq_len) else random.randint(0, T - seq_len)
        X, Y = X[start:start + seq_len], Y[start:start + seq_len]
    else:
        pad = seq_len - T
        X = np.concatenate([X, np.repeat(X[-1:], pad, axis=0)], axis=0)
        Y = np.concatenate([Y, np.repeat(Y[-1:], pad, axis=0)], axis=0)
    return X.astype(np.float32), Y[..., None].astype(np.float32)


def npz_generator(file_list, seq_len, shuffle=True, deterministic=False):
    files = list(file_list)
    if shuffle:
        random.shuffle(files)
    for p in files:
        X, Y = load_npz_as_arrays(p)
        Xs, Ys = pad_or_crop_seq(X, Y, seq_len, deterministic)
        yield Xs, Ys


def build_cine_dataset(file_list, seq_len, batch_size=1, training=True):
    """Build a tf.data.Dataset from preprocessed .npz cine-loop files."""
    C = 1
    sig = (
        tf.TensorSpec(shape=(seq_len, CROP_H, CROP_W, C), dtype=tf.float32),
        tf.TensorSpec(shape=(seq_len, CROP_H, CROP_W, 1), dtype=tf.float32),
    )
    ds = tf.data.Dataset.from_generator(
        lambda: npz_generator(file_list, seq_len,
                               shuffle=training, deterministic=not training),
        output_signature=sig,
    )
    if training:
        ds = ds.shuffle(buffer_size=min(4 * batch_size, max(1, len(file_list))), seed=SEED)
    return ds.batch(batch_size).prefetch(2)


# ============================================================
# TRAINING RUNNERS
# ============================================================

def train_baseline(args):
    """Train the Residual U-Net on static HC18 frames."""
    work_dir = args.work_dir
    split_csv = os.path.join(work_dir, "dataset_split.csv")
    model_dir = os.path.join(work_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    img_type = args.img_type
    train_imgs, train_masks = get_baseline_paths(work_dir, split_csv, img_type, 'train')
    val_imgs, val_masks = get_baseline_paths(work_dir, split_csv, img_type, 'val')

    print(f"Train: {len(train_imgs)} | Val: {len(val_imgs)} | Variant: {img_type}")

    train_ds = build_baseline_dataset(train_imgs, train_masks, BATCH_SIZE, True, True)
    val_ds = build_baseline_dataset(val_imgs, val_masks, BATCH_SIZE, False, False)

    model = build_residual_unet(input_shape=(CROP_H, CROP_W, 1))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss=hybrid_loss,
        metrics=[dice_coef],
    )

    steps_per_epoch = max(1, math.ceil(len(train_imgs) / BATCH_SIZE))
    val_steps = max(1, math.ceil(len(val_imgs) / BATCH_SIZE))

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(model_dir, f"resunet_{img_type}_best.h5"),
            monitor='val_dice_coef', mode='max', save_best_only=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_dice_coef', mode='max', factor=0.5, patience=6, verbose=1),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_dice_coef', mode='max', patience=12,
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.CSVLogger(
            os.path.join(model_dir, f"log_{img_type}.csv")),
    ]

    model.fit(train_ds, epochs=EPOCHS,
              steps_per_epoch=steps_per_epoch,
              validation_data=val_ds,
              validation_steps=val_steps,
              callbacks=callbacks, verbose=1)

    model.save(os.path.join(model_dir, f"resunet_{img_type}.keras"))
    print(f"Saved: resunet_{img_type}.keras")


def train_cine(args):
    """Train the 3D U-Net on preprocessed cine-loop .npz files."""
    preproc_dir = args.preproc_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    all_files = sorted(glob.glob(os.path.join(preproc_dir, "*.npz")))
    if args.sample_n:
        all_files = all_files[:args.sample_n]

    train_files, temp = train_test_split(all_files, test_size=0.25, random_state=SEED)
    val_files, _ = train_test_split(temp, test_size=0.20, random_state=SEED)

    print(f"Train: {len(train_files)} | Val: {len(val_files)}")

    train_ds = build_cine_dataset(train_files, SEQ_LEN, batch_size=1, training=True)
    val_ds = build_cine_dataset(val_files, SEQ_LEN, batch_size=1, training=False)

    model = build_3d_unet(SEQ_LEN, CROP_H, CROP_W, channels=1, base_filters=8)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=hybrid_loss,
        metrics=[dice_coef],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, "best_model.weights.h5"),
            save_weights_only=True, monitor='val_loss', save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', patience=2, factor=0.5, verbose=1),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5,
            restore_best_weights=True, verbose=1),
    ]

    model.fit(train_ds, validation_data=val_ds,
              epochs=args.epochs or EPOCHS, callbacks=callbacks)

    print("Training complete.")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fetal head segmentation models")
    parser.add_argument("--mode", choices=["baseline", "cine"], required=True)

    # Baseline args
    parser.add_argument("--img_type", choices=["raw", "seg", "imp", "skl"], default="raw",
                        help="Input variant for baseline model")
    parser.add_argument("--work_dir", default="data/preprocessed_baseline",
                        help="Root dir with dataset_split.csv and stage folders")

    # Cine args
    parser.add_argument("--preproc_dir", default="data/cine_preproc_npz/")
    parser.add_argument("--output_dir", default="data/segmentation_output/")
    parser.add_argument("--sample_n", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=None)

    args = parser.parse_args()

    if args.mode == "baseline":
        train_baseline(args)
    else:
        train_cine(args)
