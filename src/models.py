"""
models.py
Model architectures used in the fetal head segmentation project.

  - ResidualUNet : 2D encoder-decoder with residual blocks (baseline, static frames)
  - UNet3D       : Lightweight 3D U-Net for cine-loop sequences (temporal model)
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, Input


# ============================================================
# RESIDUAL U-NET (Baseline — static frames)
# ============================================================

def conv_block_2d(x, filters, kernel_size=3, activation='relu'):
    """Conv2D → BatchNorm → Activation."""
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    return x


def residual_block_2d(x, filters):
    """
    Residual block: two conv layers with a skip connection.
    Channel alignment handled automatically via 1×1 Conv if needed.
    """
    shortcut = x
    x = conv_block_2d(x, filters)
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def encoder_block_2d(x, filters):
    skip = residual_block_2d(x, filters)
    pooled = layers.MaxPooling2D((2, 2))(skip)
    return skip, pooled


def decoder_block_2d(x, skip, filters):
    x = layers.Conv2DTranspose(filters, (2, 2), strides=2, padding='same')(x)
    x = layers.Concatenate()([x, skip])
    x = residual_block_2d(x, filters)
    return x


def build_residual_unet(input_shape=(256, 384, 1)):
    """
    Residual U-Net for static ultrasound frame segmentation.

    Args:
        input_shape: (H, W, C) — default 256×384 grayscale

    Returns:
        tf.keras.Model with sigmoid output (binary segmentation map)
    """
    inputs = Input(shape=input_shape)

    # Encoder
    s1, p1 = encoder_block_2d(inputs, 64)
    s2, p2 = encoder_block_2d(p1, 128)
    s3, p3 = encoder_block_2d(p2, 256)
    s4, p4 = encoder_block_2d(p3, 512)

    # Bridge
    bridge = residual_block_2d(p4, 1024)

    # Decoder
    d1 = decoder_block_2d(bridge, s4, 512)
    d2 = decoder_block_2d(d1, s3, 256)
    d3 = decoder_block_2d(d2, s2, 128)
    d4 = decoder_block_2d(d3, s1, 64)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(d4)

    return Model(inputs, outputs, name="Residual_U-Net")


# ============================================================
# 3D U-NET (Cine-loop model — temporal sequences)
# ============================================================

def conv_block_3d(x, filters, kernel_size=(3, 3, 3), padding='same'):
    """Two Conv3D layers with BatchNorm and ReLU."""
    x = layers.Conv3D(filters, kernel_size, padding=padding, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(filters, kernel_size, padding=padding, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    return x


def encoder_block_3d(x, filters):
    """Conv3D block + spatial-only MaxPool (preserves temporal dimension)."""
    skip = conv_block_3d(x, filters)
    pooled = layers.MaxPool3D(pool_size=(1, 2, 2))(skip)
    return skip, pooled


def decoder_block_3d(x, skip, filters):
    """Spatial-only upsampling + skip connection + Conv3D block."""
    x = layers.UpSampling3D(size=(1, 2, 2))(x)
    x = layers.Concatenate()([x, skip])
    x = conv_block_3d(x, filters)
    return x


def build_3d_unet(seq_len, h, w, channels=1, base_filters=8):
    """
    Lightweight 3D U-Net for cine-loop segmentation.

    Processes short video clips (seq_len frames) jointly via 3D convolutions,
    enforcing temporal consistency across the sequence.

    Args:
        seq_len   : Number of frames per clip (e.g. 16)
        h, w      : Spatial dimensions (e.g. 256, 384)
        channels  : Input channels (1 = grayscale)
        base_filters: Base filter count — doubles at each encoder level

    Returns:
        tf.keras.Model outputting (seq_len, H, W, 1) sigmoid masks
    """
    inp = Input(shape=(seq_len, h, w, channels), name='input_seq')

    # Encoder
    c1, p1 = encoder_block_3d(inp, base_filters)
    c2, p2 = encoder_block_3d(p1, base_filters * 2)
    c3, p3 = encoder_block_3d(p2, base_filters * 4)

    # Bridge
    bridge = conv_block_3d(p3, base_filters * 8)

    # Decoder
    d3 = decoder_block_3d(bridge, c3, base_filters * 4)
    d2 = decoder_block_3d(d3, c2, base_filters * 2)
    d1 = decoder_block_3d(d2, c1, base_filters)

    out = layers.Conv3D(1, (1, 1, 1), activation='sigmoid', name='mask_pred')(d1)

    return Model(inputs=inp, outputs=out, name="3D_U-Net")


# ============================================================
# LOSSES & METRICS
# ============================================================

def dice_coef(y_true, y_pred, smooth=1e-6):
    """Dice coefficient metric."""
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth
    )


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def hybrid_loss(y_true, y_pred):
    """0.5 × BCE + 0.5 × Dice — balances pixel accuracy with contour overlap."""
    bce = tf.keras.losses.binary_crossentropy(
        tf.cast(y_true, tf.float32),
        tf.cast(y_pred, tf.float32)
    )
    return 0.5 * bce + 0.5 * dice_loss(y_true, y_pred)


if __name__ == "__main__":
    # Quick smoke test
    print("=== Residual U-Net ===")
    baseline = build_residual_unet(input_shape=(256, 384, 1))
    baseline.summary()

    print("\n=== 3D U-Net ===")
    cine_model = build_3d_unet(seq_len=16, h=256, w=384, channels=1, base_filters=8)
    cine_model.summary()
