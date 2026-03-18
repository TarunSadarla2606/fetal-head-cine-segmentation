"""
pseudo_lddm.py
Enhanced Pseudo-Latent Dynamic Diffusion Model (Pseudo-LDDM)
Converts static HC18 ultrasound images into temporally coherent cine-loop sequences.

Three levels of temporal realism:
  1. Anatomically constrained motion (rigid + non-rigid)
  2. Hierarchical temporal dynamics (sinusoidal flicker + stochastic drift)
  3. Artifact-aware image perturbation (Rician speckle, shadow, TGC drift)
"""

import os
import glob
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

# ---------------------------
# DEFAULT CONFIG
# ---------------------------
DEFAULT_CONFIG = {
    "image_dir": "data/hc18/training_set/training_set/",
    "output_dir": "data/cine_loops/",
    "sample_limit": 400,       # set None to process all
    "max_workers": 8,
    "fps": 15,
    "total_seconds": 10,
    "params": {
        "max_rot": 5.0,
        "max_trans": 15,
        "jitter": 0.7,
        "speckle": 0.04,
        "brightness_drift": 0.05,
        "shadow_attenuation": 0.15,
        "tgc_drift": True,
        "non_rigid_scale_max": 1.5,
        "motion_type": "random_walk",
        "unique_flows": True
    }
}


# ---------------------------
# UTILITY FUNCTIONS
# ---------------------------

def fill_annotation_to_mask(ann_path, target_shape=None):
    """Load annotation image and fill the interior to produce a binary mask (0/255)."""
    ann = cv2.imread(ann_path, cv2.IMREAD_GRAYSCALE)
    if ann is None:
        return None
    _, th = cv2.threshold(ann, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(th, dtype=np.uint8)
    if contours:
        cv2.drawContours(mask, contours, -1, 255, thickness=-1)
    if target_shape is not None and (mask.shape[0], mask.shape[1]) != tuple(target_shape):
        mask = cv2.resize(mask, (target_shape[1], target_shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    return mask


def create_shadow_mask(H, W, M_skull, ty_t, shadow_attenuation):
    """Create a soft vertical shadow mask beneath the skull region."""
    skull_translated = cv2.warpAffine(
        M_skull.astype(np.uint8),
        np.float32([[1, 0, 0], [0, 1, ty_t]]),
        (W, H)
    ) > 128
    rows = np.where(skull_translated)[0]
    deepest_row = np.max(rows) if rows.size > 0 else H
    shadow_raw = np.zeros((H, W), dtype=np.float32)
    if deepest_row < H:
        for r in range(deepest_row, H):
            shadow_raw[r, :] = (r - deepest_row) / max(1, (H - deepest_row))
    shadow_mask = gaussian_filter(shadow_raw, sigma=5)
    base_mask = gaussian_filter(M_skull.astype(np.float32), sigma=10)
    shadow_mask = shadow_mask * base_mask
    return np.clip(shadow_mask * shadow_attenuation, 0.0, shadow_attenuation)


def apply_tgc_drift(I_base_float, H, W, tgc_center_t):
    """Add depth-varying TGC-like brightness profile."""
    Y = np.arange(H)
    Y_grid = np.tile(Y[:, np.newaxis], (1, W))
    tgc_profile = np.exp(-((Y_grid - tgc_center_t) / 50) ** 2) * 0.05
    return np.clip(I_base_float + tgc_profile, 0, 1)


def add_rician_speckle(image, strength):
    """Add Rician-approximated speckle noise."""
    N1 = np.random.normal(0, strength, image.shape)
    return np.clip(image + N1, 0, 1)


# ---------------------------
# MOTION GENERATION
# ---------------------------

def create_motion_for_phase(height, width, total_frames, params, fps=15):
    """Generate per-frame motion arrays: rotation, translation, flow, brightness, TGC."""
    rng = np.random.default_rng(42)

    # Rotation random walk
    rot_noise = rng.normal(0, 0.03, total_frames)
    theta_raw = np.cumsum(rot_noise)
    theta = (theta_raw / (np.max(np.abs(theta_raw)) + 1e-12)) * params['max_rot']

    # Translation random walk
    raw_tx = np.cumsum(rng.uniform(-0.1, 0.1, total_frames))
    raw_ty = np.cumsum(rng.uniform(-0.1, 0.1, total_frames))
    tx_drift = gaussian_filter(raw_tx, sigma=fps * 2)
    ty_drift = gaussian_filter(raw_ty, sigma=fps * 2)
    tx = (tx_drift / (np.max(np.abs(tx_drift)) + 1e-12)) * params['max_trans']
    ty = (ty_drift / (np.max(np.abs(ty_drift)) + 1e-12)) * params['max_trans']

    # Jitter
    jitter = params.get('jitter', 0)
    tx += rng.uniform(-jitter, jitter, total_frames)
    ty += rng.uniform(-jitter, jitter, total_frames)

    # Non-rigid flow (spatial, tiled over time)
    base_flow_x = gaussian_filter(rng.uniform(-1, 1, (height, width)), sigma=20)
    base_flow_y = gaussian_filter(rng.uniform(-1, 1, (height, width)), sigma=20)
    flow_x_array = np.tile(base_flow_x, (total_frames, 1, 1))
    flow_y_array = np.tile(base_flow_y, (total_frames, 1, 1))
    non_rigid_scale = np.ones(total_frames) * (params['non_rigid_scale_max'] / 3.0)

    # Brightness drift
    brightness_scale = np.zeros(total_frames)
    if params.get('brightness_drift', 0) > 0:
        gain_noise = rng.normal(0, 0.01, total_frames)
        brightness_scale = np.clip(
            np.cumsum(gain_noise),
            -params['brightness_drift'],
            params['brightness_drift']
        )

    # TGC center drift
    tgc_center = np.ones(total_frames) * (height * 0.5)
    if params.get('tgc_drift', False):
        tgc_noise = rng.normal(0, 0.005, total_frames)
        tgc_drift = np.cumsum(tgc_noise)
        tgc_center = (tgc_drift / (np.max(np.abs(tgc_drift)) + 1e-12)) * height * 0.2 + height * 0.5

    return theta, tx, ty, flow_x_array, flow_y_array, non_rigid_scale, brightness_scale, tgc_center


# ---------------------------
# PER-IMAGE WORKER
# ---------------------------

def process_video_worker(image_path, output_dir, config):
    """
    Worker function: generates one cine-loop (image + mask .avi) from a static HC18 image.
    Returns (success, filename, error_message).
    """
    fname = os.path.basename(image_path)
    base = fname.replace('_HC.png', '')
    ann_path = image_path.replace('.png', '_Annotation.png')

    if not os.path.exists(ann_path):
        return (False, fname, "Annotation not found")

    I = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if I is None:
        return (False, fname, "Could not load image")

    M_filled = fill_annotation_to_mask(ann_path, target_shape=I.shape)
    if M_filled is None:
        return (False, fname, "Mask creation failed")

    H, W = I.shape
    params = config['params']
    fps = config.get('fps', 15)
    total_frames = config.get('total_seconds', 10) * fps

    theta, tx, ty, flow_x, flow_y, non_rigid_scale, brightness_scale, tgc_center = \
        create_motion_for_phase(H, W, total_frames, params, fps=fps)

    os.makedirs(output_dir, exist_ok=True)
    img_out_path = os.path.join(output_dir, f"{base}_Phase4.avi")
    mask_out_path = os.path.join(output_dir, f"{base}_Phase4_mask.avi")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer_img = cv2.VideoWriter(img_out_path, fourcc, fps, (W, H), isColor=False)
    writer_mask = cv2.VideoWriter(mask_out_path, fourcc, fps, (W, H), isColor=False)

    if not writer_img.isOpened() or not writer_mask.isOpened():
        writer_img.release()
        writer_mask.release()
        return (False, fname, "VideoWriter open failed")

    I_base_float = (I.astype(np.float32) / 255.0).copy()
    M_bin = (M_filled > 127).astype(np.uint8) * 255
    X_grid, Y_grid = np.meshgrid(np.arange(W), np.arange(H))

    try:
        for t in range(total_frames):
            I_current = np.clip(I_base_float + brightness_scale[t], 0, 1)
            if params.get('tgc_drift'):
                I_current = apply_tgc_drift(I_current, H, W, tgc_center[t])

            R = cv2.getRotationMatrix2D((W / 2, H / 2), float(theta[t]), 1.0)
            R[0, 2] += float(tx[t])
            R[1, 2] += float(ty[t])

            X_rigid = R[0, 0] * X_grid + R[0, 1] * Y_grid + R[0, 2]
            Y_rigid = R[1, 0] * X_grid + R[1, 1] * Y_grid + R[1, 2]

            M_soft = gaussian_filter(M_bin.astype(np.float32), sigma=5) / 255.0
            X_final = (X_rigid + flow_x[t] * non_rigid_scale[t] * M_soft).astype(np.float32)
            Y_final = (Y_rigid + flow_y[t] * non_rigid_scale[t] * M_soft).astype(np.float32)

            I_warp = cv2.remap(I_current.astype(np.float32), X_final, Y_final,
                               interpolation=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)
            M_warp = cv2.remap(M_bin.astype(np.uint8), X_final, Y_final,
                               interpolation=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)

            if params.get('shadow_attenuation', 0) > 0:
                shadow = create_shadow_mask(H, W, M_bin, ty[t], params['shadow_attenuation'])
                I_warp = I_warp * (1.0 - shadow)

            I_final = add_rician_speckle(I_warp, strength=params.get('speckle', 0))

            writer_img.write((np.clip(I_final, 0, 1) * 255).astype(np.uint8))
            writer_mask.write((M_warp > 127).astype(np.uint8) * 255)

        writer_img.release()
        writer_mask.release()
        return (True, fname, None)

    except Exception as e:
        writer_img.release()
        writer_mask.release()
        return (False, fname, f"Exception: {e}")


# ---------------------------
# MAIN RUNNER
# ---------------------------

def run_all_videos(config):
    """
    Generate cine-loops for all HC18 images using parallel processing.

    Args:
        config (dict): Configuration dict. See DEFAULT_CONFIG for structure.
    """
    image_dir = config['image_dir']
    output_dir = config['output_dir']
    max_workers = config.get('max_workers', 4)
    sample_limit = config.get('sample_limit', None)

    image_files = sorted(glob.glob(os.path.join(image_dir, "*_HC.png")))
    if not image_files:
        raise FileNotFoundError(f"No images found in {image_dir} matching '*_HC.png'")

    if sample_limit:
        image_files = image_files[:sample_limit]

    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating {len(image_files)} cine-loops -> {output_dir}")

    start = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = {
            exe.submit(process_video_worker, img, output_dir, config): img
            for img in image_files
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating"):
            try:
                results.append(future.result())
            except Exception as e:
                src = futures[future]
                results.append((False, os.path.basename(src), str(e)))

    success = sum(1 for r in results if r[0])
    elapsed = (time.time() - start) / 60
    print(f"Done. {success}/{len(image_files)} succeeded in {elapsed:.1f} min")

    failures = [(f, m) for ok, f, m in results if not ok]
    if failures:
        print(f"Failed ({len(failures)}):")
        for f, m in failures[:5]:
            print(f"  {f}: {m}")


if __name__ == "__main__":
    run_all_videos(DEFAULT_CONFIG)
