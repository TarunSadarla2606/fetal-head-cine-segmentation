"""
preprocess_cine.py
Sequence-consistent preprocessing pipeline for cine-loop data.

Extends the baseline static-frame pipeline to operate on temporal sequences:
  - Isotropic pad + resize
  - CLAHE contrast normalization (uniform across frames)
  - Temporal smoothing (sliding-window mean)
  - Sobel edge channel + frame-difference channel
  - ROI cropping from mask union
  - Sequence-consistent augmentation (same affine applied to all frames)

Outputs .npz files with shape X: (T, H, W, C), Y: (T, H, W)
"""

import os
import glob
import random
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

import numpy as np
import cv2
from scipy.ndimage import median_filter
from skimage import exposure

# ---------------------------
# DEFAULT CONFIG
# ---------------------------
TARGET_H, TARGET_W = 256, 384
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)
TEMPORAL_SMOOTH_K = 3
ADD_EDGE_CHANNEL = True
APPLY_OPTICAL_FLOW = False

clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)


# ---------------------------
# FRAME-LEVEL UTILITIES
# ---------------------------

def isotropic_pad_and_resize(img, target_h, target_w, interp=cv2.INTER_LINEAR):
    """Pad isotropically to preserve aspect ratio, then resize to (target_h, target_w)."""
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    return cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                              borderType=cv2.BORDER_CONSTANT, value=0)


def apply_clahe_to_frame(img_uint8):
    """Apply CLAHE to a single grayscale uint8 frame."""
    return clahe.apply(img_uint8)


def sobel_magnitude(stack):
    """
    Compute Sobel gradient magnitude per frame.
    stack: (T, H, W) float32 in [0, 1] → returns (T, H, W) float32
    """
    out = np.empty_like(stack, dtype=np.float32)
    for t in range(stack.shape[0]):
        frame_u8 = (stack[t] * 255).astype(np.uint8)
        gx = cv2.Sobel(frame_u8, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(frame_u8, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        out[t] = mag / (mag.max() + 1e-12)
    return out


def temporal_smoothing_stack(stack, k=3):
    """
    Apply 1D sliding-window mean along the temporal axis.
    stack: (T, H, W) or (T, H, W, C) → same shape, float32
    """
    if k <= 1:
        return stack.copy().astype(np.float32)
    stack_f = np.asarray(stack, dtype=np.float32)
    T = stack_f.shape[0]
    pad = k // 2
    pad_width = [(pad, pad)] + [(0, 0)] * (stack_f.ndim - 1)
    padded = np.pad(stack_f, pad_width=pad_width, mode='reflect')
    out = np.empty_like(stack_f)
    for t in range(T):
        out[t] = np.mean(padded[t:t + k], axis=0)
    return out


def frame_difference_channel(stack):
    """
    Compute absolute frame-to-frame differences.
    stack: (T, H, W) float32 → (T, H, W) float32
    """
    stack = np.asarray(stack, dtype=np.float32)
    T = stack.shape[0]
    if T <= 1:
        return np.zeros_like(stack)
    diffs = np.abs(stack[1:] - stack[:-1])
    return np.concatenate([diffs, diffs[-1:]], axis=0)


def compute_optical_flow_stack(stack):
    """
    Dense optical flow magnitude per consecutive frame pair.
    stack: (T, H, W) float32 → (T, H, W) float32
    """
    stack = np.asarray(stack, dtype=np.float32)
    T = stack.shape[0]
    if T <= 1:
        return np.zeros_like(stack)
    flows = []
    prev = (stack[0] * 255).astype(np.uint8)
    for t in range(1, T):
        curr = (stack[t] * 255).astype(np.uint8)
        flow = cv2.calcOpticalFlowFarneback(prev, curr, None,
                                            pyr_scale=0.5, levels=3, winsize=15,
                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        flows.append(mag / (mag.max() + 1e-12))
        prev = curr
    flows = np.stack(flows, axis=0)
    return np.concatenate([flows, flows[-1:]], axis=0).astype(np.float32)


# ---------------------------
# SEQUENCE-CONSISTENT AUGMENTATION
# ---------------------------

def random_affine_params(max_rot_deg=10, max_trans_px=20, scale_range=(0.95, 1.05)):
    """Sample random affine parameters to apply consistently across all frames."""
    return (
        random.uniform(-max_rot_deg, max_rot_deg),
        random.uniform(-max_trans_px, max_trans_px),
        random.uniform(-max_trans_px, max_trans_px),
        random.uniform(*scale_range),
    )


def apply_affine_seq(frames, rot, tx, ty, scale, interp=cv2.INTER_LINEAR):
    """
    Apply the same affine transform to all frames.
    frames: (T, H, W) float32 or uint8 → same shape and dtype
    """
    frames = np.asarray(frames)
    T, H, W = frames.shape[:3]
    M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), rot, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    out = np.empty_like(frames)
    for t in range(T):
        src = frames[t]
        if np.issubdtype(src.dtype, np.floating):
            tmp = (np.clip(src, 0.0, 1.0) * 255).astype(np.uint8)
            warped = cv2.warpAffine(tmp, M, (W, H), flags=interp,
                                    borderMode=cv2.BORDER_REFLECT)
            out[t] = warped.astype(np.float32) / 255.0
        else:
            out[t] = cv2.warpAffine(src, M, (W, H), flags=interp,
                                    borderMode=cv2.BORDER_REFLECT)
    return out.astype(frames.dtype)


def random_brightness_contrast_seq(frames, brightness_delta=0.05, contrast_range=(0.9, 1.1)):
    """Apply uniform brightness + contrast jitter to all frames."""
    b = random.uniform(-brightness_delta, brightness_delta)
    c = random.uniform(*contrast_range)
    return np.clip(frames * c + b, 0.0, 1.0)


# ---------------------------
# VIDEO READING
# ---------------------------

def read_video_frames_gray(video_path, max_frames=None):
    """Read all frames of an .avi into a (T, H, W) uint8 array."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    success, frame = cap.read()
    while success:
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)
        if max_frames and len(frames) >= max_frames:
            break
        success, frame = cap.read()
    cap.release()
    if not frames:
        return np.zeros((0, 1, 1), dtype=np.uint8)
    return np.stack(frames, axis=0)


# ---------------------------
# PER-SEQUENCE PREPROCESSING
# ---------------------------

def preprocess_sequence_pair(video_path, mask_video_path, out_dir):
    """
    Preprocess one (image cine-loop, mask cine-loop) pair and save as .npz.

    Output:
        X: (T, H, W, C) float32  — preprocessed image channels
        Y: (T, H, W) uint8 {0,1} — binary skull mask
    """
    base = os.path.basename(video_path).replace('.avi', '')
    out_path = os.path.join(out_dir, base + ".npz")

    try:
        if os.path.exists(out_path):
            return (True, base, "exists")

        imgs = read_video_frames_gray(video_path)
        masks = read_video_frames_gray(mask_video_path)

        if imgs.size == 0:
            return (False, base, "no frames in video")
        if masks.size == 0:
            return (False, base, "no frames in mask video")

        T = min(imgs.shape[0], masks.shape[0])
        imgs, masks = imgs[:T], masks[:T]

        # Resize
        imgs_rs = np.stack([isotropic_pad_and_resize(imgs[t], TARGET_H, TARGET_W)
                            for t in range(T)], axis=0)
        masks_rs = np.stack([isotropic_pad_and_resize(masks[t], TARGET_H, TARGET_W,
                                                       interp=cv2.INTER_NEAREST)
                             for t in range(T)], axis=0)
        masks_rs = (masks_rs > 127).astype(np.uint8) * 255

        # CLAHE + temporal smoothing
        imgs_clahe = np.stack([apply_clahe_to_frame(imgs_rs[t]) for t in range(T)], axis=0)
        imgs_f = imgs_clahe.astype(np.float32) / 255.0
        imgs_med = np.stack([
            median_filter((imgs_f[t] * 255).astype(np.uint8), size=3)
            for t in range(T)
        ], axis=0).astype(np.float32) / 255.0
        imgs_temporal = temporal_smoothing_stack(imgs_med, k=TEMPORAL_SMOOTH_K)

        # Feature channels
        edge_chan = sobel_magnitude(imgs_temporal) if ADD_EDGE_CHANNEL else None
        diff_chan = frame_difference_channel(imgs_temporal)
        flow_chan = compute_optical_flow_stack(imgs_temporal) if APPLY_OPTICAL_FLOW else None

        # ROI crop from mask union
        mask_union = (masks_rs > 127).any(axis=0).astype(np.uint8) * 255
        ys, xs = np.where(mask_union > 0)
        if ys.size == 0:
            cy, cx = TARGET_H // 2, TARGET_W // 2
            y0, y1 = max(0, cy - TARGET_H // 4), min(TARGET_H, cy + TARGET_H // 4)
            x0, x1 = max(0, cx - TARGET_W // 4), min(TARGET_W, cx + TARGET_W // 4)
        else:
            y0 = max(0, ys.min() - 8)
            y1 = min(TARGET_H, ys.max() + 8)
            x0 = max(0, xs.min() - 8)
            x1 = min(TARGET_W, xs.max() + 8)

        def resize_stack(stack, interp=cv2.INTER_LINEAR):
            if stack is None:
                return None
            out = np.empty((T, TARGET_H, TARGET_W), dtype=np.float32)
            for t in range(T):
                tmp = np.clip(stack[t, y0:y1, x0:x1] * 255, 0, 255).astype(np.uint8)
                out[t] = cv2.resize(tmp, (TARGET_W, TARGET_H),
                                    interpolation=interp).astype(np.float32) / 255.0
            return out

        imgs_final = resize_stack(imgs_temporal)
        masks_crop = masks_rs[:, y0:y1, x0:x1]
        masks_final = np.stack([
            (cv2.resize(masks_crop[t], (TARGET_W, TARGET_H),
                        interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8) * 255
            for t in range(T)
        ], axis=0)

        edge_final = resize_stack(edge_chan) if edge_chan is not None else None
        diff_final = resize_stack(diff_chan)
        flow_final = resize_stack(flow_chan) if flow_chan is not None else None

        # Augmentation
        rot, tx, ty, scale = random_affine_params(max_rot_deg=5, max_trans_px=12,
                                                   scale_range=(0.98, 1.02))
        imgs_final = apply_affine_seq(imgs_final, rot, tx, ty, scale).astype(np.float32)
        masks_final = apply_affine_seq(masks_final, rot, tx, ty, scale,
                                        interp=cv2.INTER_NEAREST).astype(np.uint8)
        imgs_final = random_brightness_contrast_seq(imgs_final, 0.03, (0.98, 1.02))

        # Stack channels
        channels = [np.clip(imgs_final, 0, 1)[..., None]]
        if edge_final is not None:
            channels.append(np.clip(edge_final, 0, 1)[..., None])
        channels.append(np.clip(diff_final, 0, 1)[..., None])
        if flow_final is not None:
            channels.append(np.clip(flow_final, 0, 1)[..., None])

        X = np.concatenate(channels, axis=-1).astype(np.float32)
        Y = (masks_final > 127).astype(np.uint8)

        np.savez_compressed(out_path, X=X, Y=Y, meta=np.array([base], dtype='S'))
        return (True, base, None)

    except Exception as e:
        tb = traceback.format_exc()
        return (False, base, tb)


# ---------------------------
# MAIN RUNNER
# ---------------------------

def run_preprocessing_all(input_dir, mask_suffix, out_dir,
                           sample_limit=None, num_workers=4):
    """
    Preprocess all cine-loop pairs and save as .npz files.

    Args:
        input_dir (str): Directory containing .avi cine-loops
        mask_suffix (str): Suffix distinguishing mask videos (e.g. '_mask.avi')
        out_dir (str): Output directory for .npz files
        sample_limit (int): Limit number of sequences (None = all)
        num_workers (int): Number of parallel workers
    """
    video_files = sorted([
        v for v in glob.glob(os.path.join(input_dir, "*.avi"))
        if not v.endswith(mask_suffix)
    ])

    if sample_limit:
        video_files = video_files[:sample_limit]

    pairs = [(v, v.replace(".avi", mask_suffix))
             for v in video_files
             if os.path.exists(v.replace(".avi", mask_suffix))]

    if not pairs:
        raise FileNotFoundError("No video+mask pairs found.")

    os.makedirs(out_dir, exist_ok=True)
    print(f"Preprocessing {len(pairs)} sequences → {out_dir}")

    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as exe:
        futures = {exe.submit(preprocess_sequence_pair, v, m, out_dir): (v, m)
                   for v, m in pairs}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing"):
            try:
                results.append(fut.result())
            except Exception as e:
                v, _ = futures[fut]
                results.append((False, os.path.basename(v), str(e)))

    ok = sum(1 for r in results if r[0])
    print(f"Done: {ok}/{len(results)} succeeded.")


if __name__ == "__main__":
    run_preprocessing_all(
        input_dir="data/cine_loops/",
        mask_suffix="_mask.avi",
        out_dir="data/cine_preproc_npz/",
        sample_limit=400,
        num_workers=4,
    )
