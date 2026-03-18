"""
preprocess_baseline.py
Static-frame preprocessing pipeline for the Residual U-Net baseline.

Pipeline stages:
  grayscale → resize → median/Gaussian/Wiener denoise → Sobel edge
  → k-means clustering → thresholding → morphological refinement
  → component filtering → skeletonization → mask imposition
"""

import os
import cv2
import numpy as np
import json
import glob
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
from scipy.signal import wiener
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from skimage import morphology

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------
# DEFAULT PARAMETERS
# ---------------------------
PARAMS = {
    "resize_hw": (540, 800),
    "median_kernel": 5,
    "gaussian_kernel": (7, 7),
    "gaussian_sigma": 1.0,
    "wiener_size": (7, 7),
    "sobel_ksize": 15,
    "kmeans_clusters": 3,
    "threshold_value": 50,
    "threshold_max": 255,
    "morph_kernel": (3, 3),
    "morph_iterations": 0,
    "mask_combination": "AND",
    "min_object_size": 400,
    "thinning": True,
    "random_state": 42,
}


# ---------------------------
# PREPROCESSING FUNCTIONS
# ---------------------------

def to_grayscale(img):
    """Convert to grayscale if needed."""
    if len(img.shape) == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def resize_image(img, hw=None):
    """Resize to consistent resolution."""
    hw = hw or PARAMS["resize_hw"]
    return cv2.resize(img, (hw[1], hw[0]))


def denoise_image(gray):
    """Apply median → Gaussian → Wiener filtering to suppress speckle noise."""
    img = cv2.medianBlur(gray, PARAMS["median_kernel"])
    img = cv2.GaussianBlur(img, PARAMS["gaussian_kernel"], PARAMS["gaussian_sigma"])
    img = wiener(img, PARAMS["wiener_size"])
    return np.uint8(np.clip(img, 0, 255))


def sobel_edge(img):
    """Compute Sobel gradient magnitude to emphasize skull boundary transitions."""
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=PARAMS["sobel_ksize"])
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=PARAMS["sobel_ksize"])
    grad = np.sqrt(gx ** 2 + gy ** 2)
    grad = (grad / (grad.max() + 1e-12)) * 255
    return np.uint8(grad)


def kmeans_segmentation(img):
    """Apply K-Means clustering for initial region segmentation."""
    X = img.reshape(-1, 1).astype(np.float32)
    km = KMeans(n_clusters=PARAMS["kmeans_clusters"], n_init=10,
                random_state=PARAMS["random_state"])
    labels = km.fit_predict(X)
    return labels.reshape(img.shape)


def apply_threshold(img):
    """Binary threshold to separate foreground from background."""
    _, mask = cv2.threshold(img, PARAMS["threshold_value"],
                            PARAMS["threshold_max"], cv2.THRESH_BINARY)
    return mask


def morphological_refinement(mask):
    """Morphological closing + opening to clean segmentation mask."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, PARAMS["morph_kernel"])
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,
                            iterations=PARAMS["morph_iterations"])
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,
                            iterations=PARAMS["morph_iterations"])
    return mask


def filter_components(mask):
    """Remove small isolated connected components below the size threshold."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= PARAMS["min_object_size"]:
            cleaned[labels == i] = 255
    return cleaned


def skeletonize_mask(mask):
    """Optionally thin the mask to a compact boundary skeleton."""
    if PARAMS["thinning"]:
        return morphology.thin(mask // 255).astype(np.uint8) * 255
    return mask


def combine_masks(mask1, mask2):
    """Combine two masks via AND (conservative) or OR."""
    if PARAMS["mask_combination"] == "AND":
        return cv2.bitwise_and(mask1, mask2)
    return cv2.bitwise_or(mask1, mask2)


def impose_mask(original, mask):
    """Apply binary mask to original image."""
    return cv2.bitwise_and(original, original, mask=mask)


# ---------------------------
# FULL PIPELINE PER IMAGE
# ---------------------------

def process_single_image(raw_path, split, output_dir):
    """
    Run the full preprocessing pipeline on one image.
    Saves intermediate stage outputs to output_dir/<stage>/<filename>.png.

    Returns a metrics dict or None on failure.
    """
    try:
        fname = os.path.splitext(os.path.basename(raw_path))[0]

        img = cv2.imread(raw_path)
        if img is None:
            return None

        resized = resize_image(img)
        gray = to_grayscale(resized)
        denoised = denoise_image(gray)
        sobel = sobel_edge(denoised)
        mask_kmeans = kmeans_segmentation(denoised)
        mask_thresh = apply_threshold(denoised)
        mask_combined = combine_masks((mask_kmeans.astype(np.uint8) * 255), mask_thresh)
        mask_refined = morphological_refinement(mask_combined)
        mask_clean = filter_components(mask_refined)
        skeleton = skeletonize_mask(mask_clean)
        imposed = impose_mask(sobel, skeleton)

        stages = {
            "resized": resized,
            "denoised": denoised,
            "sobel": sobel,
            "kmeans": (mask_kmeans * 255).astype(np.uint8),
            "threshold": mask_thresh,
            "mask_combined": mask_combined,
            "skeleton": skeleton,
            "imposed": imposed,
        }

        for stage_name, img_out in stages.items():
            stage_dir = os.path.join(output_dir, stage_name)
            os.makedirs(stage_dir, exist_ok=True)
            cv2.imwrite(os.path.join(stage_dir, f"{fname}.png"), img_out)

        return {"file": fname, "height": resized.shape[0],
                "width": resized.shape[1], "split": split}

    except Exception as e:
        print(f"Error processing {raw_path}: {e}")
        return None


def process_single_mask(mask_path, split, output_dir):
    """
    Preprocess a segmentation mask:
    fill contours → morphological refinement → resize → save to output_dir/masks/.
    """
    try:
        fname = os.path.splitext(os.path.basename(mask_path))[0]
        mask = cv2.imread(mask_path)
        if mask is None:
            return None

        gray = to_grayscale(mask)
        contours, _ = cv2.findContours(
            (gray > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        filled = np.zeros_like(gray)
        for cnt in contours:
            cv2.drawContours(filled, [cnt], 0, 255, thickness=-1)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel)
        filled = cv2.morphologyEx(filled, cv2.MORPH_OPEN, kernel)

        resized = resize_image(filled)
        resized = (resized > 127).astype(np.uint8) * 255

        masks_dir = os.path.join(output_dir, "masks")
        os.makedirs(masks_dir, exist_ok=True)
        cv2.imwrite(os.path.join(masks_dir, f"{fname}.png"), resized,
                    [cv2.IMWRITE_PNG_COMPRESSION, 3])

        return {"file": fname, "height": resized.shape[0],
                "width": resized.shape[1], "split": split}

    except Exception as e:
        print(f"Error processing mask {mask_path}: {e}")
        return None


# ---------------------------
# DATASET PREPARATION
# ---------------------------

def prepare_dataset_split(input_dir, output_dir, test_size=0.25, val_size=0.2, seed=42):
    """
    Pair raw images with annotations, split into train/val/test, and save a CSV.

    Returns:
        splits (dict): {'train': [(raw, mask), ...], 'val': [...], 'test': [...]}
    """
    all_files = sorted(glob.glob(os.path.join(input_dir, "*.*")))
    raw_files = [f for f in all_files if "annotation" not in f.lower()]
    mask_files = [f for f in all_files if "annotation" in f.lower()]

    pairs = []
    for raw in raw_files:
        base = os.path.splitext(os.path.basename(raw))[0]
        candidates = [
            m for m in mask_files
            if os.path.splitext(os.path.basename(m))[0].lower()
            == (base + "_annotation").lower()
        ]
        if len(candidates) == 1:
            pairs.append((raw, candidates[0]))

    indices = np.arange(len(pairs))
    train_idx, temp_idx = train_test_split(indices, test_size=test_size,
                                           random_state=seed, shuffle=True)
    val_idx, test_idx = train_test_split(temp_idx, test_size=val_size, random_state=seed)

    splits = {
        "train": [pairs[i] for i in train_idx],
        "val": [pairs[i] for i in val_idx],
        "test": [pairs[i] for i in test_idx],
    }

    df = pd.DataFrame(
        [(s, raw, mask) for s, items in splits.items() for (raw, mask) in items],
        columns=["split", "raw_image", "mask_image"],
    )
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "dataset_split.csv"), index=False)
    return splits


def preprocess_all(csv_path, output_dir, num_workers=None):
    """
    Run full preprocessing pipeline (images + masks) in parallel.

    Args:
        csv_path (str): Path to dataset_split.csv
        output_dir (str): Root output directory for stage folders
        num_workers (int): Number of parallel workers (default: cpu_count - 1)
    """
    df = pd.read_csv(csv_path)
    num_workers = num_workers or max(1, os.cpu_count() - 1)

    img_tasks = [(row.raw_image, row.split, output_dir) for row in df.itertuples(index=False)]
    mask_tasks = [(row.mask_image, row.split, output_dir) for row in df.itertuples(index=False)]

    print(f"Preprocessing {len(img_tasks)} images with {num_workers} workers...")

    img_results = []
    with ProcessPoolExecutor(max_workers=num_workers) as exe:
        futures = [exe.submit(process_single_image, *t) for t in img_tasks]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Images"):
            r = f.result()
            if r:
                img_results.append(r)

    pd.DataFrame(img_results).to_csv(
        os.path.join(output_dir, "preprocessing_metrics.csv"), index=False
    )

    mask_results = []
    with ProcessPoolExecutor(max_workers=num_workers) as exe:
        futures = [exe.submit(process_single_mask, *t) for t in mask_tasks]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Masks"):
            r = f.result()
            if r:
                mask_results.append(r)

    pd.DataFrame(mask_results).to_csv(
        os.path.join(output_dir, "mask_preprocessing_metrics.csv"), index=False
    )

    print("Preprocessing complete.")


if __name__ == "__main__":
    INPUT_DIR = "data/hc18/training_set/training_set"
    OUTPUT_DIR = "data/preprocessed_baseline"

    splits = prepare_dataset_split(INPUT_DIR, OUTPUT_DIR)
    print(f"Train: {len(splits['train'])} | Val: {len(splits['val'])} | Test: {len(splits['test'])}")

    csv_path = os.path.join(OUTPUT_DIR, "dataset_split.csv")
    preprocess_all(csv_path, OUTPUT_DIR)

    # Save params for reproducibility
    with open(os.path.join(OUTPUT_DIR, "preprocessing_params.json"), "w") as f:
        json.dump(PARAMS, f, indent=4)
