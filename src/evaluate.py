"""
evaluate.py
Evaluation pipeline: post-processing, ellipse-based HC computation, and metrics.

Covers both baseline (per-model: raw/seg/imp/skl) and cine-loop evaluation.
Outputs per-sample CSVs and a final summary table.

Usage:
    python src/evaluate.py --mode baseline --output_dir data/preprocessed_baseline
    python src/evaluate.py --mode cine     --output_dir data/segmentation_output
"""

import os
import argparse
import cv2
import numpy as np
import pandas as pd
from math import pi, sqrt
from sklearn.metrics import r2_score


# ---------------------------
# POST-PROCESSING
# ---------------------------

def keep_largest_component(mask):
    """Retain only the largest connected component in a binary mask."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return mask
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    out = np.zeros_like(mask)
    out[labels == largest] = 255
    return out


def postprocess_mask(mask):
    """
    Refine a raw segmentation mask:
      threshold → morphological close+open → largest component
    """
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return keep_largest_component(mask)


def postprocess_directory(pred_dir, post_dir):
    """Postprocess all masks in pred_dir and save to post_dir."""
    os.makedirs(post_dir, exist_ok=True)
    for fname in os.listdir(pred_dir):
        path = os.path.join(pred_dir, fname)
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.max() == 0:
            continue
        refined = postprocess_mask(mask.astype(np.uint8))
        cv2.imwrite(os.path.join(post_dir, fname), refined)


# ---------------------------
# BIOMETRIC COMPUTATION
# ---------------------------

def ellipse_circumference(a, b):
    """Ramanujan approximation of ellipse perimeter."""
    h = ((a - b) ** 2) / ((a + b) ** 2 + 1e-12)
    return pi * (a + b) * (1 + (3 * h) / (10 + sqrt(4 - 3 * h)))


def compute_hc_from_mask(mask, px_size_mm):
    """
    Fit an ellipse to the largest contour and return HC in mm.
    Returns None if no valid contour is found.
    """
    bin_mask = (mask > 127).astype(np.uint8)
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 5:
        return None
    (_, _), (MA, ma), _ = cv2.fitEllipse(largest)
    a, b = max(MA, ma) / 2, min(MA, ma) / 2
    return ellipse_circumference(a, b) * px_size_mm


# ---------------------------
# SEGMENTATION METRICS
# ---------------------------

def dice_score(y_true, y_pred):
    y_true = (y_true > 127).astype(np.uint8)
    y_pred = (y_pred > 127).astype(np.uint8)
    inter = np.sum(y_true * y_pred)
    return (2 * inter + 1e-6) / (np.sum(y_true) + np.sum(y_pred) + 1e-6)


def iou_score(y_true, y_pred):
    y_true = (y_true > 127).astype(np.uint8)
    y_pred = (y_pred > 127).astype(np.uint8)
    inter = np.sum(y_true * y_pred)
    union = np.sum(np.clip(y_true + y_pred, 0, 1))
    return (inter + 1e-6) / (union + 1e-6)


# ---------------------------
# BASELINE EVALUATION
# ---------------------------

def evaluate_baseline(output_dir, gt_csv, gt_mask_dir, target_size=(384, 256)):
    """
    Evaluate all four baseline model variants (raw, seg, imp, skl).

    Args:
        output_dir : Root directory containing pred_masks_post_<key>/ folders
        gt_csv     : Path to HC18 training_set_pixel_size_and_HC.csv
        gt_mask_dir: Path to HC18 training_set/ (for annotation PNGs)
        target_size: (W, H) resize target

    Returns:
        summary_df : DataFrame with one row per model variant
    """
    gt_df = pd.read_csv(gt_csv)
    model_keys = ["raw", "seg", "imp", "skl"]
    summary = []

    for key in model_keys:
        post_dir = os.path.join(output_dir, f"pred_masks_post_{key}")
        if not os.path.exists(post_dir):
            print(f"Skipping {key} — {post_dir} not found")
            continue

        records = []
        for _, row in gt_df.iterrows():
            fname = row["filename"]
            px_size = float(row["pixel size(mm)"])
            hc_true = float(row["head circumference (mm)"])

            pred_path = os.path.join(post_dir, fname)
            gt_path = os.path.join(gt_mask_dir, fname.replace("_HC.png", "_HC_Annotation.png"))

            if not os.path.exists(pred_path) or not os.path.exists(gt_path):
                continue

            pred = cv2.resize(cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE), target_size)
            gt = cv2.resize(cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE), target_size)

            hc_pred = compute_hc_from_mask(pred, px_size)
            if hc_pred is None:
                continue

            records.append({
                "filename": fname,
                "dice": dice_score(gt, pred),
                "iou": iou_score(gt, pred),
                "HC_pred_mm": hc_pred,
                "HC_true_mm": hc_true,
                "error_mm": hc_pred - hc_true,
            })

        if not records:
            continue

        df = pd.DataFrame(records)
        df.to_csv(os.path.join(output_dir, f"eval_{key}.csv"), index=False)

        mae = np.mean(np.abs(df["error_mm"]))
        rmse = np.sqrt(np.mean(df["error_mm"] ** 2))
        r2 = np.corrcoef(df["HC_true_mm"], df["HC_pred_mm"])[0, 1] ** 2

        summary.append({
            "Model": key.upper(),
            "Mean Dice (%)": round(df["dice"].mean() * 100, 2),
            "Mean IoU (%)": round(df["iou"].mean() * 100, 2),
            "MAE (mm)": round(mae, 2),
            "RMSE (mm)": round(rmse, 2),
            "R² (%)": round(r2 * 100, 2),
        })
        print(f"{key.upper()}: Dice={summary[-1]['Mean Dice (%)']:.2f}% | "
              f"MAE={mae:.2f}mm | RMSE={rmse:.2f}mm | R²={r2:.4f}")

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(output_dir, "final_summary.csv"), index=False)
    print("\n=== Baseline Summary ===")
    print(summary_df.to_string(index=False))
    return summary_df


# ---------------------------
# CINE-LOOP EVALUATION
# ---------------------------

def evaluate_cine(post_dir, cine_gt_dir, gt_mask_dir, gt_csv,
                  output_metrics, target_size=(384, 256)):
    """
    Evaluate the 3D U-Net cine-loop model on held-out HC18 test cases.

    Computes per-frame Dice/IoU, HC prediction, temporal stability, and reliability score.

    Args:
        post_dir       : Directory of postprocessed predicted masks
        cine_gt_dir    : Directory of ground-truth cine .avi files
        gt_mask_dir    : HC18 annotation PNG directory
        gt_csv         : HC18 pixel size + HC CSV
        output_metrics : Path to save results CSV
        target_size    : (W, H) resize target

    Returns:
        results_df: DataFrame with per-sample metrics
    """
    gt_df = pd.read_csv(gt_csv)
    results = []

    for pred_fname in os.listdir(post_dir):
        prefix = pred_fname.replace("_Phase4_frame0_mask.png", "")
        row = gt_df[gt_df["filename"] == f"{prefix}_HC.png"]
        if row.empty:
            continue

        px_size = float(row["pixel size(mm)"].iloc[0])
        hc_true = float(row["head circumference (mm)"].iloc[0])

        pred_path = os.path.join(post_dir, pred_fname)
        pred_mask = cv2.resize(cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE), target_size)

        # Load GT frames (cine or static fallback)
        gt_frames = []
        cine_path = os.path.join(cine_gt_dir, f"{prefix}_Phase4_mask.avi")
        if os.path.exists(cine_path):
            cap = cv2.VideoCapture(cine_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                gt_frames.append(cv2.resize(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), target_size
                ))
            cap.release()
        else:
            static = os.path.join(gt_mask_dir, f"{prefix}_HC_Annotation.png")
            if os.path.exists(static):
                gt_frames.append(cv2.resize(
                    cv2.imread(static, cv2.IMREAD_GRAYSCALE), target_size
                ))

        if not gt_frames:
            continue

        dice_list, iou_list, hc_list = [], [], []
        for gt_f in gt_frames:
            dice_list.append(dice_score(gt_f, pred_mask))
            iou_list.append(iou_score(gt_f, pred_mask))
            hc = compute_hc_from_mask(pred_mask, px_size)
            hc_list.append(hc if hc is not None else np.nan)

        hc_mean = np.nanmean(hc_list)
        temporal_std = np.nanstd(hc_list)
        reliability = 1.0 / (1.0 + temporal_std)

        results.append({
            "prefix": prefix,
            "HC_true_mm": hc_true,
            "HC_pred_mm": hc_mean,
            "error_mm": hc_mean - hc_true,
            "dice": np.nanmean(dice_list),
            "iou": np.nanmean(iou_list),
            "temporal_std_mm": temporal_std,
            "reliability": reliability,
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_metrics, index=False)

    print(f"\n=== Cine-Loop Evaluation ({len(results_df)} samples) ===")
    mae = np.mean(np.abs(results_df["error_mm"]))
    rmse = np.sqrt(np.mean(results_df["error_mm"] ** 2))
    print(f"Mean Dice : {results_df['dice'].mean():.4f}")
    print(f"Mean IoU  : {results_df['iou'].mean():.4f}")
    print(f"MAE       : {mae:.2f} mm")
    print(f"RMSE      : {rmse:.2f} mm")
    print(f"Reliability (mean): {results_df['reliability'].mean():.4f}")
    return results_df


# ---------------------------
# CLI
# ---------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fetal head segmentation models")
    parser.add_argument("--mode", choices=["baseline", "cine"], required=True)
    parser.add_argument("--output_dir", default="data/segmentation_output/")
    parser.add_argument("--gt_csv",
                        default="data/hc18/training_set/training_set_pixel_size_and_HC.csv")
    parser.add_argument("--gt_mask_dir",
                        default="data/hc18/training_set/training_set/")
    parser.add_argument("--cine_gt_dir",
                        default="data/cine_loops/")
    args = parser.parse_args()

    if args.mode == "baseline":
        evaluate_baseline(
            output_dir=args.output_dir,
            gt_csv=args.gt_csv,
            gt_mask_dir=args.gt_mask_dir,
        )
    else:
        # First postprocess
        pred_dir = os.path.join(args.output_dir, "pred_masks_png")
        post_dir = os.path.join(args.output_dir, "pred_masks_post")
        postprocess_directory(pred_dir, post_dir)

        evaluate_cine(
            post_dir=post_dir,
            cine_gt_dir=args.cine_gt_dir,
            gt_mask_dir=args.gt_mask_dir,
            gt_csv=args.gt_csv,
            output_metrics=os.path.join(args.output_dir, "final_eval_metrics.csv"),
        )
