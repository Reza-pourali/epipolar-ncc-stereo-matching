"""Run epipolar-constrained NCC matching on a stereo pair."""

from pathlib import Path
import argparse
import csv

import cv2

from src.stereo_matching import (
    FlightGeometry,
    compute_epipolar_search_space,
    evaluate_matches,
    load_points_csv,
    match_points_ncc,
)


def parse_args():
    p = argparse.ArgumentParser(description="Epipolar-constrained NCC stereo matching")
    p.add_argument("left_image", help="Path to the left/reference image")
    p.add_argument("right_image", help="Path to the right/search image")
    p.add_argument(
        "--left-points",
        default="data/left_points.csv",
        help="CSV with point_id,x,y for the left image",
    )
    p.add_argument(
        "--ground-truth",
        default="data/right_ground_truth.csv",
        help="CSV with point_id,x,y for right-image reference coordinates",
    )
    p.add_argument(
        "--kernels",
        nargs="+",
        type=int,
        default=[11, 21, 31, 41, 51, 61, 71, 81],
    )
    p.add_argument("--threshold", type=float, default=0.4)
    p.add_argument("--y-half-window", type=int, default=100)
    p.add_argument("--output", default="matching_results.csv")
    return p.parse_args()


def main():
    args = parse_args()

    left = cv2.imread(args.left_image, cv2.IMREAD_UNCHANGED)
    right = cv2.imread(args.right_image, cv2.IMREAD_UNCHANGED)
    if left is None:
        raise FileNotFoundError(args.left_image)
    if right is None:
        raise FileNotFoundError(args.right_image)

    points = load_points_csv(args.left_points)
    gt = load_points_csv(args.ground_truth)
    search = compute_epipolar_search_space(points, FlightGeometry())

    rows = []
    for k in args.kernels:
        matched, scores = match_points_ncc(
            right,
            left,
            points,
            search,
            kernel_size=k,
            threshold=args.threshold,
            y_half_window=args.y_half_window,
        )
        metrics = evaluate_matches(matched, gt)
        rows.append([k, metrics["rmse_px"], metrics["n_valid"]])
        print(f"k={k:2d}: RMSE={metrics['rmse_px']:.3f} px, valid={metrics['n_valid']}")

    with Path(args.output).open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["kernel_size", "rmse_px", "valid_matches"])
        w.writerows(rows)

    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
