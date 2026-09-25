"""Epipolar-constrained NCC stereo matching for aerial photogrammetry.

This module refactors a graduate Digital Photogrammetry coursework script.
It keeps the original geometric search-strip model and NCC matching concept,
while fixing portability, validation, thresholding, and visualization issues.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import csv

import cv2
import numpy as np


@dataclass(frozen=True)
class FlightGeometry:
    """Geometry used by the original coursework search-strip model."""
    ground_elevation_m: float = 1883.73
    focal_length_m: float = 153.24e-3
    camera_elevation_m: float = 1883.73 + 766.2
    pixel_size_m: float = 18e-6
    terrain_height_uncertainty_m: float = 15.0
    photo_base_m: float = 77.5e-3
    photo_scale: float = 1.0 / 5000.0


def load_points_csv(path: str | Path) -> np.ndarray:
    """Load point_id,x,y CSV as an Nx2 float array."""
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append([float(row["x"]), float(row["y"])])
    pts = np.asarray(rows, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("Point CSV must contain x and y columns.")
    return pts


def to_gray(image: np.ndarray) -> np.ndarray:
    """Convert image to single-channel float32 grayscale."""
    if image is None:
        raise ValueError("Input image is None.")
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3:
        if image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Unsupported image shape.")
    return gray.astype(np.float32)


def compute_epipolar_search_space(
    left_points: np.ndarray,
    geometry: FlightGeometry = FlightGeometry(),
) -> np.ndarray:
    """Compute horizontal search-strip bounds [xmin, xmax] for each left point.

    The equations are preserved from the coursework:

        B = b / s
        P_x = ((B*f)/(H-Zp))/pixel_size
        x_second = round(x_left - P_x)

        S = (b * H * Delta_z) / (H-Zp)^2 / pixel_size

        search = [x_second - floor(S/2), x_second + floor(S/2)]
    """
    pts = np.asarray(left_points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("left_points must have shape (N, 2).")

    g = geometry
    denom = g.camera_elevation_m - g.ground_elevation_m
    if denom <= 0:
        raise ValueError("Camera elevation must be above ground elevation.")
    if g.pixel_size_m <= 0 or g.photo_scale <= 0:
        raise ValueError("pixel_size_m and photo_scale must be positive.")

    ground_base = g.photo_base_m / g.photo_scale
    p_x = ((ground_base * g.focal_length_m) / denom) / g.pixel_size_m
    x_second = np.round(pts[:, 0] - p_x)

    strip_width = (
        (g.photo_base_m * (g.camera_elevation_m * g.terrain_height_uncertainty_m))
        / (denom**2)
    ) / g.pixel_size_m

    half = np.floor(strip_width / 2.0)
    return np.column_stack([x_second - half, x_second + half])


def normalized_cross_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Compute zero-mean normalized cross-correlation for two equal arrays."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("NCC inputs must have identical shape.")
    a0 = a - a.mean()
    b0 = b - b.mean()
    den = np.sqrt(np.sum(a0*a0) * np.sum(b0*b0))
    if den == 0:
        return 0.0
    return float(np.sum(a0*b0) / den)


def match_points_ncc(
    right_image: np.ndarray,
    left_image: np.ndarray,
    left_points: np.ndarray,
    x_search_space: np.ndarray,
    kernel_size: int,
    threshold: float = 0.4,
    y_half_window: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Match left-image points in the right image using epipolar-constrained NCC.

    Returns
    -------
    coords : (N, 2) float array
        Matched point centers. Unmatched points are [0, 0].
    scores : (N,) float array
        Maximum NCC score for each point, NaN when no valid search was possible.

    Notes
    -----
    The original coursework used a Python sliding-window loop. This refactor
    uses OpenCV's TM_CCOEFF_NORMED, which is the same zero-mean normalized
    cross-correlation criterion, but much faster.
    """
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd integer >= 3.")
    if not (-1.0 <= threshold <= 1.0):
        raise ValueError("threshold must be between -1 and 1.")
    if y_half_window <= 0:
        raise ValueError("y_half_window must be positive.")

    left = to_gray(left_image)
    right = to_gray(right_image)
    pts = np.asarray(left_points, dtype=float)
    xs = np.asarray(x_search_space, dtype=float)

    if len(pts) != len(xs):
        raise ValueError("left_points and x_search_space must contain the same N.")

    hl, wl = left.shape
    hr, wr = right.shape
    half = kernel_size // 2

    coords = np.zeros((len(pts), 2), dtype=float)
    scores = np.full(len(pts), np.nan, dtype=float)

    for idx, (x_f, y_f) in enumerate(pts):
        x, y = int(round(x_f)), int(round(y_f))

        # Template bounds are checked against the LEFT image dimensions.
        tx1, tx2 = x - half, x + half + 1
        ty1, ty2 = y - half, y + half + 1
        if tx1 < 0 or ty1 < 0 or tx2 > wl or ty2 > hl:
            continue

        templ = left[ty1:ty2, tx1:tx2]

        # Search bounds are checked independently against the RIGHT image.
        x_min = max(0, int(np.floor(xs[idx, 0])))
        x_max = min(wr, int(np.ceil(xs[idx, 1])))
        y_min = max(0, y - y_half_window)
        y_max = min(hr, y + y_half_window)

        search = right[y_min:y_max, x_min:x_max]
        if search.shape[0] < kernel_size or search.shape[1] < kernel_size:
            continue

        corr = cv2.matchTemplate(search, templ, cv2.TM_CCOEFF_NORMED)
        _, max_score, _, max_loc = cv2.minMaxLoc(corr)
        scores[idx] = float(max_score)

        # Corrected threshold logic: only positive best correlation matters.
        if max_score < threshold:
            continue

        top_left_x, top_left_y = max_loc
        coords[idx] = [
            x_min + top_left_x + half,
            y_min + top_left_y + half,
        ]

    return coords, scores


def evaluate_matches(
    matched: np.ndarray,
    ground_truth: np.ndarray,
) -> dict:
    """Compute valid-mask, per-point Euclidean error, and global RMSE."""
    matched = np.asarray(matched, dtype=float)
    gt = np.asarray(ground_truth, dtype=float)
    if matched.shape != gt.shape:
        raise ValueError("matched and ground_truth must have the same shape.")

    valid = ~np.all(matched == 0, axis=1)
    if not np.any(valid):
        return {
            "valid_mask": valid,
            "errors_px": np.array([], dtype=float),
            "rmse_px": float("nan"),
            "n_valid": 0,
        }

    diff = matched[valid] - gt[valid]
    errors = np.sqrt(np.sum(diff**2, axis=1))
    rmse = float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))

    return {
        "valid_mask": valid,
        "errors_px": errors,
        "rmse_px": rmse,
        "n_valid": int(np.sum(valid)),
    }


def ground_truth_coverage(
    x_search_space: np.ndarray,
    right_ground_truth: np.ndarray,
) -> np.ndarray:
    """Return whether each ground-truth x coordinate lies in its search strip."""
    xs = np.asarray(x_search_space, dtype=float)
    gt = np.asarray(right_ground_truth, dtype=float)
    return (gt[:, 0] >= xs[:, 0]) & (gt[:, 0] <= xs[:, 1])


def draw_search_regions(
    right_image: np.ndarray,
    left_points: np.ndarray,
    x_search_space: np.ndarray,
    y_half_window: int = 100,
) -> np.ndarray:
    """Draw the ACTUAL search rectangles used by match_points_ncc."""
    image = right_image.copy()
    if image.ndim == 2:
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    for (x, y), (xmin, xmax) in zip(left_points, x_search_space):
        p1 = (int(round(xmin)), int(round(y - y_half_window)))
        p2 = (int(round(xmax)), int(round(y + y_half_window)))
        cv2.rectangle(image, p1, p2, (0, 255, 0), 2)

    return image
