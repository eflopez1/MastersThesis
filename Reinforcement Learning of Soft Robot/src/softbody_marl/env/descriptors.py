"""Global shape and contact descriptors (diagnostics / centralized critic / optional actor input)."""

from __future__ import annotations

import numpy as np


def body_orientation(mode: str, active_pos: np.ndarray, centroid: np.ndarray, centroid_vel: np.ndarray,
                     membrane_pos: np.ndarray, min_speed: float = 1e-3) -> float:
    """Reference angle of the body frame.

    world: 0 (descriptor is translation-invariant only).
    bot0: angle of bot_0 about the centroid (rotation-covariant with the material).
    velocity: heading of the centroid velocity; falls back to 0 below `min_speed`.
    pca: major principal axis of the membrane points (sign fixed to +x half-plane; 180 deg ambiguity).
    """
    if mode == "world":
        return 0.0
    if mode == "bot0":
        d = active_pos[0] - centroid
        return float(np.arctan2(d[1], d[0]))
    if mode == "velocity":
        if np.linalg.norm(centroid_vel) < min_speed:
            return 0.0
        return float(np.arctan2(centroid_vel[1], centroid_vel[0]))
    if mode == "pca":
        x = membrane_pos - membrane_pos.mean(axis=0)
        w, v = np.linalg.eigh(x.T @ x)
        axis = v[:, np.argmax(w)]
        if axis[0] < 0:
            axis = -axis
        return float(np.arctan2(axis[1], axis[0]))
    raise ValueError(f"Unknown orientation mode '{mode}'")


def rotate(points: np.ndarray, angle: float) -> np.ndarray:
    c, s = np.cos(-angle), np.sin(-angle)
    return points @ np.array([[c, -s], [s, c]]).T


def radial_shape(rel_points: np.ndarray, num_sectors: int, length_scale: float) -> np.ndarray:
    """Per angular sector: [mean r, min r, max r, occupancy fraction]; radii / length_scale.

    `rel_points` must already be centroid-relative and in the body frame. Empty sectors are all zeros
    (occupancy 0 disambiguates). Radial sectors cannot represent folds or re-entrant (non-star-shaped)
    contours: several boundary points may share a sector at different radii.
    """
    r = np.linalg.norm(rel_points, axis=1) / length_scale
    th = np.mod(np.arctan2(rel_points[:, 1], rel_points[:, 0]), 2 * np.pi)
    sector = np.minimum((th / (2 * np.pi) * num_sectors).astype(int), num_sectors - 1)
    out = np.zeros((num_sectors, 4))
    n = len(r)
    for k in range(num_sectors):
        sel = r[sector == k]
        if sel.size:
            out[k] = (sel.mean(), sel.min(), sel.max(), sel.size / n)
    return out.ravel()


def fourier_shape(ordered_points: np.ndarray, centroid: np.ndarray, num_harmonics: int, length_scale: float) -> np.ndarray:
    """Magnitudes |c_k|/L for k = +1..+H and -1..-H of the topologically ordered contour.

    Uses ring order (not angle), so folded/concave contours remain well defined. Magnitudes are
    invariant to translation (centroid removed), rotation, and choice of starting point.
    """
    z = (ordered_points[:, 0] - centroid[0]) + 1j * (ordered_points[:, 1] - centroid[1])
    c = np.fft.fft(z) / len(z)
    h = num_harmonics
    pos = np.abs(c[1 : h + 1])
    neg = np.abs(c[-1 : -h - 1 : -1])
    return np.concatenate([pos, neg]) / length_scale


def contact_map(active_contact: np.ndarray, num_sectors: int) -> np.ndarray:
    """Fraction of active bots in contact per ordered sector of the ring (bot index order)."""
    return np.array([s.mean() if s.size else 0.0 for s in np.array_split(active_contact.astype(float), num_sectors)])


def polygon_area(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polygon_perimeter(points: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)))


def points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Even-odd ray casting, vectorised over points."""
    x, y = points[:, 0:1], points[:, 1:2]
    px, py = polygon[:, 0], polygon[:, 1]
    qx, qy = np.roll(px, -1), np.roll(py, -1)
    crosses = (py > y) != (qy > y)
    denom = np.where(qy - py == 0, 1e-300, qy - py)
    x_int = px + (y - py) * (qx - px) / denom
    return (np.sum(crosses & (x < x_int), axis=1) % 2) == 1
