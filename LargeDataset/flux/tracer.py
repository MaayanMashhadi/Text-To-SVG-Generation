from dataclasses import dataclass
import numpy as np
import cv2
import math
from typing import List, Optional, Tuple
@dataclass
class AutoParams:
    open_kernel: int = 3
    open_iters: int  = 1
    close_iters: int = 1
    speck_min_abs: int = 1200
    speck_min_rel: float = 0.001
    drop_border: bool = True
    eps_rel: float = 0.0025
    # post-trace filters
    min_path_area_px: int = 300
    min_path_bbox_frac: float = 0.0005   
    min_circularity: float = 0.02        
    min_solidity: float = 0.35           
    # speckle targets
    max_components: int = 120            
    target_small_frac: float = 0.006     
    # safety caps
    max_open_kernel: int = 9
    max_open_iters: int = 3
    max_close_iters: int = 2
    min_eps_rel: float = 0.0012
    max_eps_rel: float = 0.01


def _apply_morph(bw: np.ndarray, k: int, open_iters: int, close_iters: int):
    if k <= 1 or (open_iters == 0 and close_iters == 0):
        return bw
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    out = bw.copy()
    if open_iters:
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kern, iterations=open_iters)
    if close_iters:
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kern, iterations=close_iters)
    return out


def _component_stats(bw: np.ndarray):
    """
    Connected components on a binary mask.
    Returns: (n_labels, labels, stats, on_pixels, total_pixels)
    stats shape: [n_labels, 5] (LEFT, TOP, WIDTH, HEIGHT, AREA)
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    on = int(bw.sum())
    return n, labels, stats, on, bw.size


def _filter_components(bw: np.ndarray,
                       labels: np.ndarray,
                       stats: np.ndarray,
                       drop_border: bool,
                       min_abs: int,
                       min_rel: int) -> np.ndarray:
    """
    Keep only components that pass size/border rules.
    IMPORTANT: uses *the same* labels/stats you pass in (no recompute).
    """
    H, W = bw.shape[:2]
    n = stats.shape[0]                    
    keep = np.zeros(n, dtype=bool)
    keep[0] = False                        # background

    for i in range(1, n):
        x, y, w, h, a = stats[i]
        drop = (a < min_abs) or (a < min_rel)
        if drop_border and not drop:
            if x == 0 or y == 0 or (x + w) >= W or (y + h) >= H:
                drop = True
        keep[i] = not drop

    out = ((labels > 0) & keep[labels]).astype(np.uint8)
    return out


def _auto_clean_mask(mask01: np.ndarray, p: AutoParams) -> tuple[np.ndarray, AutoParams]:
    """
    Auto-increase morphology + size thresholds until speckle is suppressed,
    but cap to not kill thin details.
    """
    H, W = mask01.shape[:2]
    bw = (mask01.astype(np.uint8) > 0).astype(np.uint8)

    params = AutoParams(**p.__dict__)
    for _ in range(6):  
        if params.open_kernel > 1 and (params.open_iters or params.close_iters):
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params.open_kernel, params.open_kernel))
            m = bw * 255
            if params.open_iters:
                m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=params.open_iters)
            if params.close_iters:
                m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=params.close_iters)
            m = (m > 0).astype(np.uint8)
        else:
            m = bw

        n, labels, stats, on, total = _component_stats(m)
        if on == 0:
            bw = m
            break

        min_rel_px = max(1, int(on * params.speck_min_rel))
        m2 = _filter_components(m, labels, stats, params.drop_border, params.speck_min_abs, min_rel_px)

        n2, _, stats2, on2, _ = _component_stats(m2)
        small_frac = 1.0 - (on2 / max(on, 1))

        if (n2 <= params.max_components) and (small_frac <= params.target_small_frac):
            bw = m2
            break

        if params.open_kernel < params.max_open_kernel:
            params.open_kernel += 2
        elif params.open_iters < params.max_open_iters:
            params.open_iters += 1
        elif params.close_iters < params.max_close_iters:
            params.close_iters += 1

        params.speck_min_abs = int(params.speck_min_abs * 1.25)
        params.speck_min_rel = min(0.01, params.speck_min_rel * 1.25)

        bw = m2

    return bw, params

def _poly_metrics(cnt: np.ndarray):
    if len(cnt) < 3:
        return 0.0, 0.0, 0.0, 0.0
    area = float(cv2.contourArea(cnt))
    peri = float(cv2.arcLength(cnt, True))
    circ = (4.0 * math.pi * area / (peri * peri)) if peri > 0 else 0.0
    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    sol = (area / hull_area) if hull_area > 0 else 0.0
    return area, peri, circ, sol

def trace_contours_paths_autotuned(mask01: np.ndarray, width: int, height: int, base_params: Optional[AutoParams]=None) -> List[str]:
    """
    Auto-clean + trace with adaptive epsilon and post-filters.
    """
    if base_params is None:
        base_params = AutoParams()

    cleaned, tuned = _auto_clean_mask(mask01, base_params)
    if cleaned.max() == 0:
        return []

    n_comp, _, _, on_px, _ = _component_stats(cleaned)
    density = on_px / (width * height)
    eps = np.clip(tuned.eps_rel * (1.0 + 2.0 * density + 0.002 * n_comp), tuned.min_eps_rel, tuned.max_eps_rel)

    m255 = (cleaned * 255).astype(np.uint8)
    contours, _ = cv2.findContours(m255, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    d_list: List[str] = []
    W, H = width, height
    area_img = float(W * H)
    min_bbox_area = tuned.min_path_bbox_frac * area_img

    for cnt in contours:
        if len(cnt) < 3:
            continue
        perim = max(1.0, cv2.arcLength(cnt, True))
        cnt_s = cv2.approxPolyDP(cnt, epsilon=perim * float(eps), closed=True)

        a, p, circ, sol = _poly_metrics(cnt_s)
        if a < tuned.min_path_area_px:
            continue
        if circ < tuned.min_circularity and sol < tuned.min_solidity:
            continue

        x, y, w, h = cv2.boundingRect(cnt_s)
        if (w * h) < min_bbox_area:
            continue

        pts = cnt_s.reshape(-1, 2).astype(float)
        if pts.shape[0] < 3:
            continue

        def r(v): return str(int(round(v)))
        d = [f"M {r(pts[0,0])} {r(pts[0,1])}"]
        d.extend(f"L {r(x)} {r(y)}" for x, y in pts[1:])
        d.append("Z")
        d_list.append(" ".join(d))

    return d_list
