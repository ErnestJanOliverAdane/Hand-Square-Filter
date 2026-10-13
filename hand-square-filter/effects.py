"""The filters that appear inside the square. Each takes and returns a BGR picture."""
from __future__ import annotations

import cv2
import numpy as np

NAMES = ["sketch", "neon", "thermal", "night vision", "x-ray", "cartoon", "glitch", "pixel", "mono", "invert", "blur"]


def _gray3(img):
    return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


def apply(img: np.ndarray, name: str) -> np.ndarray:
    if name == "mono":
        return _gray3(img)
    if name == "invert":
        return 255 - img
    if name == "blur":
        return cv2.GaussianBlur(img, (0, 0), 14)
    if name == "pixel":
        h, w = img.shape[:2]
        return cv2.resize(cv2.resize(img, (max(1, w // 12), max(1, h // 12))), (w, h), interpolation=cv2.INTER_NEAREST)
    if name == "sketch":
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        inv = 255 - cv2.GaussianBlur(g, (21, 21), 0)
        return cv2.cvtColor(cv2.divide(g, 255 - inv + 1, scale=256), cv2.COLOR_GRAY2BGR)
    if name == "neon":
        e = cv2.dilate(cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3, 3), 0), 60, 140), np.ones((2, 2), np.uint8))
        out = np.zeros_like(img)
        out[..., 1], out[..., 0] = e, (e * 0.4).astype(np.uint8)
        return cv2.addWeighted(out, 1.0, cv2.GaussianBlur(out, (0, 0), 6), 1.2, 0)
    if name == "thermal":
        return cv2.applyColorMap(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_INFERNO)
    if name == "night vision":
        g = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        out = np.zeros_like(img)
        out[..., 1] = np.clip(g * 1.1, 0, 255).astype(np.uint8)
        out[..., 0] = (g * 0.15).astype(np.uint8)
        grain = np.random.default_rng().integers(0, 25, g.shape, dtype=np.uint8)
        out[..., 1] = cv2.add(out[..., 1], grain)
        out[::3] = (out[::3] * 0.8).astype(np.uint8)  # scan lines
        return out
    if name == "x-ray":
        g = 255 - cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = cv2.createCLAHE(3.0, (8, 8)).apply(g)
        out = cv2.applyColorMap(g, cv2.COLORMAP_BONE)
        out[..., 0] = np.clip(out[..., 0].astype(int) + 40, 0, 255).astype(np.uint8)
        return out
    if name == "cartoon":
        edges = cv2.adaptiveThreshold(cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 7), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 7)
        smooth = cv2.bilateralFilter(img, 7, 60, 60)
        q = (smooth // 40) * 40 + 20
        return cv2.bitwise_and(q.astype(np.uint8), q.astype(np.uint8), mask=edges)
    if name == "glitch":
        out = img.copy()
        s = 8
        out[..., 2] = np.roll(img[..., 2], s, axis=1)
        out[..., 0] = np.roll(img[..., 0], -s, axis=1)
        h = img.shape[0]
        for y in np.random.default_rng().integers(0, max(1, h - 6), 4):
            out[y:y + 4] = np.roll(out[y:y + 4], int(np.random.default_rng().integers(-30, 30)), axis=1)
        return out
    return img
