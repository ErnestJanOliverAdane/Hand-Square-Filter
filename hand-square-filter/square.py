"""The square filter and the thumb + little-finger switch. Pure Python/numpy: no camera needed.

Square:   two hands (thumb + index finger out on each) -> the tidy rectangle through both thumb tips and both index tips
          one hand making an "L" (thumb + index out, other fingers folded) -> a square whose diagonal runs thumb tip to index tip
Switch:   touch your thumb to your LITTLE finger (either hand) -> next filter.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

import effects

WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_MCP, PINKY_TIP = 0, 4, 8, 9, 20
FINGER_JOINTS = {"index": (6, 8), "middle": (10, 12), "ring": (14, 16), "pinky": (18, 20)}


def dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def palm_size(h) -> float:
    return max(dist(h[WRIST], h[MIDDLE_MCP]), 1e-6)


def thumb_pinky_ratio(h) -> float:
    """Gap between thumb tip and little-finger tip as a share of palm size (0 = touching)."""
    return dist(h[THUMB_TIP], h[PINKY_TIP]) / palm_size(h)


def thumb_index_ratio(h) -> float:
    return dist(h[THUMB_TIP], h[INDEX_TIP]) / palm_size(h)


def fingers_up(h) -> dict:
    """True for each finger that is straight (tip farther from the wrist than its middle joint)."""
    return {n: dist(h[t], h[WRIST]) > dist(h[p], h[WRIST]) * 1.08 for n, (p, t) in FINGER_JOINTS.items()}


def thumb_extended(h) -> bool:
    return dist(h[THUMB_TIP], h[17]) > dist(h[3], h[17]) * 1.05 and dist(h[THUMB_TIP], h[5]) > 0.55 * palm_size(h)


def is_l_shape(h) -> bool:
    """Thumb and index finger out, other three fingers folded (an 'L')."""
    up = fingers_up(h)
    return up["index"] and thumb_extended(h) and not (up["middle"] or up["ring"] or up["pinky"])


def is_corner_hand(h) -> bool:
    """Thumb and index finger both out (other fingers may be anything)."""
    return fingers_up(h)["index"] and thumb_extended(h)


def gesture_name(h) -> str:
    """Short label for the HUD."""
    up = fingers_up(h)
    n = sum(up.values())
    if thumb_pinky_ratio(h) < 0.4:
        return "THUMB_PINKY"
    if n == 0:
        return "FIST"
    if thumb_index_ratio(h) < 0.3:
        return "PINCH"
    if n == 4:
        return "OPEN_PALM"
    if up["index"] and not (up["middle"] or up["ring"] or up["pinky"]):
        return "POINT"
    if up["index"] and up["middle"] and not (up["ring"] or up["pinky"]):
        return "PEACE"
    return "HAND"


class FilterSwitcher:
    """Touch thumb to little finger -> next filter. Fires once per touch (needs a clear gap before it can fire again)."""

    def __init__(self, names: Sequence[str] = effects.NAMES, close: float = 0.35, reopen: float = 0.55, cooldown: float = 0.7):
        self.names = list(names)
        self.index = 0
        self.close, self.reopen, self.cooldown = close, reopen, cooldown
        self._touching = [False, False]
        self._ready_at = 0.0
        self.flash_until = 0.0

    @property
    def name(self) -> str:
        return self.names[self.index]

    @property
    def touching(self) -> bool:
        return any(self._touching)

    def next(self, t: float) -> None:
        self.index = (self.index + 1) % len(self.names)
        self.flash_until = t + 0.35

    def update(self, hands: Sequence, t: float) -> bool:
        """Returns True on the frame the filter changed."""
        changed = False
        for i in range(2):
            if i >= len(hands):
                self._touching[i] = False
                continue
            r = thumb_pinky_ratio(hands[i])
            if not self._touching[i] and r < self.close:
                self._touching[i] = True
                if t >= self._ready_at:
                    self.next(t)
                    self._ready_at = t + self.cooldown
                    changed = True
            elif self._touching[i] and r > self.reopen:
                self._touching[i] = False
        return changed


def square_polygon(hands: Sequence, w: int, h: int, min_area: float = 0.01):
    """Corners (4x2 int array) of the square, or None. Uses thumb tip + index tip of each hand."""
    pt = lambda hd, i: (hd[i][0] * w, hd[i][1] * h)  # noqa: E731
    if len(hands) >= 2 and all(is_corner_hand(hd) for hd in hands[:2]):
        pts = np.array([pt(hd, i) for hd in hands[:2] for i in (THUMB_TIP, INDEX_TIP)], dtype=np.float32)
        box = cv2.boxPoints(cv2.minAreaRect(pts))
        poly = np.round(box).astype(np.int32)
        return poly if cv2.contourArea(poly) >= min_area * w * h else None
    if len(hands) == 1 and is_l_shape(hands[0]):
        a, b = np.array(pt(hands[0], THUMB_TIP)), np.array(pt(hands[0], INDEX_TIP))
        d = float(np.linalg.norm(b - a))
        if d < 1e-3 or thumb_index_ratio(hands[0]) < 0.45:
            return None
        m, v = (a + b) / 2, (b - a) / d
        perp = np.array([-v[1], v[0]])
        poly = np.round(np.array([a, m + perp * d / 2, b, m - perp * d / 2])).astype(np.int32)
        return poly if cv2.contourArea(poly) >= min_area * w * h else None
    return None


class SquareFilter:
    """Applies the current filter inside the square; keeps the last square while the thumb touches the little finger."""

    def __init__(self, switcher: Optional[FilterSwitcher] = None):
        self.switcher = switcher or FilterSwitcher()
        self.last_poly = None
        self.poly = None

    def render(self, frame: np.ndarray, hands: Sequence, t: float) -> np.ndarray:
        h, w = frame.shape[:2]
        self.switcher.update(hands, t)
        poly = square_polygon(hands, w, h)
        if self.switcher.touching and self.last_poly is not None:
            poly = self.last_poly  # thumb moved to the little finger: hold the square still
        elif poly is not None:
            self.last_poly = poly
        self.poly = poly
        if poly is None:
            return frame
        inner = effects.apply(frame, self.switcher.name)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(mask, poly, 255)
        out = frame.copy()
        out[mask > 0] = inner[mask > 0]
        if t < self.switcher.flash_until:  # white flash when the filter changes
            k = (self.switcher.flash_until - t) / 0.35
            out[mask > 0] = cv2.addWeighted(out, 1 - 0.6 * k, np.full_like(out, 255), 0.6 * k, 0)[mask > 0]
        return out
