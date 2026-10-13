"""Sci-fi HUD overlay (cyan skeleton, red face box, neural-net graphic, status panel)."""
from __future__ import annotations

import math
import os
import time
from typing import Optional, Sequence

import cv2
import numpy as np

import square

CYAN, WHITE, RED, DIM = (255, 230, 40), (255, 255, 255), (60, 60, 255), (170, 150, 40)
FONT = cv2.FONT_HERSHEY_SIMPLEX
CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
               (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


class FaceBox:
    """Finds the biggest face with OpenCV's built-in detector (no download, no identification)."""

    def __init__(self, every: int = 3):
        path = os.path.join(getattr(getattr(cv2, "data", None), "haarcascades", ""), "haarcascade_frontalface_default.xml")
        self.det = cv2.CascadeClassifier(path) if os.path.isfile(path) else None
        self.available = self.det is not None and not self.det.empty()
        self.every, self.n = every, 0
        self.box: Optional[np.ndarray] = None
        self.seen = 0.0

    def update(self, frame: np.ndarray, t: float):
        if not self.available:
            return None
        self.n += 1
        if self.n % self.every == 0:
            small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None, fx=0.5, fy=0.5)
            faces = self.det.detectMultiScale(small, 1.15, 5, minSize=(40, 40))
            if len(faces):
                f = np.array(max(faces, key=lambda r: r[2] * r[3]), dtype=np.float32) * 2
                self.box = f if self.box is None else self.box + 0.5 * (f - self.box)
                self.seen = t
        if self.box is not None and t - self.seen > 0.8:
            self.box = None
        return None if self.box is None else tuple(int(v) for v in self.box)


def _text(img, s, org, scale=0.45, col=CYAN, th=1):
    cv2.putText(img, s, org, FONT, scale, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, col, th, cv2.LINE_AA)


def _panel(img, x0, y0, x1, y1, alpha=0.55):
    sub = img[y0:y1, x0:x1]
    img[y0:y1, x0:x1] = cv2.addWeighted(sub, 1 - alpha, np.zeros_like(sub), alpha, 0)
    cv2.rectangle(img, (x0, y0), (x1, y1), DIM, 1)


def _brackets(img, x0, y0, x1, y1, col, n=14, th=2):
    for (x, y, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        cv2.line(img, (x, y), (x + dx * n, y), col, th, cv2.LINE_AA)
        cv2.line(img, (x, y), (x, y + dy * n), col, th, cv2.LINE_AA)


def draw_skeleton(img, hands: Sequence, t: float) -> None:
    h, w = img.shape[:2]
    layer = np.zeros_like(img)
    for hd in hands:
        p = [(int(x * w), int(y * h)) for x, y in hd]
        for a, b in CONNECTIONS:
            cv2.line(layer, p[a], p[b], CYAN, 2, cv2.LINE_AA)
        for i in range(21):
            cv2.circle(layer, p[i], 4 if i in (4, 8, 12, 16, 20) else 3, WHITE, -1, cv2.LINE_AA)
        for i in (4, 8, 12, 16, 20):
            cv2.circle(layer, p[i], 11 + int(2 * math.sin(t * 5 + i)), CYAN, 1, cv2.LINE_AA)
        cx, cy = p[9]
        cv2.line(layer, (cx - 9, cy), (cx + 9, cy), WHITE, 1, cv2.LINE_AA)
        cv2.line(layer, (cx, cy - 9), (cx, cy + 9), WHITE, 1, cv2.LINE_AA)
    glow = cv2.GaussianBlur(layer, (0, 0), 5)
    img[:] = cv2.add(cv2.add(img, layer), glow)


def draw_face(img, box) -> None:
    x, y, bw, bh = box
    _brackets(img, x, y, x + bw, y + bh, RED, max(10, bw // 6), 2)
    cv2.rectangle(img, (x, y), (x + bw, y + bh), RED, 1, cv2.LINE_AA)
    _text(img, "FACE DETECTION ACTIVE", (x - 6, max(12, y - 8)), 0.4, RED)
    cx, cy = x + bw // 2, y + bh // 2
    cv2.line(img, (cx - 7, cy), (cx + 7, cy), WHITE, 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy - 7), (cx, cy + 7), WHITE, 1, cv2.LINE_AA)
    s = max(10, bw // 7)
    for sx, sgn in ((x - 2 * s - 8, 1), (x + bw + 8, 1)):  # small readout boxes beside the face, like the reference
        for k in range(2):
            yy = y + bh // 4 + k * (s + 8)
            cv2.rectangle(img, (sx, yy), (sx + 2 * s, yy + s), CYAN, 1, cv2.LINE_AA)
            cv2.circle(img, (sx + s, yy + s // 2), max(2, s // 4), CYAN, 1, cv2.LINE_AA)


def draw_network(img, t: float) -> None:
    h, w = img.shape[:2]
    layers = [3, 5, 5, 2]
    x0, x1, yc, span = int(w * 0.03), int(w * 0.2), int(h * 0.55), int(h * 0.3)
    nodes = []
    for li, n in enumerate(layers):
        x = int(x0 + (x1 - x0) * li / (len(layers) - 1))
        nodes.append([(x, int(yc - span / 2 + span * (i + 0.5) / n)) for i in range(n)])
    layer = np.zeros_like(img)
    for a, b in zip(nodes, nodes[1:]):
        for p in a:
            for q in b:
                cv2.line(layer, p, q, DIM, 1, cv2.LINE_AA)
    for li, col in enumerate(nodes):
        for i, p in enumerate(col):
            pulse = 0.5 + 0.5 * math.sin(t * 3 + li * 1.3 + i)
            cv2.circle(layer, p, 4, (int(255 * pulse), 255, int(40 + 150 * pulse)), -1, cv2.LINE_AA)
    img[:] = cv2.add(img, layer)


def draw_square_label(img, poly, name: str, t: float) -> None:
    if poly is None:
        return
    pts = poly.astype(int)
    cv2.polylines(img, [pts], True, CYAN, 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, tuple(int(v) for v in p), 6, WHITE, 2, cv2.LINE_AA)
        cv2.circle(img, tuple(int(v) for v in p), 12, CYAN, 1, cv2.LINE_AA)
    top = pts[np.argmin(pts[:, 1])]
    _text(img, f"FILTER: {name.upper()}", (int(top[0]) + 10, max(14, int(top[1]) - 10)), 0.5, WHITE, 1)


def draw_hud(img, hands: Sequence, face, sf: "square.SquareFilter", fps: float, t: float, show_face: bool = True) -> None:
    h, w = img.shape[:2]
    draw_network(img, t)
    draw_skeleton(img, hands, t)
    if face is not None and show_face:
        draw_face(img, face)
    draw_square_label(img, sf.poly, sf.switcher.name, t)

    x1, x0 = w - 12, w - 250
    _panel(img, x0, 10, x1, 130)
    _text(img, "SYSTEM STATUS", (x0 + 10, 30), 0.5, CYAN, 1)
    rows = [f"HAND TRACKING: {'ACTIVE' if hands else 'SEARCHING'}", f"HANDS: {len(hands)}", f"FILTER: {sf.switcher.name.upper()}",
            f"SQUARE: {'LOCKED' if sf.poly is not None else 'NONE'}", f"FPS: {fps:4.1f}"]
    for i, r in enumerate(rows):
        _text(img, r, (x0 + 10, 52 + 16 * i), 0.4, CYAN)

    g = square.gesture_name(hands[0]) if hands else "NONE"
    _text(img, f"GESTURE: {g}", (14, h - 74), 0.45, CYAN)
    _text(img, f"BIOMETRIC: {'FACE DETECTED' if face is not None else 'NO FACE'}", (14, h - 56), 0.4, DIM)
    help_ = ["Square: both hands (thumb + index out) or one hand as an L",
             "Touch thumb to little finger = next filter", "N next   H hide HUD   S screenshot   Q quit"]
    for i, r in enumerate(help_):
        _text(img, r, (14, h - 36 + 14 * i), 0.38, (200, 200, 200))
