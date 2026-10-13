import math

KNUCKLE_X = {"index": -0.045, "middle": 0.0, "ring": 0.045, "pinky": 0.085}
IDX = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}


def make_hand(extended=("index", "middle", "ring", "pinky"), cx=0.5, cy=0.7, scale=1.0, thumb="out", rot=0.0, mirror=False):
    """Synthetic upright hand (21 points, y grows downward). thumb: 'out' | 'in' | 'pinky' (tip touches the little-finger tip)."""
    pts = [(0.0, 0.0)] * 21
    pts = list(pts)
    for name, base in IDX.items():
        mx, my = KNUCKLE_X[name], -0.20
        if name in extended:
            pts[base], pts[base + 1], pts[base + 2], pts[base + 3] = (mx, my), (mx, my - 0.06), (mx, my - 0.10), (mx, my - 0.14)
        else:
            pts[base], pts[base + 1], pts[base + 2], pts[base + 3] = (mx, my), (mx, my - 0.04), (mx, my - 0.01), (mx, my + 0.03)
    pts[1], pts[2] = (-0.05, -0.04), (-0.08, -0.08)
    if thumb == "out":
        pts[3], pts[4] = (-0.11, -0.11), (-0.16, -0.14)
    elif thumb == "in":
        pts[3], pts[4] = (-0.05, -0.12), (-0.02, -0.15)
    else:
        px, py = pts[20]
        pts[3], pts[4] = (px - 0.03, py + 0.05), (px - 0.006, py + 0.004)
    c, s = math.cos(rot), math.sin(rot)
    return [(cx + (-1 if mirror else 1) * (x * c - y * s) * scale, cy + (x * s + y * c) * scale) for x, y in pts]
