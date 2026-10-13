"""Signal filters that steady the hand tracking. Pure Python, no camera needed."""
from __future__ import annotations

import math
from typing import Optional, Sequence


class OneEuro:
    """Steady when the hand rests, quick when it moves (Casiez et al. 2012). Lower min_cutoff = smoother."""

    def __init__(self, min_cutoff: float = 1.2, beta: float = 12.0, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x: Optional[float] = None
        self.dx = 0.0
        self.t: Optional[float] = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        return 1.0 / (1.0 + (1.0 / (2 * math.pi * cutoff)) / dt)

    def reset(self) -> None:
        self.x = self.t = None
        self.dx = 0.0

    def __call__(self, x: float, t: float) -> float:
        if self.x is None or self.t is None:
            self.x, self.t = x, t
            return x
        dt = max(t - self.t, 1e-3)
        self.dx += self._alpha(self.d_cutoff, dt) * ((x - self.x) / dt - self.dx)
        self.x += self._alpha(self.min_cutoff + self.beta * abs(self.dx), dt) * (x - self.x)
        self.t = t
        return self.x


class EMA:
    """Plain exponential average. alpha near 0 = very smooth and laggy, near 1 = raw."""

    def __init__(self, alpha: float = 0.35):
        self.alpha, self.x = alpha, None

    def reset(self) -> None:
        self.x = None

    def __call__(self, x: float, t: float = 0.0) -> float:
        self.x = x if self.x is None else self.x + self.alpha * (x - self.x)
        return self.x


class Kalman1D:
    """Constant-velocity Kalman filter. Predicts where the finger is going, so it lags less than an average."""

    def __init__(self, process: float = 40.0, measure: float = 0.0009):
        self.q, self.r = process, measure
        self.reset()

    def reset(self) -> None:
        self.x: Optional[float] = None
        self.v = 0.0
        self.p = [[1.0, 0.0], [0.0, 1.0]]
        self.t: Optional[float] = None

    def __call__(self, z: float, t: float) -> float:
        if self.x is None or self.t is None:
            self.x, self.t = z, t
            return z
        dt = max(t - self.t, 1e-3)
        self.t = t
        # predict
        x, v, P = self.x + self.v * dt, self.v, self.p
        p00 = P[0][0] + dt * (P[1][0] + P[0][1]) + dt * dt * P[1][1] + self.q * dt ** 3 / 3
        p01 = P[0][1] + dt * P[1][1] + self.q * dt ** 2 / 2
        p10 = P[1][0] + dt * P[1][1] + self.q * dt ** 2 / 2
        p11 = P[1][1] + self.q * dt
        # update
        s = p00 + self.r
        k0, k1 = p00 / s, p10 / s
        y = z - x
        self.x, self.v = x + k0 * y, v + k1 * y
        self.p = [[(1 - k0) * p00, (1 - k0) * p01], [p10 - k1 * p00, p11 - k1 * p01]]
        return self.x


class Passthrough:
    def reset(self) -> None:  # noqa: D102
        pass

    def __call__(self, x: float, t: float = 0.0) -> float:
        return x


STRENGTH = {"low": 2.5, "medium": 1.2, "high": 0.6}  # One Euro min_cutoff per preset


def make_filter(kind: str = "oneeuro", strength: str = "medium"):
    if kind == "none":
        return Passthrough()
    if kind == "ema":
        return EMA({"low": 0.6, "medium": 0.35, "high": 0.18}[strength])
    if kind == "kalman":
        return Kalman1D(process={"low": 120.0, "medium": 40.0, "high": 12.0}[strength])
    return OneEuro(min_cutoff=STRENGTH[strength])


class LandmarkSmoother:
    """Filters all 21 hand points so finger states do not flicker between frames."""

    def __init__(self, enabled: bool = True, strength: str = "medium"):
        self.enabled = enabled
        self._f = [(OneEuro(STRENGTH[strength] * 1.5, 25.0), OneEuro(STRENGTH[strength] * 1.5, 25.0)) for _ in range(21)]

    def reset(self) -> None:
        for fx, fy in self._f:
            fx.reset()
            fy.reset()

    def __call__(self, lm: Sequence[Sequence[float]], t: float):
        if not self.enabled:
            return list(lm)
        return [(fx(p[0], t), fy(p[1], t)) for p, (fx, fy) in zip(lm, self._f)]


class PoseDebounce:
    """A pose only counts after it has been seen for n frames in a row (kills one-frame glitches)."""

    def __init__(self, n: int = 2, initial: str = "none"):
        self.n, self.stable, self._cand, self._count = max(1, n), initial, initial, 0

    def __call__(self, pose: str) -> str:
        if pose == self._cand:
            self._count += 1
        else:
            self._cand, self._count = pose, 1
        if self._count >= self.n:
            self.stable = self._cand
        return self.stable
