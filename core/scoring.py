"""Metrics, scoring, stability and movement quality.

Scoring only ever averages metrics that were actually measured. A joint the
camera could not see, or one the body map says is untrackable, drops out of the
weighting entirely rather than scoring zero — a user is never marked down for
something the system failed to observe.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Iterable, List, Optional, Sequence

import numpy as np

from .landmarks import MetricStatus
from .phases import RepRecord


@dataclass
class Metric:
    """One reading, plus everything needed to score and display it."""
    key: str
    label: str
    value: Optional[float]
    status: MetricStatus
    unit: str = "°"
    lo: Optional[float] = None      # start of the ideal band
    hi: Optional[float] = None      # end of the ideal band
    falloff: float = 30.0           # distance past the band that scores zero
    weight: float = 1.0

    @property
    def measured(self) -> bool:
        return self.status is MetricStatus.MEASURED and self.value is not None

    @property
    def target(self) -> Optional[str]:
        if self.lo is not None and self.hi is not None:
            return f"{self.lo:.0f}–{self.hi:.0f}{self.unit}"
        if self.hi is not None:
            return f"≤{self.hi:.0f}{self.unit}"
        if self.lo is not None:
            return f"≥{self.lo:.0f}{self.unit}"
        return None

    @property
    def display(self) -> str:
        if self.status is MetricStatus.NOT_APPLICABLE:
            return "not applicable"
        if not self.measured:
            return "not visible"
        return f"{self.value:.0f}{self.unit}"

    def score(self) -> Optional[float]:
        """0–100, or None when there is nothing trustworthy to score."""
        if not self.measured:
            return None
        if self.lo is None and self.hi is None:
            return None
        value = float(self.value)

        excess = 0.0
        if self.lo is not None and value < self.lo:
            excess = self.lo - value
        elif self.hi is not None and value > self.hi:
            excess = value - self.hi

        if excess <= 0.0:
            return 100.0
        if self.falloff <= 0:
            return 0.0
        return float(max(0.0, 100.0 * (1.0 - excess / self.falloff)))

    @property
    def in_band(self) -> Optional[bool]:
        s = self.score()
        return None if s is None else s >= 99.9


def aggregate_score(metrics: Sequence[Metric], penalties: float = 0.0) -> Optional[float]:
    """Weighted mean of the scorable metrics, less any active-error penalty."""
    pairs = [(m.score(), m.weight) for m in metrics]
    pairs = [(s, w) for s, w in pairs if s is not None and w > 0]
    if not pairs:
        return None
    total_w = sum(w for _s, w in pairs)
    mean = sum(s * w for s, w in pairs) / total_w
    return float(max(0.0, min(100.0, mean - penalties)))


class StabilityTracker:
    """Body sway over a rolling window, expressed as a 0–100 steadiness score.

    Deliberately forgiving of small natural movement: a human holding a pose is
    never perfectly still, and a system that treats micro-movement as a fault is
    wrong about what balance is.
    """

    def __init__(self, window: int = 45, tolerance: float = 0.10) -> None:
        self.window = window
        self.tolerance = tolerance         # sway (torso-relative) that scores zero
        self._x: Deque[float] = deque(maxlen=window)
        self._y: Deque[float] = deque(maxlen=window)

    def reset(self) -> None:
        self._x.clear()
        self._y.clear()

    def update(self, cx: Optional[float], cy: Optional[float],
               scale: Optional[float]) -> None:
        if cx is None or cy is None or scale is None or scale <= 1e-6:
            return
        self._x.append(cx / scale)
        self._y.append(cy / scale)

    @property
    def ready(self) -> bool:
        return len(self._x) >= 10

    def sway(self) -> Optional[float]:
        """Combined lateral+vertical drift, in torso lengths."""
        if not self.ready:
            return None
        return float(np.hypot(np.std(self._x), np.std(self._y)))

    def score(self) -> Optional[float]:
        s = self.sway()
        if s is None:
            return None
        # A small dead zone means natural micro-movement still scores 100.
        dead = self.tolerance * 0.25
        if s <= dead:
            return 100.0
        span = max(1e-6, self.tolerance - dead)
        return float(max(0.0, min(100.0, 100.0 * (1.0 - (s - dead) / span))))


class MovementQuality:
    """Consistency across completed reps — are they the same rep every time?

    Two things vary in a tiring set: how long each rep takes, and how deep it
    goes. Low variance in both is what separates controlled work from flailing.
    """

    def __init__(self) -> None:
        self.reps: List[RepRecord] = []

    def reset(self) -> None:
        self.reps = []

    def sync(self, reps: Sequence[RepRecord]) -> None:
        self.reps = list(reps)

    @property
    def clean_reps(self) -> int:
        return sum(1 for r in self.reps if r.clean)

    def consistency(self) -> Optional[float]:
        """0–100. None until there are enough reps to compare."""
        if len(self.reps) < 2:
            return None

        durations = np.array([r.duration for r in self.reps], dtype=float)
        peaks = np.array([r.peak for r in self.reps], dtype=float)

        # Coefficient of variation: spread relative to the average.
        def cv(values: np.ndarray) -> float:
            mean = float(np.mean(values))
            if abs(mean) < 1e-6:
                return 1.0
            return float(np.std(values) / abs(mean))

        # 25% variation scores zero; anything tighter scales up from there.
        tempo = max(0.0, 1.0 - cv(durations) / 0.25)
        depth = max(0.0, 1.0 - cv(peaks) / 0.25)
        return float(max(0.0, min(100.0, 100.0 * (0.5 * tempo + 0.5 * depth))))

    def summary(self) -> str:
        if not self.reps:
            return "No completed reps yet."
        c = self.consistency()
        tail = "" if c is None else f" · consistency {c:.0f}/100"
        return f"{self.clean_reps}/{len(self.reps)} clean reps{tail}"


class HoldQuality:
    """How good a static hold was: time in position, weighted by steadiness."""

    def __init__(self) -> None:
        self._samples: List[float] = []

    def reset(self) -> None:
        self._samples = []

    def update(self, in_position: bool, stability: Optional[float]) -> None:
        if not in_position:
            return
        self._samples.append(100.0 if stability is None else stability)

    def quality(self, in_position_ratio: float) -> Optional[float]:
        if not self._samples:
            return None
        steadiness = float(np.mean(self._samples))
        # Being steady matters more than being in position for every second,
        # but a pose you never actually reached should not score well either.
        return float(max(0.0, min(100.0,
                                  0.7 * steadiness + 30.0 * min(1.0, in_position_ratio * 1.4))))
