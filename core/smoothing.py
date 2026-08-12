"""One Euro filter plus stale-landmark tolerance.

Raw MediaPipe output jitters enough that a naive angle threshold flickers between
"correct" and "wrong" several times a second. Smoothing is what makes the coaching
layer usable, so it lives in the engine rather than in any one exercise.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

from . import config
from .geometry import torso_scale
from .landmarks import Landmark, PoseFrame

# --- outlier rejection -------------------------------------------------------
# MediaPipe occasionally "teleports" a single landmark for one frame. A jump of
# more than this many torso-lengths between consecutive tracked frames is
# physically implausible, so we treat it as a spike and coast instead — unless
# the landmark stays in the new place next frame (real fast motion is sustained;
# a detector glitch is not).
OUTLIER_JUMP = 0.9


def _world_dist(a: Landmark, b: Landmark) -> float:
    """Distance in metric world space — the same space torso_scale measures in."""
    return math.sqrt((a.wx - b.wx) ** 2 + (a.wy - b.wy) ** 2 + (a.wz - b.wz) ** 2)


class _LowPass:
    def __init__(self) -> None:
        self.value: Optional[float] = None

    def apply(self, x: float, alpha: float) -> float:
        if self.value is None:
            self.value = x
        else:
            self.value = alpha * x + (1.0 - alpha) * self.value
        return self.value


class OneEuro:
    """Adaptive low-pass: heavy smoothing when still, light smoothing when moving fast."""

    def __init__(
        self,
        min_cutoff: float = config.ONE_EURO_MIN_CUTOFF,
        beta: float = config.ONE_EURO_BETA,
        d_cutoff: float = config.ONE_EURO_D_CUTOFF,
    ) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x = _LowPass()
        self._dx = _LowPass()
        self._last_t: Optional[float] = None
        self._last_x: Optional[float] = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, t: float) -> float:
        if self._last_t is None or t <= self._last_t:
            dt = 1.0 / 30.0
        else:
            dt = t - self._last_t
        self._last_t = t

        dx = 0.0 if self._last_x is None else (x - self._last_x) / dt
        self._last_x = x

        edx = self._dx.apply(dx, self._alpha(self.d_cutoff, dt))
        cutoff = self.min_cutoff + self.beta * abs(edx)
        return self._x.apply(x, self._alpha(cutoff, dt))


class _LandmarkSmoother:
    """One filter per coordinate of one landmark."""

    def __init__(self) -> None:
        self.filters: Dict[str, OneEuro] = {k: OneEuro() for k in ("x", "y", "z", "wx", "wy", "wz")}
        self.last: Optional[Landmark] = None
        self.stale_frames = 0
        # Outlier state: the last raw sample we *accepted* (jump reference), the
        # raw sample we most recently *rejected* (so sustained motion into the
        # same new region is accepted on its second frame), and how many frames
        # in a row we have rejected (to tell our own coasting from real dropout).
        self.last_raw: Optional[Landmark] = None
        self.pending: Optional[Landmark] = None
        self.rejected_streak = 0

    def update(self, lm: Landmark, t: float) -> Landmark:
        smoothed = Landmark(
            idx=lm.idx,
            x=self.filters["x"](lm.x, t),
            y=self.filters["y"](lm.y, t),
            z=self.filters["z"](lm.z, t),
            visibility=lm.visibility,
            wx=self.filters["wx"](lm.wx, t),
            wy=self.filters["wy"](lm.wy, t),
            wz=self.filters["wz"](lm.wz, t),
        )
        self.last = smoothed
        self.stale_frames = 0
        self.last_raw = lm
        self.pending = None
        self.rejected_streak = 0
        return smoothed

    def is_spike(self, lm: Landmark, scale: Optional[float]) -> bool:
        """True when a new sample looks like a one-frame teleport, not real motion.

        Rejection never adds lag: samples inside the jump limit go straight to
        the filters, and anything we cannot judge (no torso scale, no history,
        already coasting through a genuine dropout) is accepted as-is.
        """
        if scale is None or self.last_raw is None:
            return False                    # cannot judge the jump -> accept
        if self.stale_frames > self.rejected_streak:
            # Staleness beyond our own rejections means a genuine dropout: the
            # last position is old, so a large jump is entirely plausible.
            return False
        limit = OUTLIER_JUMP * scale
        if _world_dist(lm, self.last_raw) <= limit:
            return False                    # ordinary motion
        if self.pending is not None and _world_dist(lm, self.pending) <= limit:
            # Same far-away region as the frame we just rejected: the person
            # genuinely moved fast. Never fight sustained motion.
            return False
        return True

    def reject(self, lm: Landmark) -> Optional[Landmark]:
        """Drop a spike: coast on the last good value, remember the newcomer.

        Returns None when we have coasted past MAX_STALE_FRAMES — the caller
        then accepts the sample rather than losing a landmark that is detected.
        """
        self.pending = lm
        self.rejected_streak += 1
        return self.coast()

    def coast(self) -> Optional[Landmark]:
        """Re-use the last known position for a short while, with decaying confidence.

        This is what stops a single dropped frame from wiping a joint out of the
        analysis and making the coach stutter.
        """
        if self.last is None:
            return None
        self.stale_frames += 1
        if self.stale_frames > config.MAX_STALE_FRAMES:
            return None
        decay = 1.0 - (self.stale_frames / (config.MAX_STALE_FRAMES + 1))
        return Landmark(
            idx=self.last.idx,
            x=self.last.x, y=self.last.y, z=self.last.z,
            visibility=self.last.visibility * decay,
            wx=self.last.wx, wy=self.last.wy, wz=self.last.wz,
        )


class PoseSmoother:
    """Smooths a whole PoseFrame and bridges short landmark dropouts."""

    def __init__(self) -> None:
        self._per_landmark: Dict[int, _LandmarkSmoother] = {}

    def reset(self) -> None:
        self._per_landmark.clear()

    def apply(self, frame: PoseFrame) -> PoseFrame:
        out: Dict[int, Landmark] = {}
        seen = set(frame.landmarks)

        # Torso scale of the incoming raw frame gives the jump limit in
        # body-relative units. When the torso is not measurable we cannot judge
        # jumps, so rejection is disabled and every sample is accepted.
        scale = torso_scale(frame)

        for idx, lm in frame.landmarks.items():
            smoother = self._per_landmark.setdefault(idx, _LandmarkSmoother())
            if smoother.is_spike(lm, scale):
                coasted = smoother.reject(lm)
                if coasted is not None:
                    out[idx] = coasted      # hold position, confidence decaying
                    continue
                # Coasting expired: believe the detector rather than losing a
                # landmark that is actually present in the frame.
            out[idx] = smoother.update(lm, frame.timestamp)

        # Landmarks that vanished this frame get to coast briefly.
        for idx, smoother in self._per_landmark.items():
            if idx in seen:
                continue
            coasted = smoother.coast()
            if coasted is not None:
                out[idx] = coasted

        return PoseFrame(
            landmarks=out,
            frame_index=frame.frame_index,
            timestamp=frame.timestamp,
            detected=frame.detected,
        )
