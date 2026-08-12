"""Personal Body Map.

Answers one question before any coaching happens: which landmarks can this
camera reliably see on this person? Anything untrackable is marked NOT_APPLICABLE
rather than treated as an error, which is what lets the same analyzers serve users
with different physical configurations without penalising them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from . import bodygroups, config
from .landmarks import LM, MetricStatus, PoseFrame


class BodyMode(str, Enum):
    STANDARD = "standard"   # full symmetric landmark set available
    ADAPTIVE = "adaptive"   # some landmarks unavailable; metrics reduced accordingly


# Landmarks whose absence should switch the session into adaptive mode.
_CORE_SET: List[LM] = [
    LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER,
    LM.LEFT_ELBOW, LM.RIGHT_ELBOW,
    LM.LEFT_WRIST, LM.RIGHT_WRIST,
    LM.LEFT_HIP, LM.RIGHT_HIP,
    LM.LEFT_KNEE, LM.RIGHT_KNEE,
    LM.LEFT_ANKLE, LM.RIGHT_ANKLE,
]


@dataclass
class BodyMap:
    """The result of calibration. Immutable for the rest of the session."""
    trackable: Set[int] = field(default_factory=set)
    hit_ratio: Dict[int, float] = field(default_factory=dict)
    mode: BodyMode = BodyMode.STANDARD
    frames_seen: int = 0

    def is_trackable(self, lm: LM) -> bool:
        return int(lm) in self.trackable

    def all_trackable(self, *lms: LM) -> bool:
        return all(int(l) in self.trackable for l in lms)

    def status_for(self, *lms: LM) -> MetricStatus:
        """A metric is only applicable if every landmark it needs is trackable."""
        if self.all_trackable(*lms):
            return MetricStatus.MEASURED
        return MetricStatus.NOT_APPLICABLE

    def applicable(self, *lms: LM) -> bool:
        """Reads better than all_trackable at a guard site: does this apply here?"""
        return self.all_trackable(*lms)

    def missing_core(self) -> List[LM]:
        return [lm for lm in _CORE_SET if int(lm) not in self.trackable]

    # -- body groups -------------------------------------------------------
    def available_groups(self) -> List[str]:
        """Named parts of the body this camera can see on this person."""
        return bodygroups.available_groups(self)

    def missing_groups(self) -> List[str]:
        """Named parts we are not tracking. Not faults — just absent."""
        return bodygroups.missing_groups(self)

    def group_status(self) -> List[Tuple[str, bool]]:
        return bodygroups.group_status(self)

    def symmetry_applicable(self) -> bool:
        """True only when both arms and both legs are tracked.

        Everything bilateral — left/right symmetry, hip and shoulder level —
        depends on this. A user with one leg is not lopsided; the comparison
        simply has no second term, so the metric stands down rather than fails.
        """
        return bodygroups.symmetry_available(self)

    def summary(self) -> str:
        if self.mode is BodyMode.STANDARD:
            return "Standard mode — full landmark set available."
        missing = ", ".join(lm.name.replace("_", " ").title() for lm in self.missing_core())
        return f"Adaptive mode — not tracking: {missing}. Related metrics are dropped, not failed."

    def group_summary(self) -> str:
        """One line naming the groups we are not tracking, for the calibration screen."""
        gone = self.missing_groups()
        if not gone:
            return "Tracking your whole body."
        return "Tracking everything except: " + ", ".join(gone).lower() + "."


class BodyMapCalibrator:
    """Accumulates frames during the calibration window and produces a BodyMap."""

    def __init__(self, duration: float = config.CALIBRATION_SECONDS) -> None:
        self.duration = duration
        self._counts: Dict[int, int] = {}
        self._frames = 0
        self._start: Optional[float] = None

    def reset(self) -> None:
        self._counts.clear()
        self._frames = 0
        self._start = None

    @property
    def frames(self) -> int:
        return self._frames

    def progress(self, now: float) -> float:
        if self._start is None:
            return 0.0
        return min(1.0, (now - self._start) / self.duration)

    def is_complete(self, now: float) -> bool:
        return self._start is not None and (now - self._start) >= self.duration

    def observe(self, frame: PoseFrame) -> None:
        if not frame.detected:
            return
        if self._start is None:
            self._start = frame.timestamp
        self._frames += 1
        for idx, lm in frame.landmarks.items():
            if lm.visibility >= config.VISIBILITY_THRESHOLD:
                self._counts[idx] = self._counts.get(idx, 0) + 1

    def build(self) -> BodyMap:
        if self._frames == 0:
            return BodyMap(mode=BodyMode.ADAPTIVE)

        ratios = {idx: count / self._frames for idx, count in self._counts.items()}
        trackable = {idx for idx, r in ratios.items() if r >= config.TRACKABLE_HIT_RATIO}

        missing_core = [lm for lm in _CORE_SET if int(lm) not in trackable]
        mode = BodyMode.STANDARD if not missing_core else BodyMode.ADAPTIVE

        return BodyMap(
            trackable=trackable,
            hit_ratio=ratios,
            mode=mode,
            frames_seen=self._frames,
        )
