"""Form errors, their priority, and the debounce that stops noise becoming nagging.

Pose estimation jitters. A raw per-frame rule will flip in and out of "fault"
several times a second, which produces a coach that stutters and is instantly
distrusted. Every error therefore has to survive a hysteresis gate before it is
allowed to reach the user, and has to stay quiet for a while before it clears.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence

from .landmarks import LM


class Severity(str, Enum):
    MINOR = "minor"
    MAJOR = "major"

    @property
    def weight(self) -> float:
        return 1.0 if self is Severity.MINOR else 2.0


class Priority(int, Enum):
    """Lower number wins. Only one error is ever spoken, so the order matters.

    Ranked by what actually protects the user and what unblocks the rest of the
    movement — joint safety first, then the alignment that everything else hangs
    off, then range, then cosmetics.
    """
    SAFETY = 0        # a joint being loaded in a position we don't want to hold
    ALIGNMENT = 1     # the structural fault other faults hang off
    STABILITY = 2     # balance and control
    RANGE = 3         # depth, range of motion
    REFINEMENT = 4    # tidy-up cues


@dataclass
class FormError:
    """One detected fault."""
    key: str
    message: str                       # full sentence for the panel
    cue: str                           # short overlay text, e.g. "RIGHT KNEE -> OUT"
    severity: Severity = Severity.MINOR
    priority: Priority = Priority.REFINEMENT
    confidence: float = 1.0
    landmarks: Sequence[LM] = field(default_factory=tuple)
    phase: Optional[str] = None        # phase this was detected in
    # --- later-phase hooks, unused for now -------------------------------
    caused_by: Optional[str] = None
    root_of: Sequence[str] = field(default_factory=tuple)

    @property
    def rank(self) -> tuple:
        """Sort key: priority first, then how bad and how sure we are."""
        return (int(self.priority), -(self.severity.weight * self.confidence))

    @property
    def penalty(self) -> float:
        """Points knocked off the score while this is active."""
        base = 6.0 if self.severity is Severity.MINOR else 14.0
        return base * self.confidence


@dataclass
class _Track:
    hits: int = 0
    misses: int = 0
    active: bool = False
    latest: Optional[FormError] = None
    first_seen: Optional[float] = None


class ErrorTracker:
    """Hysteresis gate over per-frame error candidates.

    An error must be seen on `on_frames` consecutive frames before it is
    reported, and must be absent for `off_frames` before it is withdrawn. The
    asymmetry is deliberate: slow to complain, slower to forget.
    """

    def __init__(self, on_frames: int = 5, off_frames: int = 10) -> None:
        self.on_frames = on_frames
        self.off_frames = off_frames
        self._tracks: Dict[str, _Track] = {}

    def reset(self) -> None:
        self._tracks.clear()

    def update(self, candidates: Sequence[FormError], now: float) -> List[FormError]:
        """Feed this frame's raw detections; get back the ones worth showing."""
        seen = {e.key: e for e in candidates}

        for key, error in seen.items():
            track = self._tracks.setdefault(key, _Track())
            track.hits += 1
            track.misses = 0
            track.latest = error
            if track.first_seen is None:
                track.first_seen = now
            if track.hits >= self.on_frames:
                track.active = True

        for key, track in self._tracks.items():
            if key in seen:
                continue
            track.misses += 1
            if track.misses >= self.off_frames:
                track.active = False
                track.hits = 0
                track.first_seen = None

        active = [t.latest for t in self._tracks.values() if t.active and t.latest]
        return sorted(active, key=lambda e: e.rank)

    def primary(self, active: Sequence[FormError]) -> Optional[FormError]:
        """The single error the coach should surface. Highest priority wins."""
        return active[0] if active else None

    def active_keys(self) -> List[str]:
        return [k for k, t in self._tracks.items() if t.active]
