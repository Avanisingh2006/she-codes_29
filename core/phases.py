"""Movement phase detection for repetition-based exercises.

A rep is not "angle went below a threshold". It is a directed traversal:
top -> descending -> bottom -> ascending -> top. Detecting direction as well as
level is what lets an analyzer say *when* in the movement a fault happened,
which is the difference between "your knees cave" and "your knees cave at the
bottom of the descent".
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple


@dataclass
class PhaseNames:
    """Display names for the four stages, so each exercise reads naturally."""
    top: str
    descending: str
    bottom: str
    ascending: str

    def all(self) -> List[str]:
        return [self.top, self.descending, self.bottom, self.ascending]


@dataclass
class RepRecord:
    """One completed repetition."""
    index: int
    duration: float
    peak: float             # deepest value the driver angle reached
    error_keys: tuple = ()

    @property
    def clean(self) -> bool:
        return not self.error_keys


class RepPhaseMachine:
    """Tracks one driver angle through the rep cycle.

    The driver is an angle that *decreases* toward the bottom of the movement:
    knee angle for a squat, elbow angle for a curl. Direction comes from a
    smoothed derivative, so a couple of noisy frames cannot fake a phase change.
    """

    def __init__(
        self,
        top: float,
        bottom: float,
        names: PhaseNames,
        velocity_window: int = 5,
        min_rep_seconds: float = 0.4,
    ) -> None:
        self.top = top
        self.bottom = bottom
        self.names = names
        self.min_rep_seconds = min_rep_seconds

        self._history: Deque[Tuple[float, float]] = deque(maxlen=velocity_window)
        self.phase: str = names.top
        self.reps: List[RepRecord] = []
        self._reached_bottom = False
        self._rep_start: Optional[float] = None
        self._peak = 180.0
        self._rep_errors: set = set()

    # -- state -------------------------------------------------------------
    def reset(self) -> None:
        self._history.clear()
        self.phase = self.names.top
        self.reps = []
        self._reached_bottom = False
        self._rep_start = None
        self._peak = 180.0
        self._rep_errors = set()

    @property
    def rep_count(self) -> int:
        return len(self.reps)

    @property
    def last_rep(self) -> Optional[RepRecord]:
        return self.reps[-1] if self.reps else None

    @property
    def depth_progress(self) -> float:
        """0 at the top of the movement, 1 at the bottom. Drives the ghost overlay."""
        if not self._history:
            return 0.0
        value = self._history[-1][1]
        span = self.top - self.bottom
        if span <= 0:
            return 0.0
        return max(0.0, min(1.0, (self.top - value) / span))

    def note_error(self, key: str) -> None:
        """Attribute an active error to the rep in progress."""
        self._rep_errors.add(key)

    # -- update ------------------------------------------------------------
    def _velocity(self) -> float:
        """Smoothed rate of change of the driver, degrees per second."""
        if len(self._history) < 2:
            return 0.0
        (t0, v0), (t1, v1) = self._history[0], self._history[-1]
        dt = t1 - t0
        return 0.0 if dt <= 1e-6 else (v1 - v0) / dt

    def update(self, driver: Optional[float], now: float) -> str:
        """Feed one frame. Returns the current phase name."""
        if driver is None:
            return self.phase

        self._history.append((now, driver))
        velocity = self._velocity()

        if driver >= self.top:
            # Back at the top. If we got all the way down, that was a rep.
            if self._reached_bottom and self._rep_start is not None:
                duration = now - self._rep_start
                if duration >= self.min_rep_seconds:
                    self.reps.append(RepRecord(
                        index=len(self.reps) + 1,
                        duration=duration,
                        peak=self._peak,
                        error_keys=tuple(sorted(self._rep_errors)),
                    ))
                self._reached_bottom = False
                self._rep_start = None
                self._peak = 180.0
                self._rep_errors = set()
            self.phase = self.names.top

        elif driver <= self.bottom:
            if not self._reached_bottom:
                self._reached_bottom = True
            if self._rep_start is None:
                self._rep_start = now
            self._peak = min(self._peak, driver)
            self.phase = self.names.bottom

        else:
            # Mid-range: direction decides which half of the rep we are in.
            if self._rep_start is None and velocity < -8.0:
                self._rep_start = now          # descent has begun
            if velocity < -8.0:
                self.phase = self.names.descending
            elif velocity > 8.0:
                self.phase = self.names.ascending
            # else: hold the previous phase rather than flickering

        return self.phase


class HoldStateMachine:
    """Static-pose equivalent: are we in position, and for how long.

    Requires the pose to be held for a short qualifying period before the timer
    starts, and tolerates a brief wobble out of position before resetting — so a
    single noisy frame does not zero someone's hold.
    """

    def __init__(self, setup_name: str = "setting up", hold_name: str = "holding",
                 enter_frames: int = 4, exit_frames: int = 8) -> None:
        self.setup_name = setup_name
        self.hold_name = hold_name
        self.enter_frames = enter_frames
        self.exit_frames = exit_frames
        self.reset()

    def reset(self) -> None:
        self.phase = self.setup_name
        self._in_streak = 0
        self._out_streak = 0
        self._start: Optional[float] = None
        self.best_hold = 0.0
        self.total_in_position = 0.0
        self.total_time = 0.0
        self._last_t: Optional[float] = None

    def update(self, in_position: bool, now: float) -> str:
        if self._last_t is not None:
            dt = max(0.0, now - self._last_t)
            self.total_time += dt
            if self.phase == self.hold_name:
                self.total_in_position += dt
        self._last_t = now

        if in_position:
            self._in_streak += 1
            self._out_streak = 0
            if self._in_streak >= self.enter_frames and self.phase != self.hold_name:
                self.phase = self.hold_name
                self._start = now
        else:
            self._out_streak += 1
            self._in_streak = 0
            if self._out_streak >= self.exit_frames and self.phase == self.hold_name:
                self.best_hold = max(self.best_hold, self.duration(now))
                self.phase = self.setup_name
                self._start = None

        return self.phase

    def duration(self, now: float) -> float:
        if self._start is None:
            return 0.0
        return max(0.0, now - self._start)

    @property
    def in_position_ratio(self) -> float:
        """Fraction of the session actually spent in the pose."""
        if self.total_time <= 0.1:
            return 0.0
        return min(1.0, self.total_in_position / self.total_time)
