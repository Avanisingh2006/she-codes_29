"""The contract every exercise profile implements.

An exercise declares what it needs and how to read it. It never touches
MediaPipe, smoothing, or the camera — it only ever sees an EngineFrame, and it
always returns the same structured result regardless of which exercise it is.
That uniformity is what keeps the UI from being hard-coded around any one
movement.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence

from core import config
from core.bodygroups import trackable_subset
from core.engine import EngineFrame
from core.errors import ErrorTracker, FormError, Priority, Severity
from core.landmarks import LM, MetricStatus
from core.motion import MotionAnalyzer
from core.scoring import (HoldQuality, Metric, MovementQuality, StabilityTracker,
                          aggregate_score)


class Category(str, Enum):
    YOGA = "yoga"
    GYM = "gym"


class MovementType(str, Enum):
    STATIC = "static"     # scored by hold quality
    DYNAMIC = "dynamic"   # scored by reps and phases


@dataclass
class ExerciseResult:
    """The common structured result. Every analyzer returns exactly this."""
    exercise: str
    exercise_name: str = ""
    movement: str = "static"
    phase: str = "idle"
    score: Optional[float] = None            # 0–100, None when unscorable
    rep_count: int = 0
    hold_duration: float = 0.0
    metrics: List[Metric] = field(default_factory=list)
    errors: List[FormError] = field(default_factory=list)
    primary_error: Optional[FormError] = None
    confidence: float = 0.0                  # how much of the body we could see

    # Supporting detail — still uniform across exercises.
    ready: bool = False
    quality: Optional[float] = None          # consistency (dynamic) / hold quality (static)
    quality_label: str = ""
    stability: Optional[float] = None
    notes: List[str] = field(default_factory=list)
    reference_progress: float = 0.0          # 0=top, 1=bottom; drives the ghost
    # Movement-control read-out, inferred from landmark motion over time.
    # Descriptive only — never a physical-force measurement.
    control: Optional[float] = None          # 0–100, None until the window fills
    unstable_events: int = 0                 # distinct unstable episodes this session

    def metric(self, key: str) -> Optional[Metric]:
        for m in self.metrics:
            if m.key == key:
                return m
        return None

    @property
    def scorable_metrics(self) -> List[Metric]:
        return [m for m in self.metrics if m.score() is not None]


class ExerciseProfile(ABC):
    """Base class for the four movement profiles."""

    key: str = "unknown"
    name: str = "Unknown"
    category: Category = Category.GYM
    movement: MovementType = MovementType.STATIC
    description: str = ""

    # Landmarks without which this exercise cannot be analysed at all.
    required_landmarks: Sequence[LM] = ()

    def __init__(self) -> None:
        self.tracker = ErrorTracker(on_frames=config.ERROR_ON_FRAMES,
                                    off_frames=config.ERROR_OFF_FRAMES)
        self.stability = StabilityTracker()
        self.quality_dynamic = MovementQuality()
        self.quality_static = HoldQuality()
        self.reset()

    # -- lifecycle ---------------------------------------------------------
    def reset(self) -> None:
        """Clear all per-session state. Called on exercise switch and on Start."""
        self.tracker.reset()
        self.stability.reset()
        self.quality_dynamic.reset()
        self.quality_static.reset()
        self.motion = MotionAnalyzer(self.key)
        self.phase = "idle"
        self._reset_state()

    def _reset_state(self) -> None:
        """Subclass hook for its own machines."""

    # -- shared helpers ----------------------------------------------------
    def needed_landmarks(self, frame: EngineFrame) -> Sequence[LM]:
        """What this exercise requires *of this body*.

        A landmark the body map says is untrackable is not a requirement — it is
        a fact about the user. Demanding it would lock them out of the exercise
        entirely, which is the harshest penalty the system could apply for
        something that was never their form.
        """
        return trackable_subset(frame.body_map, self.required_landmarks)

    def can_analyse(self, frame: EngineFrame) -> bool:
        """True when the frame has the minimum landmarks this exercise needs."""
        if not frame.detected:
            return False
        needed = self.needed_landmarks(frame)
        if self.required_landmarks and not needed:
            return False    # nothing this exercise is about is tracked at all
        return frame.pose.has(*needed, min_visibility=config.VISIBILITY_THRESHOLD)

    def observed_confidence(self, frame: EngineFrame) -> float:
        """How much of what this exercise needs we could actually see."""
        if not frame.detected or not self.required_landmarks:
            return 0.0
        needed = self.needed_landmarks(frame)
        if not needed:
            return 0.0
        return frame.pose.visibility_of(*needed)

    def track_stability(self, frame: EngineFrame) -> Optional[float]:
        """Feed hip drift into the stability tracker and return its score."""
        from core.geometry import midpoint, torso_scale
        centre = midpoint(frame, LM.LEFT_HIP, LM.RIGHT_HIP)
        scale = torso_scale(frame)
        if centre is not None and scale is not None:
            self.stability.update(float(centre[0]), float(centre[1]), scale)
        return self.stability.score()

    def metric(self, key: str, label: str, value: Optional[float],
               frame: EngineFrame, angle_key: Optional[str] = None,
               lo: Optional[float] = None, hi: Optional[float] = None,
               falloff: float = 30.0, weight: float = 1.0,
               unit: str = "°") -> Metric:
        """Build a Metric, inheriting status from the engine's angle reading."""
        if angle_key:
            status = frame.status(angle_key)
        elif value is not None:
            status = MetricStatus.MEASURED
        else:
            status = MetricStatus.UNMEASURED

        if value is None and status is MetricStatus.MEASURED:
            status = MetricStatus.UNMEASURED

        return Metric(key=key, label=label, value=value, status=status, unit=unit,
                      lo=lo, hi=hi, falloff=falloff, weight=weight)

    def finish(self, frame: EngineFrame, result: ExerciseResult,
               candidates: Sequence[FormError]) -> ExerciseResult:
        """Common tail: debounce errors, pick the primary, compute the score.

        Every analyzer routes through here so error gating and scoring behave
        identically across the four exercises.
        """
        all_candidates = list(candidates)
        # Motion is only judged on frames we actually trust: never while the
        # engine is calibrating, never on a result that is not ready.
        if result.ready and not frame.calibrating:
            self.motion.update(frame)
            all_candidates.extend(self.motion.candidates(result.phase, frame.timestamp))

        control = self.motion.control_score()
        result.control = control
        result.unstable_events = self.motion.unstable_events
        # weight=0.0 is deliberate: movement control is shown, but must never
        # move the accuracy score the existing metrics and tests define.
        result.metrics.insert(0, Metric(
            key="movement_control", label="Movement control", value=control,
            status=(MetricStatus.MEASURED if control is not None
                    else MetricStatus.UNMEASURED),
            lo=60.0, falloff=60.0, weight=0.0, unit=""))

        active = self.tracker.update(all_candidates, frame.timestamp)
        result.errors = active
        result.primary_error = self.tracker.primary(active)

        penalty = sum(e.penalty for e in active)
        result.score = aggregate_score(result.metrics, penalties=penalty)
        result.confidence = self.observed_confidence(frame)
        result.exercise_name = self.name
        result.movement = self.movement.value
        return result

    # -- the contract ------------------------------------------------------
    @abstractmethod
    def analyse(self, frame: EngineFrame) -> ExerciseResult:
        """Read one EngineFrame and report the full structured result."""

    @abstractmethod
    def recognition_score(self, frame: EngineFrame) -> float:
        """0..1 — how much this frame looks like this exercise.

        Used by the registry for auto-detection. Deliberately heuristic: pose
        signatures over a trained classifier, per the hackathon constraints.
        """

    def not_ready(self, frame: EngineFrame, note: str) -> ExerciseResult:
        """Uniform result for 'we cannot see enough to say anything'."""
        result = ExerciseResult(exercise=self.key, exercise_name=self.name,
                                movement=self.movement.value, phase=self.phase)
        result.notes.append(note)
        result.confidence = self.observed_confidence(frame)
        result.rep_count = getattr(self, "reps", 0)
        return result
