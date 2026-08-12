"""Squat — gym, dynamic.

Full movement tracking: STANDING → DESCENDING → BOTTOM → ASCENDING → STANDING.
Errors are evaluated against the phase they belong to, because most squat faults
only mean anything at a particular point in the rep — knees caving matters at the
bottom, depth is a property of the finished rep, not of any single frame.
"""
from __future__ import annotations

from typing import List, Optional

from core.engine import EngineFrame
from core.errors import FormError, Priority, Severity
from core.geometry import (angle_to_vertical, distance, horizontal_offset,
                           joint_angle)
from core.landmarks import LM, MetricStatus
from core.phases import PhaseNames, RepPhaseMachine

from .base import Category, ExerciseProfile, ExerciseResult, MovementType

# Phase boundaries on mean knee angle.
TOP_ANGLE = 160.0
BOTTOM_ANGLE = 115.0

DEPTH_TARGET = 100.0        # knee angle we want the rep to reach
KNEE_TRACK_TOLERANCE = 0.12  # torso-relative inward drift before it counts
BACK_ANGLE_LIMIT = 55.0      # torso lean from vertical at the bottom
SYMMETRY_LIMIT = 18.0        # left/right knee difference, degrees

PHASES = PhaseNames(top="standing", descending="descending",
                    bottom="bottom", ascending="ascending")


class SquatProfile(ExerciseProfile):
    key = "squat"
    name = "Squat"
    category = Category.GYM
    movement = MovementType.DYNAMIC
    description = "Full-cycle lower-body movement: depth, knee tracking, back angle, symmetry."

    required_landmarks = (
        LM.LEFT_HIP, LM.RIGHT_HIP,
        LM.LEFT_KNEE, LM.RIGHT_KNEE,
        LM.LEFT_ANKLE, LM.RIGHT_ANKLE,
    )

    def _reset_state(self) -> None:
        self.machine = RepPhaseMachine(TOP_ANGLE, BOTTOM_ANGLE, PHASES)
        self.phase = PHASES.top
        self.reps = 0
        self._last_rep_shallow = False

    # ------------------------------------------------------------------
    def _mean_knee(self, frame: EngineFrame) -> Optional[float]:
        left, right = frame.angle("left_knee"), frame.angle("right_knee")
        vals = [v for v in (left, right) if v is not None]
        return sum(vals) / len(vals) if vals else None

    def _mean_hip(self, frame: EngineFrame) -> Optional[float]:
        left, right = frame.angle("left_hip"), frame.angle("right_hip")
        vals = [v for v in (left, right) if v is not None]
        return sum(vals) / len(vals) if vals else None

    def _back_angle(self, frame: EngineFrame) -> Optional[float]:
        """Torso lean from vertical, averaged over both sides where visible."""
        vals = [v for v in (
            angle_to_vertical(frame, LM.LEFT_SHOULDER, LM.LEFT_HIP),
            angle_to_vertical(frame, LM.RIGHT_SHOULDER, LM.RIGHT_HIP),
        ) if v is not None]
        return sum(vals) / len(vals) if vals else None

    def _knee_drift(self, frame: EngineFrame, side: str) -> Optional[float]:
        """How far the knee sits inside the ankle, in torso lengths. Positive = inward."""
        knee = LM.LEFT_KNEE if side == "left" else LM.RIGHT_KNEE
        ankle = LM.LEFT_ANKLE if side == "left" else LM.RIGHT_ANKLE
        offset = horizontal_offset(frame, knee, ankle)
        if offset is None:
            return None
        # World +x points to the image right. The person's left side sits at
        # negative x, so that knee moving inward increases its offset; on the
        # right side inward decreases it.
        return offset if side == "left" else -offset

    def _worst_knee_drift(self, frame: EngineFrame) -> Optional[float]:
        """Largest inward drift across the sides this user actually has tracked."""
        drifts = [d for d in (self._knee_drift(frame, "left"),
                              self._knee_drift(frame, "right")) if d is not None]
        return max(drifts) if drifts else None

    def _tracked_side_key(self, frame: EngineFrame, joint: str) -> Optional[str]:
        """Pick a side that is actually measured for a both-legs metric.

        knee_angle and hip_angle are averaged across whichever legs are visible,
        so pinning their status to "left_*" marked them not-applicable for a user
        whose left leg isn't tracked — even though the value came from the right.
        That turned a missing landmark into a lost metric, and a lost metric
        re-weights the score against the very user Adaptive Mode exists for.
        """
        for side in ("left", "right"):
            key = f"{side}_{joint}"
            if frame.status(key) is MetricStatus.MEASURED:
                return key
        return f"left_{joint}"

    def _symmetry(self, frame: EngineFrame) -> Optional[float]:
        left, right = frame.angle("left_knee"), frame.angle("right_knee")
        if left is None or right is None:
            return None
        return abs(left - right)

    @staticmethod
    def _symmetric_body(frame: EngineFrame) -> bool:
        """Whether left/right comparison means anything for this user."""
        return frame.body_map is None or frame.body_map.symmetry_applicable()

    # ------------------------------------------------------------------
    def analyse(self, frame: EngineFrame) -> ExerciseResult:
        if not self.can_analyse(frame):
            return self.not_ready(frame, "Step back so your hips, knees and ankles are all in view.")

        result = ExerciseResult(exercise=self.key, ready=True)

        knee = self._mean_knee(frame)
        hip = self._mean_hip(frame)
        back = self._back_angle(frame)
        # Only compare sides when this user has two of them tracked.
        symmetric = self._symmetric_body(frame)
        symmetry = self._symmetry(frame) if symmetric else None

        # --- phase machine -----------------------------------------------
        previous_reps = self.machine.rep_count
        self.phase = self.machine.update(knee, frame.timestamp)

        if self.machine.rep_count > previous_reps:
            last = self.machine.last_rep
            self._last_rep_shallow = bool(last and last.peak > DEPTH_TARGET)

        if self.phase == PHASES.bottom:
            self._last_rep_shallow = False   # a new rep is under way

        self.reps = self.machine.rep_count
        self.quality_dynamic.sync(self.machine.reps)

        stability = self.track_stability(frame)

        result.phase = self.phase
        result.rep_count = self.reps
        result.stability = stability
        result.reference_progress = self.machine.depth_progress
        result.quality = self.quality_dynamic.consistency()
        result.quality_label = "Movement consistency"

        # --- metrics -------------------------------------------------------
        deepest = self.machine.last_rep.peak if self.machine.last_rep else None
        drift = self._worst_knee_drift(frame)
        result.metrics = [
            self.metric("knee_angle", "Knee angle", knee, frame,
                        self._tracked_side_key(frame, "knee"),
                        lo=BOTTOM_ANGLE - 40, hi=TOP_ANGLE + 20, falloff=40, weight=0.6),
            self.metric("depth", "Last rep depth", deepest, frame,
                        hi=DEPTH_TARGET, falloff=35, weight=1.4),
            self.metric("hip_angle", "Hip angle", hip, frame,
                        self._tracked_side_key(frame, "hip"),
                        lo=60, hi=180, falloff=40, weight=0.6),
            self.metric("back_angle", "Back angle", back, frame,
                        hi=BACK_ANGLE_LIMIT, falloff=35, weight=1.2),
            self.metric("knee_track", "Knee tracking",
                        None if drift is None else drift * 100,
                        frame, hi=KNEE_TRACK_TOLERANCE * 100, falloff=25,
                        weight=1.4, unit="%"),
            self.metric("symmetry", "L/R symmetry", symmetry, frame,
                        hi=SYMMETRY_LIMIT, falloff=25, weight=1.0),
            self.metric("stability", "Stability", stability, frame,
                        lo=60, falloff=60, weight=0.8, unit=""),
        ]

        if not symmetric:
            # Not "you failed symmetry" — there is no symmetry to measure.
            result.metrics = [m for m in result.metrics if m.key != "symmetry"]

        candidates = self._detect(frame, knee, back, symmetry)
        for error in candidates:
            self.machine.note_error(error.key)
        return self.finish(frame, result, candidates)

    # ------------------------------------------------------------------
    def _detect(self, frame: EngineFrame, knee: Optional[float],
                back: Optional[float], symmetry: Optional[float]) -> List[FormError]:
        found: List[FormError] = []
        loaded = self.phase in (PHASES.descending, PHASES.bottom, PHASES.ascending)

        # --- knee valgus. Only judged under load, where it matters. --------
        for side in ("left", "right"):
            angle_key = f"{side}_knee"
            if frame.status(angle_key) is not MetricStatus.MEASURED or not loaded:
                continue
            drift = self._knee_drift(frame, side)
            if drift is None or drift <= KNEE_TRACK_TOLERANCE:
                continue
            found.append(FormError(
                key=f"{side}_knee_valgus",
                message=f"Your {side} knee is collapsing inward — push it out over your foot.",
                cue=f"{side.upper()} KNEE -> OUT",
                severity=Severity.MAJOR if drift > 0.20 else Severity.MINOR,
                priority=Priority.SAFETY,
                confidence=frame.angles[angle_key].confidence,
                landmarks=(LM.LEFT_KNEE if side == "left" else LM.RIGHT_KNEE,),
                phase=self.phase,
                caused_by="ankle_dorsiflexion",
            ))

        # --- back angle. A rounded, folded torso under load. ---------------
        if back is not None and loaded and back > BACK_ANGLE_LIMIT:
            found.append(FormError(
                key="back_angle",
                message="Your chest is dropping — lift it and keep your back closer to upright.",
                cue="STRAIGHTEN TORSO",
                severity=Severity.MAJOR if back > 70 else Severity.MINOR,
                priority=Priority.ALIGNMENT,
                confidence=0.85,
                landmarks=(LM.LEFT_SHOULDER, LM.LEFT_HIP),
                phase=self.phase,
                caused_by="ankle_dorsiflexion",
            ))

        # --- asymmetry. One side working harder than the other. ------------
        # Skipped outright for a body without two tracked legs and arms: there is
        # no "other side" for this user, so there is nothing to even out.
        if (symmetry is not None and self._symmetric_body(frame)
                and loaded and symmetry > SYMMETRY_LIMIT):
            found.append(FormError(
                key="asymmetry",
                message=f"You're favouring one side — {symmetry:.0f}° difference between your knees.",
                cue="EVEN OUT BOTH LEGS",
                severity=Severity.MINOR,
                priority=Priority.ALIGNMENT,
                confidence=0.8,
                landmarks=(LM.LEFT_KNEE, LM.RIGHT_KNEE),
                phase=self.phase,
            ))

        # --- depth. A property of the completed rep, not of any one frame. --
        if self._last_rep_shallow:
            found.append(FormError(
                key="shallow_depth",
                message="Sit deeper on the next rep — aim for thighs closer to parallel.",
                cue="GO DEEPER",
                severity=Severity.MINOR,
                priority=Priority.RANGE,
                confidence=0.9,
                landmarks=(LM.LEFT_KNEE, LM.RIGHT_KNEE),
                phase=self.phase,
            ))

        return found

    # ------------------------------------------------------------------
    def recognition_score(self, frame: EngineFrame) -> float:
        """Signature: both knees bent together, narrow stance, arms not held wide.

        Gated rather than purely additive — without a genuine symmetric knee bend
        this is not a squat, however much else matches.
        """
        if not self.can_analyse(frame):
            return 0.0

        left, right = frame.angle("left_knee"), frame.angle("right_knee")
        if left is None or right is None:
            return 0.0
        if left > 165 or right > 165 or abs(left - right) > 30:
            return 0.0

        score = 0.45
        mean_knee = (left + right) / 2.0
        if mean_knee < 140:
            score += 0.25
        elif mean_knee < 155:
            score += 0.12

        wrist_span = distance(frame, LM.LEFT_WRIST, LM.RIGHT_WRIST)
        shoulder_span = distance(frame, LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER)
        if wrist_span and shoulder_span:
            score += 0.30 if wrist_span < shoulder_span * 1.8 else -0.25

        ankle_span = distance(frame, LM.LEFT_ANKLE, LM.RIGHT_ANKLE)
        hip_span = distance(frame, LM.LEFT_HIP, LM.RIGHT_HIP)
        if ankle_span and hip_span and ankle_span < hip_span * 1.8:
            score += 0.15

        return max(0.0, min(1.0, score))
