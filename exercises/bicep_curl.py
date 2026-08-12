"""Bicep curl — gym, dynamic.

Tracks START → CURL → PEAK → RETURN per arm. The interesting faults here are not
about the elbow angle itself but about what the rest of the body does to help:
a drifting elbow and a swinging torso both mean the biceps stopped doing the work.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from core.bodygroups import (LEFT_ARM, RIGHT_ARM, has_group, landmarks_for,
                             mark_unavailable, tracked)
from core.engine import EngineFrame
from core.errors import FormError, Priority, Severity
from core.geometry import angle_to_vertical, distance
from core.landmarks import LM, MetricStatus
from core.phases import PhaseNames, RepPhaseMachine

from .base import Category, ExerciseProfile, ExerciseResult, MovementType

TOP_ANGLE = 150.0      # arm considered extended
PEAK_ANGLE = 60.0      # arm considered fully curled
ELBOW_DRIFT_LIMIT = 28.0   # upper-arm lean from vertical, degrees
SWING_LIMIT = 18.0         # torso lean from vertical, degrees
ROM_TARGET = 100.0         # degrees of travel we want per rep

PHASES = PhaseNames(top="start", descending="curl", bottom="peak", ascending="return")


class BicepCurlProfile(ExerciseProfile):
    key = "bicep_curl"
    name = "Bicep Curl"
    category = Category.GYM
    movement = MovementType.DYNAMIC
    description = "Per-arm reps through start, curl, peak and return. Elbow stability and range."

    required_landmarks = (
        LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER,
        LM.LEFT_ELBOW, LM.RIGHT_ELBOW,
        LM.LEFT_WRIST, LM.RIGHT_WRIST,
    )

    def _reset_state(self) -> None:
        # One machine per arm — the arms are genuinely independent here.
        self.machines: Dict[str, RepPhaseMachine] = {
            "left": RepPhaseMachine(TOP_ANGLE, PEAK_ANGLE, PHASES),
            "right": RepPhaseMachine(TOP_ANGLE, PEAK_ANGLE, PHASES),
        }
        self.phase = PHASES.top
        self.reps = 0

    # ------------------------------------------------------------------
    def _lead_arm(self, frame: EngineFrame) -> str:
        """Whichever arm is further through its curl leads the displayed phase."""
        left, right = frame.angle("left_elbow"), frame.angle("right_elbow")
        if left is None:
            return "right"
        if right is None:
            return "left"
        return "left" if left <= right else "right"

    def _elbow_drift(self, frame: EngineFrame, side: str) -> Optional[float]:
        shoulder = LM.LEFT_SHOULDER if side == "left" else LM.RIGHT_SHOULDER
        elbow = LM.LEFT_ELBOW if side == "left" else LM.RIGHT_ELBOW
        return angle_to_vertical(frame, shoulder, elbow)

    @staticmethod
    def _arm_tracked(frame: EngineFrame, side: str) -> bool:
        """Whether this user's body map includes the arm at all."""
        return has_group(frame.body_map, LEFT_ARM if side == "left" else RIGHT_ARM)

    def _rom(self, side: str) -> Optional[float]:
        """Degrees travelled on the last completed rep of this arm."""
        last = self.machines[side].last_rep
        return None if last is None else TOP_ANGLE - last.peak

    # ------------------------------------------------------------------
    def analyse(self, frame: EngineFrame) -> ExerciseResult:
        if not self.can_analyse(frame):
            return self.not_ready(frame, "Make sure both arms are visible from shoulder to wrist.")

        result = ExerciseResult(exercise=self.key, ready=True)

        left = frame.angle("left_elbow")
        right = frame.angle("right_elbow")

        self.machines["left"].update(left, frame.timestamp)
        self.machines["right"].update(right, frame.timestamp)

        lead = self._lead_arm(frame)
        self.phase = self.machines[lead].phase
        self.reps = max(m.rep_count for m in self.machines.values())

        # Consistency is judged on the arm doing the most work.
        busiest = max(self.machines.values(), key=lambda m: m.rep_count)
        self.quality_dynamic.sync(busiest.reps)

        stability = self.track_stability(frame)
        swing = (angle_to_vertical(frame, LM.LEFT_SHOULDER, LM.LEFT_HIP)
                 if tracked(frame.body_map, LM.LEFT_SHOULDER, LM.LEFT_HIP) else None)

        result.phase = self.phase
        result.rep_count = self.reps
        result.stability = stability
        result.reference_progress = self.machines[lead].depth_progress
        result.quality = self.quality_dynamic.consistency()
        result.quality_label = "Movement consistency"

        left_drift = self._elbow_drift(frame, "left")
        right_drift = self._elbow_drift(frame, "right")
        worst_drift = max([d for d in (left_drift, right_drift) if d is not None],
                          default=None)
        rom = max([r for r in (self._rom("left"), self._rom("right"))
                   if r is not None], default=None)

        result.metrics = [
            self.metric("left_elbow", "Left elbow", left, frame, "left_elbow",
                        lo=PEAK_ANGLE - 30, hi=TOP_ANGLE + 25, falloff=40, weight=0.5),
            self.metric("right_elbow", "Right elbow", right, frame, "right_elbow",
                        lo=PEAK_ANGLE - 30, hi=TOP_ANGLE + 25, falloff=40, weight=0.5),
            self.metric("rom", "Range of motion", rom, frame,
                        lo=ROM_TARGET, falloff=45, weight=1.4),
            self.metric("elbow_stability", "Elbow stability", worst_drift, frame,
                        hi=ELBOW_DRIFT_LIMIT, falloff=30, weight=1.3),
            self.metric("body_swing", "Body swing", swing, frame,
                        hi=SWING_LIMIT, falloff=25, weight=1.2),
            self.metric("left_reps", "Left reps",
                        float(self.machines["left"].rep_count), frame, unit=""),
            self.metric("right_reps", "Right reps",
                        float(self.machines["right"].rep_count), frame, unit=""),
        ]

        # An arm this body does not have reports "not applicable", never zero
        # reps and never a fault.
        result.metrics = mark_unavailable(result.metrics, frame.body_map, {
            "left_reps": landmarks_for(LEFT_ARM),
            "right_reps": landmarks_for(RIGHT_ARM),
            "left_elbow": landmarks_for(LEFT_ARM),
            "right_elbow": landmarks_for(RIGHT_ARM),
            "body_swing": (LM.LEFT_SHOULDER, LM.LEFT_HIP),
        })

        candidates = self._detect(frame, left, right, swing)
        for error in candidates:
            busiest.note_error(error.key)
        return self.finish(frame, result, candidates)

    # ------------------------------------------------------------------
    def _detect(self, frame: EngineFrame, left: Optional[float],
                right: Optional[float], swing: Optional[float]) -> List[FormError]:
        found: List[FormError] = []

        # --- torso swing. Momentum replacing the biceps. -------------------
        if swing is not None and swing > SWING_LIMIT:
            found.append(FormError(
                key="body_swing",
                message="You're swinging your body — slow down and let the arm do the work.",
                cue="STOP SWINGING",
                severity=Severity.MAJOR,
                priority=Priority.ALIGNMENT,
                confidence=0.8,
                landmarks=(LM.LEFT_SHOULDER, LM.LEFT_HIP),
                phase=self.phase,
            ))

        for side, value in (("left", left), ("right", right)):
            angle_key = f"{side}_elbow"
            # An untracked arm is not a form fault, so it never produces one.
            if not self._arm_tracked(frame, side):
                continue
            if frame.status(angle_key) is not MetricStatus.MEASURED:
                continue
            confidence = frame.angles[angle_key].confidence
            elbow_lm = LM.LEFT_ELBOW if side == "left" else LM.RIGHT_ELBOW

            # --- elbow drifting away from the ribs ------------------------
            drift = self._elbow_drift(frame, side)
            if drift is not None and drift > ELBOW_DRIFT_LIMIT:
                found.append(FormError(
                    key=f"{side}_elbow_drift",
                    message=f"Keep your {side} elbow tucked in at your side.",
                    cue=f"KEEP {side.upper()} ELBOW STABLE",
                    severity=Severity.MAJOR if drift > 45 else Severity.MINOR,
                    priority=Priority.STABILITY,
                    confidence=confidence,
                    landmarks=(elbow_lm,),
                    phase=self.phase,
                ))

            # --- partial range on the finished rep ------------------------
            rom = self._rom(side)
            if rom is not None and rom < ROM_TARGET:
                found.append(FormError(
                    key=f"{side}_partial_rom",
                    message=f"Fuller range on your {side} arm — curl higher and extend further.",
                    cue=f"FULL RANGE {side.upper()}",
                    severity=Severity.MINOR,
                    priority=Priority.RANGE,
                    confidence=confidence,
                    landmarks=(elbow_lm,),
                    phase=self.phase,
                ))

        return found

    # ------------------------------------------------------------------
    def recognition_score(self, frame: EngineFrame) -> float:
        """Signature: upright, legs straight, at least one elbow genuinely curled."""
        if not self.can_analyse(frame):
            return 0.0

        left, right = frame.angle("left_elbow"), frame.angle("right_elbow")
        elbows = [v for v in (left, right) if v is not None]
        if not elbows or not any(v <= 150 for v in elbows):
            return 0.0

        score = 0.40
        if any(v < 90 for v in elbows):
            score += 0.20

        lk, rk = frame.angle("left_knee"), frame.angle("right_knee")
        knees = [v for v in (lk, rk) if v is not None]
        if knees:
            score += 0.30 if all(v > 160 for v in knees) else -0.30
        else:
            score += 0.15

        ls, rs = frame.angle("left_shoulder"), frame.angle("right_shoulder")
        if ls is not None and rs is not None and ls < 50 and rs < 50:
            score += 0.20

        return max(0.0, min(1.0, score))
