"""Tree Pose — yoga, static balance hold.

Balance is the whole point, so the sway tolerance here is deliberately generous.
A person standing on one leg is never still; a system that flags every micro-
correction as a fault has misunderstood what balancing is. Only sustained drift,
seen over a rolling window, counts.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from core.bodygroups import mark_unavailable, tracked
from core.engine import EngineFrame
from core.errors import FormError, Priority, Severity
from core.geometry import (angle_to_horizontal, angle_to_vertical, distance,
                           torso_scale)
from core.landmarks import LM, MetricStatus
from core.phases import HoldStateMachine
from core.scoring import StabilityTracker

from .base import Category, ExerciseProfile, ExerciseResult, MovementType

LIFT_THRESHOLD = 0.35        # raised foot height, in torso lengths
STANDING_LEG_MIN = 150.0     # standing leg should stay long
TORSO_TOLERANCE = 15.0
HIP_TILT_LIMIT = 12.0
SHOULDER_TILT_LIMIT = 12.0
STABILITY_FLOOR = 40.0       # below this we mention the wobble


class TreePoseProfile(ExerciseProfile):
    key = "tree_pose"
    name = "Tree Pose"
    category = Category.YOGA
    movement = MovementType.STATIC
    description = "Static balance. Standing-leg stability, hip and shoulder level, body sway."

    required_landmarks = (
        LM.LEFT_HIP, LM.RIGHT_HIP,
        LM.LEFT_KNEE, LM.RIGHT_KNEE,
        LM.LEFT_ANKLE, LM.RIGHT_ANKLE,
        LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER,
    )

    def _reset_state(self) -> None:
        self.hold = HoldStateMachine()
        self.phase = "setting up"
        self.standing_side: Optional[str] = None
        # Balance deserves a longer, more forgiving window than a static stance.
        self.stability = StabilityTracker(window=60, tolerance=0.16)

    # ------------------------------------------------------------------
    def _standing_side(self, frame: EngineFrame) -> Tuple[Optional[str], Optional[float]]:
        """Lower foot is the standing leg; returns how far the other one is lifted."""
        la, ra = frame.pose.get(LM.LEFT_ANKLE), frame.pose.get(LM.RIGHT_ANKLE)
        scale = torso_scale(frame)
        if la is None or ra is None or scale is None:
            return None, None
        # World +y points down, so the larger wy is the lower foot.
        if la.wy > ra.wy:
            return "left", float((la.wy - ra.wy) / scale)
        return "right", float((ra.wy - la.wy) / scale)

    # ------------------------------------------------------------------
    def analyse(self, frame: EngineFrame) -> ExerciseResult:
        if not self.can_analyse(frame):
            return self.not_ready(
                frame, "Step back so your full body, including both feet, is visible.")

        result = ExerciseResult(exercise=self.key, ready=True)

        side, lift = self._standing_side(frame)
        self.standing_side = side
        stability = self.track_stability(frame)

        # Level-ness is a comparison between two sides: only ask for it when this
        # body has both of them tracked.
        hips_ok = tracked(frame.body_map, LM.LEFT_HIP, LM.RIGHT_HIP)
        shoulders_ok = tracked(frame.body_map, LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER)

        torso_lean = angle_to_vertical(frame, LM.LEFT_SHOULDER, LM.LEFT_HIP)
        hip_tilt = angle_to_horizontal(frame, LM.LEFT_HIP, LM.RIGHT_HIP) if hips_ok else None
        shoulder_tilt = (angle_to_horizontal(frame, LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER)
                         if shoulders_ok else None)
        standing_leg = frame.angle(f"{side}_knee") if side else None

        in_position = (
            lift is not None and lift >= LIFT_THRESHOLD
            and (torso_lean is None or torso_lean <= TORSO_TOLERANCE + 8)
        )
        self.phase = self.hold.update(in_position, frame.timestamp)
        self.quality_static.update(in_position, stability)

        result.phase = self.phase
        result.hold_duration = self.hold.duration(frame.timestamp)
        result.stability = stability
        result.reference_progress = 1.0
        result.quality = self.quality_static.quality(self.hold.in_position_ratio)
        result.quality_label = "Hold quality"

        result.metrics = [
            self.metric("foot_lift", "Foot lift",
                        None if lift is None else lift * 100, frame,
                        lo=LIFT_THRESHOLD * 100, falloff=30, weight=1.3, unit="%"),
            self.metric("standing_leg", "Standing leg", standing_leg, frame,
                        f"{side}_knee" if side else None,
                        lo=STANDING_LEG_MIN, falloff=30, weight=1.1),
            self.metric("stability", "Stability", stability, frame,
                        lo=60, falloff=60, weight=1.5, unit=""),
            self.metric("torso_lean", "Torso lean", torso_lean, frame,
                        hi=TORSO_TOLERANCE, falloff=25, weight=1.2),
            self.metric("hip_tilt", "Hip level", hip_tilt, frame,
                        hi=HIP_TILT_LIMIT, falloff=20, weight=1.0),
            self.metric("shoulder_tilt", "Shoulder level", shoulder_tilt, frame,
                        hi=SHOULDER_TILT_LIMIT, falloff=20, weight=0.8),
        ]

        # Anything this body does not have reads "not applicable" and leaves the
        # score alone, rather than reading as a fault the user cannot fix.
        result.metrics = mark_unavailable(result.metrics, frame.body_map, {
            "foot_lift": (LM.LEFT_ANKLE, LM.RIGHT_ANKLE),
            "hip_tilt": (LM.LEFT_HIP, LM.RIGHT_HIP),
            "shoulder_tilt": (LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER),
            "torso_lean": (LM.LEFT_SHOULDER, LM.LEFT_HIP),
        })

        candidates = self._detect(frame, side, lift, stability, torso_lean,
                                  hip_tilt, shoulder_tilt, standing_leg)
        return self.finish(frame, result, candidates)

    # ------------------------------------------------------------------
    def _detect(self, frame: EngineFrame, side: Optional[str], lift: Optional[float],
                stability: Optional[float], torso_lean: Optional[float],
                hip_tilt: Optional[float], shoulder_tilt: Optional[float],
                standing_leg: Optional[float]) -> List[FormError]:
        found: List[FormError] = []
        holding = self.phase == "holding"

        # --- sustained drift. Only judged once actually balancing. ---------
        if holding and stability is not None and stability < STABILITY_FLOOR:
            found.append(FormError(
                key="unstable",
                message="You're drifting — fix your eyes on one point ahead of you.",
                cue="STEADY -> FIX YOUR GAZE",
                severity=Severity.MAJOR if stability < 20 else Severity.MINOR,
                priority=Priority.STABILITY,
                confidence=0.85,
                landmarks=(LM.LEFT_HIP, LM.RIGHT_HIP), phase=self.phase,
            ))

        # --- raised foot too low to be the pose ---------------------------
        if (lift is not None and 0.08 < lift < LIFT_THRESHOLD
                and tracked(frame.body_map, LM.LEFT_ANKLE, LM.RIGHT_ANKLE)):
            found.append(FormError(
                key="foot_too_low",
                message="Bring your raised foot higher onto your standing leg.",
                cue="FOOT HIGHER",
                severity=Severity.MINOR,
                priority=Priority.RANGE,
                confidence=frame.pose.visibility_of(LM.LEFT_ANKLE, LM.RIGHT_ANKLE),
                landmarks=(LM.LEFT_ANKLE if side == "right" else LM.RIGHT_ANKLE,),
                phase=self.phase,
            ))

        if (torso_lean is not None and torso_lean > TORSO_TOLERANCE
                and tracked(frame.body_map, LM.LEFT_SHOULDER, LM.LEFT_HIP)):
            found.append(FormError(
                key="torso_lean",
                message="Lengthen up through the crown of your head and stack your torso.",
                cue="STRAIGHTEN TORSO",
                severity=Severity.MINOR,
                priority=Priority.ALIGNMENT,
                confidence=0.85,
                landmarks=(LM.LEFT_SHOULDER, LM.LEFT_HIP), phase=self.phase,
                caused_by="unstable",
            ))

        if hip_tilt is not None and hip_tilt > HIP_TILT_LIMIT:
            found.append(FormError(
                key="hip_drop",
                message="Level your hips — don't let the raised side push out.",
                cue="LEVEL HIPS",
                severity=Severity.MINOR,
                priority=Priority.ALIGNMENT,
                confidence=frame.pose.visibility_of(LM.LEFT_HIP, LM.RIGHT_HIP),
                landmarks=(LM.LEFT_HIP, LM.RIGHT_HIP), phase=self.phase,
            ))

        if standing_leg is not None and side is not None:
            key = f"{side}_knee"
            if frame.status(key) is MetricStatus.MEASURED and standing_leg < STANDING_LEG_MIN:
                found.append(FormError(
                    key="standing_leg_soft",
                    message="Straighten your standing leg and press the foot into the floor.",
                    cue="STRAIGHTEN STANDING LEG",
                    severity=Severity.MINOR,
                    priority=Priority.STABILITY,
                    confidence=frame.angles[key].confidence,
                    landmarks=(LM.LEFT_KNEE if side == "left" else LM.RIGHT_KNEE,),
                    phase=self.phase,
                ))

        if shoulder_tilt is not None and shoulder_tilt > SHOULDER_TILT_LIMIT:
            found.append(FormError(
                key="shoulder_tilt",
                message="Even out your shoulders.",
                cue="LEVEL SHOULDERS",
                severity=Severity.MINOR,
                priority=Priority.REFINEMENT,
                confidence=frame.pose.visibility_of(LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER),
                landmarks=(LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER), phase=self.phase,
            ))

        return found

    # ------------------------------------------------------------------
    def recognition_score(self, frame: EngineFrame) -> float:
        """Signature: one foot clearly raised, standing leg straight, narrow base."""
        if not self.can_analyse(frame):
            return 0.0

        side, lift = self._standing_side(frame)
        if lift is None or lift < LIFT_THRESHOLD:
            return 0.0

        score = 0.50
        if side is not None:
            angle = frame.angle(f"{side}_knee")
            if angle is not None and angle > 155:
                score += 0.25

        ankle_span = distance(frame, LM.LEFT_ANKLE, LM.RIGHT_ANKLE)
        hip_span = distance(frame, LM.LEFT_HIP, LM.RIGHT_HIP)
        if ankle_span and hip_span and ankle_span < hip_span * 1.5:
            score += 0.25

        return min(1.0, score)
