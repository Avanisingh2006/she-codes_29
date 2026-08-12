"""Warrior II — yoga, static hold.

Every axis of the pose is checked independently — front knee, hips, shoulders,
arms, torso — because Warrior II fails in several unrelated ways and lumping them
into one "wrong" tells the user nothing. The hold only counts while the whole
shape is intact.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from core.bodygroups import mark_unavailable, tracked
from core.engine import EngineFrame
from core.errors import FormError, Priority, Severity
from core.geometry import (angle_to_horizontal, angle_to_vertical, distance,
                           midpoint, torso_scale)
from core.landmarks import LM, MetricStatus
from core.phases import HoldStateMachine

from .base import Category, ExerciseProfile, ExerciseResult, MovementType

FRONT_KNEE_TARGET = (80.0, 110.0)
BACK_LEG_MIN = 150.0        # rear leg should stay long
ARM_LEVEL_TOLERANCE = 15.0  # degrees off horizontal between the hands
ARM_DROP_TOLERANCE = 0.22   # how far below shoulder height the hands may sit
SHOULDER_TILT_LIMIT = 12.0  # degrees off level
HIP_TILT_LIMIT = 14.0
TORSO_TOLERANCE = 20.0


class WarriorTwoProfile(ExerciseProfile):
    key = "warrior_2"
    name = "Warrior II"
    category = Category.YOGA
    movement = MovementType.STATIC
    description = "Static hold. Front knee, hips, shoulders, arm line, torso and stability."

    required_landmarks = (
        LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER,
        LM.LEFT_WRIST, LM.RIGHT_WRIST,
        LM.LEFT_HIP, LM.RIGHT_HIP,
        LM.LEFT_KNEE, LM.RIGHT_KNEE,
        LM.LEFT_ANKLE, LM.RIGHT_ANKLE,
    )

    def _reset_state(self) -> None:
        self.hold = HoldStateMachine()
        self.phase = "setting up"
        self.front_side: Optional[str] = None

    # ------------------------------------------------------------------
    def _legs(self, frame: EngineFrame) -> Tuple[Optional[str], Optional[float], Optional[float]]:
        """Front leg is the more bent one. Returns (side, front angle, back angle)."""
        left, right = frame.angle("left_knee"), frame.angle("right_knee")
        if left is None and right is None:
            return None, None, None
        if left is None:
            return "right", right, None
        if right is None:
            return "left", left, None
        return ("left", left, right) if left < right else ("right", right, left)

    def _arm_drop(self, frame: EngineFrame) -> Optional[float]:
        """How far below shoulder height the hands sit, as a fraction of torso length.

        Distinct from arm *level*: both arms can sag together and still be
        perfectly level with each other, so levelness alone misses the most
        common Warrior II fault.
        """
        wrists = midpoint(frame, LM.LEFT_WRIST, LM.RIGHT_WRIST)
        shoulders = midpoint(frame, LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER)
        scale = torso_scale(frame)
        if wrists is None or shoulders is None or scale is None:
            return None
        return float((wrists[1] - shoulders[1]) / scale)

    # ------------------------------------------------------------------
    def analyse(self, frame: EngineFrame) -> ExerciseResult:
        if not self.can_analyse(frame):
            return self.not_ready(frame, "Step back so your whole body is in frame.")

        result = ExerciseResult(exercise=self.key, ready=True)

        # Each of these compares one side against the other, so each stands down
        # unless this user's body map has both sides.
        arms_ok = tracked(frame.body_map, LM.LEFT_WRIST, LM.RIGHT_WRIST)
        hips_ok = tracked(frame.body_map, LM.LEFT_HIP, LM.RIGHT_HIP)
        shoulders_ok = tracked(frame.body_map, LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER)

        arm_level = angle_to_horizontal(frame, LM.LEFT_WRIST, LM.RIGHT_WRIST) if arms_ok else None
        arm_drop = self._arm_drop(frame) if arms_ok and shoulders_ok else None
        torso_lean = angle_to_vertical(frame, LM.LEFT_SHOULDER, LM.LEFT_HIP)
        shoulder_tilt = (angle_to_horizontal(frame, LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER)
                         if shoulders_ok else None)
        hip_tilt = angle_to_horizontal(frame, LM.LEFT_HIP, LM.RIGHT_HIP) if hips_ok else None
        side, front_knee, back_knee = self._legs(frame)
        self.front_side = side

        stability = self.track_stability(frame)

        in_position = (
            arm_level is not None and arm_level <= ARM_LEVEL_TOLERANCE
            and arm_drop is not None and arm_drop <= ARM_DROP_TOLERANCE
            and front_knee is not None
            and FRONT_KNEE_TARGET[0] <= front_knee <= FRONT_KNEE_TARGET[1]
            and (torso_lean is None or torso_lean <= TORSO_TOLERANCE)
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
            self.metric("front_knee", "Front knee", front_knee, frame,
                        f"{side}_knee" if side else None,
                        lo=FRONT_KNEE_TARGET[0], hi=FRONT_KNEE_TARGET[1],
                        falloff=35, weight=1.4),
            self.metric("back_leg", "Back leg", back_knee, frame,
                        lo=BACK_LEG_MIN, falloff=35, weight=0.9),
            self.metric("arm_level", "Arm level", arm_level, frame,
                        hi=ARM_LEVEL_TOLERANCE, falloff=25, weight=1.1),
            self.metric("arm_height", "Arm height",
                        None if arm_drop is None else arm_drop * 100, frame,
                        hi=ARM_DROP_TOLERANCE * 100, falloff=35, weight=1.2, unit="%"),
            self.metric("shoulder_tilt", "Shoulder level", shoulder_tilt, frame,
                        hi=SHOULDER_TILT_LIMIT, falloff=20, weight=0.9),
            self.metric("hip_tilt", "Hip level", hip_tilt, frame,
                        hi=HIP_TILT_LIMIT, falloff=20, weight=0.9),
            self.metric("torso_lean", "Torso lean", torso_lean, frame,
                        hi=TORSO_TOLERANCE, falloff=25, weight=1.1),
            self.metric("stability", "Stability", stability, frame,
                        lo=60, falloff=60, weight=0.8, unit=""),
        ]

        # Untracked parts of the body report "not applicable" and drop out of the
        # score entirely — never a low reading for something that isn't there.
        result.metrics = mark_unavailable(result.metrics, frame.body_map, {
            "arm_level": (LM.LEFT_WRIST, LM.RIGHT_WRIST),
            "arm_height": (LM.LEFT_WRIST, LM.RIGHT_WRIST,
                           LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER),
            "shoulder_tilt": (LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER),
            "hip_tilt": (LM.LEFT_HIP, LM.RIGHT_HIP),
            "torso_lean": (LM.LEFT_SHOULDER, LM.LEFT_HIP),
        })

        candidates = self._detect(frame, arm_level, arm_drop, torso_lean,
                                  shoulder_tilt, hip_tilt, side, front_knee, back_knee,
                                  stability)
        return self.finish(frame, result, candidates)

    # ------------------------------------------------------------------
    def _detect(self, frame: EngineFrame, arm_level: Optional[float],
                arm_drop: Optional[float], torso_lean: Optional[float],
                shoulder_tilt: Optional[float], hip_tilt: Optional[float],
                side: Optional[str], front_knee: Optional[float],
                back_knee: Optional[float], stability: Optional[float]) -> List[FormError]:
        found: List[FormError] = []

        # --- front knee. Loaded joint, so it outranks everything else. -----
        if front_knee is not None and side is not None:
            key = f"{side}_knee"
            if frame.status(key) is MetricStatus.MEASURED:
                confidence = frame.angles[key].confidence
                knee_lm = LM.LEFT_KNEE if side == "left" else LM.RIGHT_KNEE
                if front_knee < FRONT_KNEE_TARGET[0]:
                    found.append(FormError(
                        key="front_knee_deep",
                        message="Ease back — your front knee has travelled past your ankle.",
                        cue="KNEE BACK OVER ANKLE",
                        severity=Severity.MAJOR,
                        priority=Priority.SAFETY,
                        confidence=confidence,
                        landmarks=(knee_lm,), phase=self.phase,
                    ))
                elif front_knee > FRONT_KNEE_TARGET[1]:
                    found.append(FormError(
                        key="front_knee_shallow",
                        message="Bend your front knee more — work toward a right angle.",
                        cue="BEND FRONT KNEE",
                        severity=Severity.MINOR,
                        priority=Priority.RANGE,
                        confidence=confidence,
                        landmarks=(knee_lm,), phase=self.phase,
                    ))

        # --- arms sagging below shoulder height ---------------------------
        if arm_drop is not None and arm_drop > ARM_DROP_TOLERANCE:
            found.append(FormError(
                key="arms_dropping",
                message="Lift your arms back up to shoulder height.",
                cue="ARMS UP -> SHOULDER LINE",
                severity=Severity.MAJOR if arm_drop > 0.40 else Severity.MINOR,
                priority=Priority.ALIGNMENT,
                confidence=frame.pose.visibility_of(LM.LEFT_WRIST, LM.RIGHT_WRIST),
                landmarks=(LM.LEFT_WRIST, LM.RIGHT_WRIST), phase=self.phase,
            ))

        if arm_level is not None and arm_level > ARM_LEVEL_TOLERANCE:
            found.append(FormError(
                key="arms_not_level",
                message="Level your arms — reach equally in both directions.",
                cue="LEVEL YOUR ARMS",
                severity=Severity.MAJOR if arm_level > 25 else Severity.MINOR,
                priority=Priority.ALIGNMENT,
                confidence=frame.pose.visibility_of(LM.LEFT_WRIST, LM.RIGHT_WRIST),
                landmarks=(LM.LEFT_WRIST, LM.RIGHT_WRIST), phase=self.phase,
            ))

        if (torso_lean is not None and torso_lean > TORSO_TOLERANCE
                and tracked(frame.body_map, LM.LEFT_SHOULDER, LM.LEFT_HIP)):
            found.append(FormError(
                key="torso_lean",
                message="Stack your torso upright over your hips — don't lean toward the front leg.",
                cue="STRAIGHTEN TORSO",
                severity=Severity.MINOR,
                priority=Priority.ALIGNMENT,
                confidence=0.85,
                landmarks=(LM.LEFT_SHOULDER, LM.LEFT_HIP), phase=self.phase,
            ))

        if hip_tilt is not None and hip_tilt > HIP_TILT_LIMIT:
            found.append(FormError(
                key="hip_tilt",
                message="Level your hips — keep them square rather than dropping one side.",
                cue="LEVEL HIPS",
                severity=Severity.MINOR,
                priority=Priority.ALIGNMENT,
                confidence=frame.pose.visibility_of(LM.LEFT_HIP, LM.RIGHT_HIP),
                landmarks=(LM.LEFT_HIP, LM.RIGHT_HIP), phase=self.phase,
            ))

        if shoulder_tilt is not None and shoulder_tilt > SHOULDER_TILT_LIMIT:
            found.append(FormError(
                key="shoulder_tilt",
                message="Even out your shoulders — one is riding higher than the other.",
                cue="LEVEL SHOULDERS",
                severity=Severity.MINOR,
                priority=Priority.REFINEMENT,
                confidence=frame.pose.visibility_of(LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER),
                landmarks=(LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER), phase=self.phase,
            ))

        if back_knee is not None and back_knee < BACK_LEG_MIN:
            found.append(FormError(
                key="back_leg_bent",
                message="Straighten your back leg and press through that heel.",
                cue="STRAIGHTEN BACK LEG",
                severity=Severity.MINOR,
                priority=Priority.REFINEMENT,
                confidence=0.8,
                landmarks=(LM.RIGHT_KNEE if side == "left" else LM.LEFT_KNEE,),
                phase=self.phase,
            ))

        if stability is not None and stability < 45 and self.phase == "holding":
            found.append(FormError(
                key="unsteady",
                message="You're drifting — ground down through both feet and fix your gaze.",
                cue="STEADY",
                severity=Severity.MINOR,
                priority=Priority.STABILITY,
                confidence=0.8,
                landmarks=(LM.LEFT_HIP, LM.RIGHT_HIP), phase=self.phase,
            ))

        return found

    # ------------------------------------------------------------------
    def recognition_score(self, frame: EngineFrame) -> float:
        """Signature: arms wide and level, wide stance, one knee bent, both feet down."""
        if not self.can_analyse(frame):
            return 0.0

        arm_level = angle_to_horizontal(frame, LM.LEFT_WRIST, LM.RIGHT_WRIST)
        wrist_span = distance(frame, LM.LEFT_WRIST, LM.RIGHT_WRIST)
        shoulder_span = distance(frame, LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER)
        if (arm_level is None or wrist_span is None or shoulder_span is None
                or arm_level > 30 or wrist_span < shoulder_span * 1.8):
            return 0.0

        # Arms must be raised, not merely wide — hanging arms are wider than the
        # shoulders too, which would otherwise let any standing pose through.
        wrists = midpoint(frame, LM.LEFT_WRIST, LM.RIGHT_WRIST)
        shoulders = midpoint(frame, LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER)
        scale = torso_scale(frame)
        if wrists is None or shoulders is None or scale is None:
            return 0.0
        if abs(float(wrists[1] - shoulders[1])) > 0.45 * scale:
            return 0.0

        score = 0.45
        ankle_span = distance(frame, LM.LEFT_ANKLE, LM.RIGHT_ANKLE)
        hip_span = distance(frame, LM.LEFT_HIP, LM.RIGHT_HIP)
        if ankle_span and hip_span and ankle_span > hip_span * 1.6:
            score += 0.30

        left, right = frame.angle("left_knee"), frame.angle("right_knee")
        if left is not None and right is not None and abs(left - right) > 25:
            score += 0.25

        return min(1.0, score)
