"""PoseEngine — the single reusable pipeline every exercise sits on top of.

Camera frame in, EngineFrame out. The engine knows nothing about squats or yoga;
exercises know nothing about MediaPipe or smoothing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import config
from .bodymap import BodyMap, BodyMapCalibrator, BodyMode
from .detector import PoseDetector
from .geometry import joint_angle
from .landmarks import LM, MetricStatus, PoseFrame
from .smoothing import PoseSmoother

# Angles the engine computes for everyone, so profiles rarely need raw geometry.
COMMON_ANGLES: Dict[str, Tuple[LM, LM, LM]] = {
    "left_elbow":  (LM.LEFT_SHOULDER, LM.LEFT_ELBOW, LM.LEFT_WRIST),
    "right_elbow": (LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW, LM.RIGHT_WRIST),
    "left_shoulder":  (LM.LEFT_ELBOW, LM.LEFT_SHOULDER, LM.LEFT_HIP),
    "right_shoulder": (LM.RIGHT_ELBOW, LM.RIGHT_SHOULDER, LM.RIGHT_HIP),
    "left_hip":  (LM.LEFT_SHOULDER, LM.LEFT_HIP, LM.LEFT_KNEE),
    "right_hip": (LM.RIGHT_SHOULDER, LM.RIGHT_HIP, LM.RIGHT_KNEE),
    "left_knee":  (LM.LEFT_HIP, LM.LEFT_KNEE, LM.LEFT_ANKLE),
    "right_knee": (LM.RIGHT_HIP, LM.RIGHT_KNEE, LM.RIGHT_ANKLE),
}


@dataclass
class AngleReading:
    """An angle plus why you may or may not trust it."""
    name: str
    value: Optional[float]
    status: MetricStatus
    confidence: float = 0.0

    @property
    def usable(self) -> bool:
        return (
            self.status is MetricStatus.MEASURED
            and self.value is not None
            and self.confidence >= config.METRIC_SPEAK_THRESHOLD
        )


@dataclass
class EngineFrame:
    """Everything an exercise profile is allowed to see."""
    pose: PoseFrame
    angles: Dict[str, AngleReading] = field(default_factory=dict)
    body_map: Optional[BodyMap] = None
    calibrating: bool = False
    calibration_progress: float = 0.0

    @property
    def detected(self) -> bool:
        return self.pose.detected

    @property
    def timestamp(self) -> float:
        return self.pose.timestamp

    def angle(self, name: str) -> Optional[float]:
        """Angle value only if it is safe to act on. None otherwise."""
        reading = self.angles.get(name)
        return reading.value if reading and reading.usable else None

    def status(self, name: str) -> MetricStatus:
        reading = self.angles.get(name)
        return reading.status if reading else MetricStatus.UNMEASURED


class PoseEngine:
    """Detector + smoothing + body map + common angles, in one reusable object."""

    def __init__(self, auto_calibrate: bool = True) -> None:
        self.detector = PoseDetector()
        self.smoother = PoseSmoother()
        self.calibrator = BodyMapCalibrator()
        self.body_map: Optional[BodyMap] = None
        self._auto_calibrate = auto_calibrate

    # -- lifecycle ---------------------------------------------------------
    def reset(self) -> None:
        self.smoother.reset()
        self.calibrator.reset()
        self.body_map = None

    def close(self) -> None:
        self.detector.close()

    def force_body_map(self, body_map: BodyMap) -> None:
        self.body_map = body_map

    # -- main loop ---------------------------------------------------------
    def process(self, bgr_frame) -> EngineFrame:
        """Detect a pose in an image, then run the rest of the pipeline."""
        return self._run(self.detector.process(bgr_frame))

    def process_pose(self, pose: PoseFrame) -> EngineFrame:
        """Run the pipeline on a pose that was supplied rather than detected.

        Used by scripted clips, which bypass detection so the analysis stack can
        be exercised on a machine with no camera.
        """
        return self._run(pose)

    def _run(self, raw: PoseFrame) -> EngineFrame:
        smoothed = self.smoother.apply(raw)

        calibrating = False
        progress = 1.0

        if self._auto_calibrate and self.body_map is None:
            self.calibrator.observe(smoothed)
            progress = self.calibrator.progress(smoothed.timestamp)
            if self.calibrator.is_complete(smoothed.timestamp):
                self.body_map = self.calibrator.build()
                calibrating = False
            else:
                calibrating = True

        angles = self._compute_angles(smoothed)

        return EngineFrame(
            pose=smoothed,
            angles=angles,
            body_map=self.body_map,
            calibrating=calibrating,
            calibration_progress=progress,
        )

    # -- internals ---------------------------------------------------------
    def _compute_angles(self, frame: PoseFrame) -> Dict[str, AngleReading]:
        readings: Dict[str, AngleReading] = {}

        for name, (a, b, c) in COMMON_ANGLES.items():
            # 1. Does this user's body map even include these joints?
            if self.body_map is not None and not self.body_map.all_trackable(a, b, c):
                readings[name] = AngleReading(name, None, MetricStatus.NOT_APPLICABLE, 0.0)
                continue

            # 2. Are they present and confident in this particular frame?
            confidence = frame.visibility_of(a, b, c)
            if confidence < config.VISIBILITY_THRESHOLD:
                readings[name] = AngleReading(name, None, MetricStatus.UNMEASURED, confidence)
                continue

            value = joint_angle(frame, a, b, c, use_world=True)
            if value is None:
                readings[name] = AngleReading(name, None, MetricStatus.UNMEASURED, confidence)
                continue

            readings[name] = AngleReading(name, value, MetricStatus.MEASURED, confidence)

        return readings
