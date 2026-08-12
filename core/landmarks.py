"""Landmark identity, per-joint status, and the frame container the engine passes around."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Dict, Iterable, Optional

import numpy as np


class LM(IntEnum):
    """MediaPipe Pose landmark indices, named for the ones we actually use."""
    NOSE = 0
    LEFT_EYE = 2
    RIGHT_EYE = 5
    LEFT_EAR = 7
    RIGHT_EAR = 8
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


class MetricStatus(str, Enum):
    """Tri-state result for every measurement.

    A metric that could not be seen is never reported as correct. This is the
    contract the whole coaching layer is built on.
    """
    MEASURED = "measured"           # observed with usable confidence
    UNMEASURED = "unmeasured"       # occluded, low confidence, or bad viewpoint
    NOT_APPLICABLE = "n/a"          # body map says this joint is not trackable for this user


@dataclass(frozen=True)
class Landmark:
    """A single joint in one frame."""
    idx: int
    x: float                 # normalized image coords (0..1)
    y: float
    z: float                 # normalized depth, relative to hips
    visibility: float        # MediaPipe's own confidence, 0..1
    wx: float = 0.0          # world coords in metres, hip-centred
    wy: float = 0.0
    wz: float = 0.0

    @property
    def px(self) -> np.ndarray:
        """2D image-space point."""
        return np.array([self.x, self.y], dtype=np.float32)

    @property
    def world(self) -> np.ndarray:
        """3D metric point. Preferred for angles — it is viewpoint-robust."""
        return np.array([self.wx, self.wy, self.wz], dtype=np.float32)


@dataclass
class PoseFrame:
    """One frame of pose, after smoothing. Landmarks may be missing — that is normal."""
    landmarks: Dict[int, Landmark] = field(default_factory=dict)
    frame_index: int = 0
    timestamp: float = 0.0
    detected: bool = False

    def get(self, lm: LM) -> Optional[Landmark]:
        return self.landmarks.get(int(lm))

    def has(self, *lms: LM, min_visibility: float = 0.0) -> bool:
        """True only if every requested landmark is present and confident enough."""
        for lm in lms:
            got = self.landmarks.get(int(lm))
            if got is None or got.visibility < min_visibility:
                return False
        return True

    def visibility_of(self, *lms: LM) -> float:
        """Weakest link confidence across the requested landmarks."""
        vals = [self.landmarks[int(l)].visibility for l in lms if int(l) in self.landmarks]
        if len(vals) != len(tuple(lms)):
            return 0.0
        return float(min(vals))

    def present(self) -> Iterable[LM]:
        for idx in self.landmarks:
            try:
                yield LM(idx)
            except ValueError:
                continue
