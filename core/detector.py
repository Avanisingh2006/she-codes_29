"""Pose detection and camera capture.

This is the only module that imports mediapipe. It supports both mediapipe
generations, because the legacy `mp.solutions.pose` API was removed in 0.10.3x:

  * Tasks API  (mediapipe >= 0.10.3x) — needs a .task model file, downloaded once.
  * Solutions API (older builds) — used automatically if Tasks is unavailable.

Either way the rest of MoveWise only ever sees a PoseFrame.
"""
from __future__ import annotations

import os
import time
import urllib.request
from pathlib import Path
from typing import Dict, Optional

import cv2
import mediapipe as mp

from . import config
from .landmarks import Landmark, PoseFrame

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "pose_landmarker_lite.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

# Which backend is available in this environment.
_HAS_SOLUTIONS = hasattr(mp, "solutions") and hasattr(getattr(mp, "solutions"), "pose")


def ensure_model(path: Path = MODEL_PATH, url: str = MODEL_URL) -> Path:
    """Download the pose model on first run. Cached afterwards."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size < 1_000_000:
        urllib.request.urlretrieve(url, path)
    return path


class _TasksBackend:
    """mediapipe.tasks PoseLandmarker, video mode."""

    def __init__(self, model_complexity: int, min_det: float, min_track: float) -> None:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        model = ensure_model()
        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_det,
            min_pose_presence_confidence=min_det,
            min_tracking_confidence=min_track,
            output_segmentation_masks=False,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._last_ms = -1

    def detect(self, rgb):
        # detect_for_video requires strictly increasing timestamps.
        ms = int(time.monotonic() * 1000)
        if ms <= self._last_ms:
            ms = self._last_ms + 1
        self._last_ms = ms

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, ms)

        if not result.pose_landmarks:
            return None, None
        world = result.pose_world_landmarks[0] if result.pose_world_landmarks else None
        return result.pose_landmarks[0], world

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:
            pass


class _SolutionsBackend:
    """Legacy mp.solutions.pose, kept so older environments still run."""

    def __init__(self, model_complexity: int, min_det: float, min_track: float) -> None:
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=False,
            smooth_landmarks=True,
            min_detection_confidence=min_det,
            min_tracking_confidence=min_track,
        )

    def detect(self, rgb):
        results = self._pose.process(rgb)
        if not results.pose_landmarks:
            return None, None
        world = results.pose_world_landmarks.landmark if results.pose_world_landmarks else None
        return results.pose_landmarks.landmark, world

    def close(self) -> None:
        try:
            self._pose.close()
        except Exception:
            pass


class PoseDetector:
    """Backend-agnostic pose detector. Emits PoseFrame objects."""

    def __init__(
        self,
        model_complexity: int = config.MODEL_COMPLEXITY,
        min_detection_confidence: float = config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = config.MIN_TRACKING_CONFIDENCE,
    ) -> None:
        if _HAS_SOLUTIONS:
            self.backend_name = "solutions"
            self._backend = _SolutionsBackend(
                model_complexity, min_detection_confidence, min_tracking_confidence)
        else:
            self.backend_name = "tasks"
            self._backend = _TasksBackend(
                model_complexity, min_detection_confidence, min_tracking_confidence)
        self._frame_index = 0

    def process(self, bgr_frame) -> PoseFrame:
        """Run detection on one BGR frame. Never raises on a bad or empty frame."""
        self._frame_index += 1
        now = time.monotonic()

        if bgr_frame is None or getattr(bgr_frame, "size", 0) == 0:
            return PoseFrame(frame_index=self._frame_index, timestamp=now, detected=False)

        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        try:
            normalized, world = self._backend.detect(rgb)
        except Exception:
            # A malformed frame or a transient backend error must never end the demo.
            return PoseFrame(frame_index=self._frame_index, timestamp=now, detected=False)

        if normalized is None:
            return PoseFrame(frame_index=self._frame_index, timestamp=now, detected=False)

        landmarks: Dict[int, Landmark] = {}
        for idx, lm in enumerate(normalized):
            visibility = float(getattr(lm, "visibility", 1.0) or 0.0)
            # Drop hopeless detections here; the engine decides what to do about gaps.
            if visibility < 0.1:
                continue
            w = world[idx] if world is not None and idx < len(world) else None
            landmarks[idx] = Landmark(
                idx=idx,
                x=float(lm.x), y=float(lm.y), z=float(lm.z),
                visibility=visibility,
                wx=float(w.x) if w is not None else 0.0,
                wy=float(w.y) if w is not None else 0.0,
                wz=float(w.z) if w is not None else 0.0,
            )

        return PoseFrame(
            landmarks=landmarks,
            frame_index=self._frame_index,
            timestamp=now,
            detected=bool(landmarks),
        )

    def close(self) -> None:
        self._backend.close()

    def __enter__(self) -> "PoseDetector":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class Camera:
    """cv2.VideoCapture with sane defaults and a safe read()."""

    def __init__(self, index: int = config.CAMERA_INDEX) -> None:
        self.index = index
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        if os.name == "nt":
            self.cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.index)
        else:
            self.cap = cv2.VideoCapture(self.index)

        if not self.cap.isOpened():
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        return True

    def read(self):
        if self.cap is None:
            return None
        ok, frame = self.cap.read()
        if not ok:
            return None
        return cv2.flip(frame, 1)  # mirror, so the user's left stays on their left

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
