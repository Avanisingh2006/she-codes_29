"""Frame sources.

MoveWise can be driven three ways, and the rest of the app does not care which:

  * CameraSource   — live webcam
  * VideoSource    — a recorded video file
  * SyntheticSource— scripted pose clips, no camera and no video needed

Every source yields a Frame. A synthetic source also supplies a ground-truth
PoseFrame, which lets the whole analysis stack be exercised on a machine with no
webcam at all — the case that matters when you zip this up and send it to someone.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import cv2

from . import config
from .landmarks import PoseFrame


@dataclass
class Frame:
    """One unit of input."""
    image: any                          # BGR numpy array
    pose: Optional[PoseFrame] = None    # ground truth, synthetic sources only
    index: int = 0
    total: Optional[int] = None         # None for live sources
    finished: bool = False

    @property
    def progress(self) -> Optional[float]:
        if not self.total:
            return None
        return min(1.0, self.index / self.total)


class FrameSource(ABC):
    """Common interface for anything that produces frames."""

    name: str = "source"
    is_live: bool = True
    needs_detection: bool = True   # False when the source supplies ground-truth pose

    @abstractmethod
    def open(self) -> bool: ...

    @abstractmethod
    def read(self) -> Optional[Frame]: ...

    def release(self) -> None:
        pass

    def __enter__(self) -> "FrameSource":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.release()


class CameraSource(FrameSource):
    """Live webcam."""

    is_live = True
    needs_detection = True

    def __init__(
        self,
        index: int = config.CAMERA_INDEX,
        mirror: bool = True
    ) -> None:
        self.index = index
        self.mirror = mirror
        self.name = f"Webcam {index}"
        self.cap: Optional[cv2.VideoCapture] = None
        self._count = 0

    def open(self) -> bool:
        # Release any previous capture first
        self.release()

        backends = []

        if os.name == "nt":
            # Try DirectShow first, then Media Foundation, then default
            backends = [
                cv2.CAP_DSHOW,
                cv2.CAP_MSMF,
                None,
            ]
        else:
            backends = [None]

        for backend in backends:
            try:
                if backend is None:
                    cap = cv2.VideoCapture(self.index)
                else:
                    cap = cv2.VideoCapture(self.index, backend)

                if cap is None or not cap.isOpened():
                    if cap is not None:
                        cap.release()
                    continue

                self.cap = cap

                # Camera properties are optional.
                # Some webcams/backends throw exceptions when setting them.
                try:
                    self.cap.set(
                        cv2.CAP_PROP_FRAME_WIDTH,
                        config.FRAME_WIDTH
                    )

                    self.cap.set(
                        cv2.CAP_PROP_FRAME_HEIGHT,
                        config.FRAME_HEIGHT
                    )

                except cv2.error as e:
                    print(
                        f"Camera opened, but requested resolution "
                        f"could not be applied: {e}"
                    )

                # Confirm that the camera can actually provide a frame
                try:
                    ok, _ = self.cap.read()
                    if not ok:
                        self.cap.release()
                        self.cap = None
                        continue
                except cv2.error as e:
                    print(f"Camera test read failed: {e}")
                    self.cap.release()
                    self.cap = None
                    continue

                print(
                    f"Webcam {self.index} opened successfully "
                    f"at {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                    f"{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
                )

                return True

            except cv2.error as e:
                print(f"OpenCV camera backend failed: {e}")

                if self.cap is not None:
                    self.cap.release()
                    self.cap = None

            except Exception as e:
                print(f"Camera backend failed: {e}")

                if self.cap is not None:
                    self.cap.release()
                    self.cap = None

        return False

    def read(self) -> Optional[Frame]:
        if self.cap is None or not self.cap.isOpened():
            return None

        try:
            ok, image = self.cap.read()
        except cv2.error as e:
            print(f"Camera read error: {e}")
            return None

        if not ok or image is None:
            return None

        if self.mirror:
            image = cv2.flip(image, 1)

        self._count += 1

        return Frame(
            image=image,
            index=self._count
        )

    def release(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            finally:
                self.cap = None

class VideoSource(FrameSource):
    """A recorded video file. Never mirrored — the footage is already correct."""

    is_live = False
    needs_detection = True

    def __init__(self, path: str, loop: bool = False, stride: int = 1) -> None:
        self.path = path
        self.loop = loop
        self.stride = max(1, stride)     # skip frames to keep long clips responsive
        self.name = os.path.basename(path)
        self.cap: Optional[cv2.VideoCapture] = None
        self.total = 0
        self.fps = 30.0
        self._count = 0

    def open(self) -> bool:
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            return False
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        return True

    def read(self) -> Optional[Frame]:
        if self.cap is None:
            return None

        image = None
        for _ in range(self.stride):
            ok, got = self.cap.read()
            if not ok:
                if self.loop:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self._count = 0
                    ok, got = self.cap.read()
                    if not ok:
                        return Frame(image=image, finished=True, index=self._count)
                else:
                    return Frame(image=image, finished=True,
                                 index=self._count, total=self.total)
            image = got
            self._count += 1

        if image is None:
            return Frame(image=None, finished=True, index=self._count, total=self.total)

        return Frame(image=image, index=self._count,
                     total=self.total if self.total else None)

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
