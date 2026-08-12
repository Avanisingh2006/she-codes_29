"""Scripted pose clips — a camera-free way to exercise the whole analysis stack.

Each clip interpolates between hand-authored keyframes and yields both a rendered
figure and the ground-truth PoseFrame behind it, so detection is bypassed entirely.
That makes these clips deterministic: the same clip produces the same metrics,
deviations and rep counts on every machine, with no webcam and no network.

They are a test and demo fixture, not a substitute for real footage — the pose is
given rather than detected, so a clip proves the analysis is right, not the
detector. Use webcam or an uploaded video to exercise detection.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .landmarks import LM, Landmark, PoseFrame
from .source import Frame, FrameSource

Joints = Dict[LM, Tuple[float, float, float]]

# ---------------------------------------------------------------------------
# Keyframes, in metres. Origin near the hips, +y points down, +z toward camera.
# ---------------------------------------------------------------------------
STANDING: Joints = {
    LM.NOSE: (0.00, -0.72, 0.05),
    LM.LEFT_EYE: (-0.03, -0.74, 0.04), LM.RIGHT_EYE: (0.03, -0.74, 0.04),
    LM.LEFT_EAR: (-0.07, -0.72, 0.00), LM.RIGHT_EAR: (0.07, -0.72, 0.00),
    LM.LEFT_SHOULDER: (-0.20, -0.50, 0.00), LM.RIGHT_SHOULDER: (0.20, -0.50, 0.00),
    LM.LEFT_ELBOW: (-0.22, -0.20, 0.00), LM.RIGHT_ELBOW: (0.22, -0.20, 0.00),
    LM.LEFT_WRIST: (-0.24, 0.10, 0.00), LM.RIGHT_WRIST: (0.24, 0.10, 0.00),
    LM.LEFT_HIP: (-0.15, 0.00, 0.00), LM.RIGHT_HIP: (0.15, 0.00, 0.00),
    LM.LEFT_KNEE: (-0.15, 0.45, 0.00), LM.RIGHT_KNEE: (0.15, 0.45, 0.00),
    LM.LEFT_ANKLE: (-0.15, 0.90, 0.00), LM.RIGHT_ANKLE: (0.15, 0.90, 0.00),
    LM.LEFT_HEEL: (-0.15, 0.94, -0.04), LM.RIGHT_HEEL: (0.15, 0.94, -0.04),
    LM.LEFT_FOOT_INDEX: (-0.15, 0.93, 0.12), LM.RIGHT_FOOT_INDEX: (0.15, 0.93, 0.12),
}


def _merge(base: Joints, **changes) -> Joints:
    out = dict(base)
    for name, value in changes.items():
        out[LM[name]] = value
    return out


SQUAT_BOTTOM: Joints = _merge(
    STANDING,
    LEFT_SHOULDER=(-0.20, -0.12, 0.10), RIGHT_SHOULDER=(0.20, -0.12, 0.10),
    LEFT_ELBOW=(-0.24, 0.14, 0.16), RIGHT_ELBOW=(0.24, 0.14, 0.16),
    LEFT_WRIST=(-0.26, 0.36, 0.30), RIGHT_WRIST=(0.26, 0.36, 0.30),
    LEFT_HIP=(-0.16, 0.40, -0.12), RIGHT_HIP=(0.16, 0.40, -0.12),
    LEFT_KNEE=(-0.23, 0.60, 0.20), RIGHT_KNEE=(0.23, 0.60, 0.20),
)

# Same depth, but the knees fall inside the ankle line — the classic valgus fault.
SQUAT_BOTTOM_VALGUS: Joints = _merge(
    SQUAT_BOTTOM,
    LEFT_KNEE=(-0.02, 0.60, 0.20), RIGHT_KNEE=(0.02, 0.60, 0.20),
)

CURL_UP: Joints = _merge(
    STANDING,
    LEFT_WRIST=(-0.26, -0.45, 0.10), RIGHT_WRIST=(0.26, -0.45, 0.10),
)

TREE: Joints = _merge(
    STANDING,
    LEFT_ELBOW=(-0.18, -0.34, 0.12), RIGHT_ELBOW=(0.18, -0.34, 0.12),
    LEFT_WRIST=(-0.06, -0.30, 0.18), RIGHT_WRIST=(0.06, -0.30, 0.18),
    RIGHT_KNEE=(0.32, 0.50, 0.06),
    RIGHT_ANKLE=(0.02, 0.42, 0.02),
    RIGHT_HEEL=(0.00, 0.44, -0.01), RIGHT_FOOT_INDEX=(-0.06, 0.40, 0.06),
)

WARRIOR_2: Joints = _merge(
    STANDING,
    LEFT_ELBOW=(-0.52, -0.50, 0.00), RIGHT_ELBOW=(0.52, -0.50, 0.00),
    LEFT_WRIST=(-0.85, -0.50, 0.00), RIGHT_WRIST=(0.85, -0.50, 0.00),
    LEFT_HIP=(-0.15, 0.15, 0.00), RIGHT_HIP=(0.15, 0.15, 0.00),
    LEFT_KNEE=(-0.55, 0.25, 0.00), LEFT_ANKLE=(-0.55, 0.90, 0.00),
    LEFT_HEEL=(-0.55, 0.94, -0.04), LEFT_FOOT_INDEX=(-0.60, 0.93, 0.10),
    RIGHT_KNEE=(0.45, 0.52, 0.00), RIGHT_ANKLE=(0.75, 0.90, 0.00),
    RIGHT_HEEL=(0.78, 0.94, -0.04), RIGHT_FOOT_INDEX=(0.82, 0.93, 0.08),
)

# Arms sagging well below shoulder height — Warrior II's most common fault.
WARRIOR_2_DROOP: Joints = _merge(
    WARRIOR_2,
    LEFT_ELBOW=(-0.50, -0.34, 0.00), RIGHT_ELBOW=(0.50, -0.34, 0.00),
    LEFT_WRIST=(-0.80, -0.14, 0.00), RIGHT_WRIST=(0.80, -0.18, 0.00),
)


def _lerp(a: Joints, b: Joints, t: float) -> Joints:
    t = max(0.0, min(1.0, t))
    out: Joints = {}
    for lm, (ax, ay, az) in a.items():
        bx, by, bz = b.get(lm, (ax, ay, az))
        out[lm] = (ax + (bx - ax) * t, ay + (by - ay) * t, az + (bz - az) * t)
    return out


def _sway(joints: Joints, amount: float, phase: float) -> Joints:
    """Add a gentle whole-body drift, so static holds are not perfectly frozen."""
    dx = amount * math.sin(phase)
    dy = amount * 0.35 * math.sin(phase * 1.7)
    return {lm: (x + dx, y + dy, z) for lm, (x, y, z) in joints.items()}


# ---------------------------------------------------------------------------
# Clip definitions
# ---------------------------------------------------------------------------
@dataclass
class Clip:
    key: str
    title: str
    exercise_key: str
    seconds: float
    caption: str
    pose_at: Callable[[float], Joints]   # t in seconds -> joints


def _squat_pose(faulty: bool) -> Callable[[float], Joints]:
    bottom = SQUAT_BOTTOM_VALGUS if faulty else SQUAT_BOTTOM
    period = 3.0

    def at(t: float) -> Joints:
        # 0 at the top, 1 at the bottom, smooth in both directions.
        depth = (1.0 - math.cos(2.0 * math.pi * t / period)) / 2.0
        return _lerp(STANDING, bottom, depth)

    return at


def _squat_stuck_pose() -> Callable[[float], Joints]:
    """Descends into a valgus bottom and stays there.

    The rep-cycle clips clear their fault at the top of every rep, so the coach
    resolves it before it can escalate. A held fault is what actually exercises
    the escalation ladder, the comfort check and the variation offer.
    """
    def at(t: float) -> Joints:
        entry = min(1.0, t / 1.5)
        return _lerp(STANDING, SQUAT_BOTTOM_VALGUS, entry)

    return at


def _squat_fast_pose() -> Callable[[float], Joints]:
    """The clean squat shape bounced violently fast — a control fault, not a form one.

    Same keyframes as the good clip, but at ~0.9 s per rep: every joint stays on
    its correct path, only the tempo is wrong. This is what exercises the
    stability & controlled-movement coach rather than the geometry rules.
    """
    period = 0.9

    def at(t: float) -> Joints:
        depth = (1.0 - math.cos(2.0 * math.pi * t / period)) / 2.0
        return _lerp(STANDING, SQUAT_BOTTOM, depth)

    return at


def _curl_pose() -> Callable[[float], Joints]:
    period = 2.4

    def at(t: float) -> Joints:
        flex = (1.0 - math.cos(2.0 * math.pi * t / period)) / 2.0
        return _lerp(STANDING, CURL_UP, flex)

    return at


def _tree_pose() -> Callable[[float], Joints]:
    def at(t: float) -> Joints:
        entry = min(1.0, t / 1.5)                 # rise into the pose
        joints = _lerp(STANDING, TREE, entry)
        # Wobble grows as the hold goes on, which is what a real balance does.
        wobble = 0.012 + 0.010 * max(0.0, t - 4.0)
        return _sway(joints, min(wobble, 0.05), t * 3.2)

    return at


def _warrior_pose() -> Callable[[float], Joints]:
    def at(t: float) -> Joints:
        entry = min(1.0, t / 1.5)
        joints = _lerp(STANDING, WARRIOR_2, entry)
        # After 5 s the arms start to sag — a fault the coach should catch.
        if t > 5.0:
            droop = min(1.0, (t - 5.0) / 3.0)
            joints = _lerp(joints, WARRIOR_2_DROOP, droop)
        return _sway(joints, 0.006, t * 2.0)

    return at


CLIPS: List[Clip] = [
    Clip("squat_good", "Squat — clean form", "squat", 9.0,
         "Three controlled reps. Depth reached, knees tracking over the feet.",
         _squat_pose(faulty=False)),
    Clip("squat_valgus", "Squat — knees caving in", "squat", 9.0,
         "Same depth, but both knees collapse inward at the bottom of every rep.",
         _squat_pose(faulty=True)),
    Clip("squat_stuck", "Squat - held fault (coaching demo)", "squat", 26.0,
         "Holds a knees-caving bottom position so the coaching ladder escalates: "
         "notice, instruct, focus, then the comfort check and an easier variation.",
         _squat_stuck_pose()),
    Clip("squat_fast", "Squat — rushed and bouncy (control coach demo)", "squat", 10.0,
         "The same clean squat shape bounced at roughly 0.9 s per rep. Demos the "
         "stability & controlled-movement coach: expect SLOW DOWN / SMOOTH IT OUT "
         "cues and a low movement-control score, with form geometry still clean.",
         _squat_fast_pose()),
    Clip("curl", "Bicep curl — four reps", "bicep_curl", 10.0,
         "Full range curls with both arms, elbows staying at the sides.",
         _curl_pose()),
    Clip("tree", "Tree pose — balance hold", "tree_pose", 10.0,
         "Rises into the pose and holds, with sway increasing as the hold goes on.",
         _tree_pose()),
    Clip("warrior", "Warrior II — arms drop", "warrior_2", 10.0,
         "Settles into the pose, then the arms sag below shoulder height after 5 s.",
         _warrior_pose()),
]

CLIPS_BY_KEY = {c.key: c for c in CLIPS}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
_BONES = [
    (LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER),
    (LM.LEFT_SHOULDER, LM.LEFT_ELBOW), (LM.LEFT_ELBOW, LM.LEFT_WRIST),
    (LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW), (LM.RIGHT_ELBOW, LM.RIGHT_WRIST),
    (LM.LEFT_SHOULDER, LM.LEFT_HIP), (LM.RIGHT_SHOULDER, LM.RIGHT_HIP),
    (LM.LEFT_HIP, LM.RIGHT_HIP),
    (LM.LEFT_HIP, LM.LEFT_KNEE), (LM.LEFT_KNEE, LM.LEFT_ANKLE),
    (LM.RIGHT_HIP, LM.RIGHT_KNEE), (LM.RIGHT_KNEE, LM.RIGHT_ANKLE),
    (LM.LEFT_ANKLE, LM.LEFT_FOOT_INDEX), (LM.RIGHT_ANKLE, LM.RIGHT_FOOT_INDEX),
]

_W, _H = 720, 720
_SCALE = 330.0          # pixels per metre
_CX, _CY = _W / 2, _H / 2 - 40


def _to_pixel(x: float, y: float) -> Tuple[int, int]:
    return int(_CX + x * _SCALE), int(_CY + y * _SCALE)


def render(joints: Joints, title: str = "") -> np.ndarray:
    """Draw the scripted figure so there is something to look at in the video pane."""
    img = np.full((_H, _W, 3), 26, dtype=np.uint8)

    # Floor line, for a sense of ground contact.
    floor_y = _to_pixel(0, 0.94)[1]
    cv2.line(img, (0, floor_y), (_W, floor_y), (52, 52, 52), 2)

    for a, b in _BONES:
        if a not in joints or b not in joints:
            continue
        cv2.line(img, _to_pixel(*joints[a][:2]), _to_pixel(*joints[b][:2]),
                 (168, 168, 168), 6, cv2.LINE_AA)

    # Head
    if LM.NOSE in joints:
        cv2.circle(img, _to_pixel(*joints[LM.NOSE][:2]), 34, (168, 168, 168), -1, cv2.LINE_AA)

    for lm, (x, y, _z) in joints.items():
        if lm in (LM.LEFT_EYE, LM.RIGHT_EYE, LM.LEFT_EAR, LM.RIGHT_EAR, LM.NOSE):
            continue
        cv2.circle(img, _to_pixel(x, y), 7, (235, 235, 235), -1, cv2.LINE_AA)

    if title:
        cv2.putText(img, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(img, "SIMULATED POSE - detection bypassed", (20, _H - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (120, 120, 120), 1, cv2.LINE_AA)
    return img


def to_pose_frame(joints: Joints, index: int, timestamp: float) -> PoseFrame:
    """Ground-truth PoseFrame. Visibility is 1.0 — these landmarks are given, not inferred."""
    landmarks: Dict[int, Landmark] = {}
    for lm, (x, y, z) in joints.items():
        px, py = _to_pixel(x, y)
        landmarks[int(lm)] = Landmark(
            idx=int(lm),
            x=px / _W, y=py / _H, z=z,
            visibility=1.0,
            wx=x, wy=y, wz=z,
        )
    return PoseFrame(landmarks=landmarks, frame_index=index,
                     timestamp=timestamp, detected=True)


# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------
class SyntheticSource(FrameSource):
    """Replays a scripted clip. No camera, no video file, no network."""

    is_live = False
    needs_detection = False

    def __init__(self, clip: Clip, fps: float = 24.0, loop: bool = True) -> None:
        self.clip = clip
        self.fps = fps
        self.loop = loop
        self.name = clip.title
        self.total = int(clip.seconds * fps)
        self._i = 0

    def open(self) -> bool:
        self._i = 0
        return True

    def read(self) -> Optional[Frame]:
        if self._i >= self.total:
            if not self.loop:
                return Frame(image=None, finished=True,
                             index=self._i, total=self.total)
            self._i = 0

        t = self._i / self.fps
        joints = self.clip.pose_at(t)
        image = render(joints, f"{self.clip.title}   t={t:4.1f}s")
        pose = to_pose_frame(joints, self._i, t)
        self._i += 1
        return Frame(image=image, pose=pose, index=self._i, total=self.total)


def export_video(clip: Clip, path: str, fps: float = 24.0) -> str:
    """Write a clip out as an .mp4, for sharing or for testing the video path."""
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (_W, _H))
    total = int(clip.seconds * fps)
    for i in range(total):
        writer.write(render(clip.pose_at(i / fps), clip.title))
    writer.release()
    return path
