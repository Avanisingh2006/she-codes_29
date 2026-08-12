"""Angle and alignment maths. Every function returns None rather than raising
when the inputs are missing — callers translate that into MetricStatus.UNMEASURED.

Every helper accepts either a PoseFrame or an EngineFrame, so exercise profiles
can pass whichever they are holding without unwrapping first.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .landmarks import LM, PoseFrame


def _pose(frame: Any) -> PoseFrame:
    """Accept a PoseFrame or an EngineFrame and return the PoseFrame."""
    inner = getattr(frame, "pose", None)
    return inner if isinstance(inner, PoseFrame) else frame


def _angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Optional[float]:
    """Interior angle at b, in degrees."""
    ba = a - b
    bc = c - b
    nba = np.linalg.norm(ba)
    nbc = np.linalg.norm(bc)
    if nba < 1e-6 or nbc < 1e-6:
        return None
    cosine = float(np.dot(ba, bc) / (nba * nbc))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def joint_angle(frame: Any, a: LM, b: LM, c: LM, use_world: bool = True) -> Optional[float]:
    """Angle at joint b formed by a-b-c.

    Uses metric world coordinates by default, which keeps the reading stable
    when the user rotates relative to the camera.
    """
    frame = _pose(frame)
    la, lb, lc = frame.get(a), frame.get(b), frame.get(c)
    if la is None or lb is None or lc is None:
        return None
    if use_world:
        return _angle_between(la.world, lb.world, lc.world)
    return _angle_between(la.px, lb.px, lc.px)


def segment_vector(frame: Any, a: LM, b: LM, use_world: bool = True) -> Optional[np.ndarray]:
    frame = _pose(frame)
    la, lb = frame.get(a), frame.get(b)
    if la is None or lb is None:
        return None
    return (lb.world - la.world) if use_world else (lb.px - la.px)


def angle_to_vertical(frame: Any, top: LM, bottom: LM) -> Optional[float]:
    """How far a body segment leans off vertical, in degrees. 0 = upright."""
    vec = segment_vector(frame, top, bottom, use_world=True)
    if vec is None:
        return None
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        return None
    # MediaPipe world space: +y points down.
    vertical = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    cosine = float(np.dot(vec / norm, vertical))
    return float(np.degrees(np.arccos(np.clip(abs(cosine), -1.0, 1.0))))


def angle_to_horizontal(frame: Any, a: LM, b: LM) -> Optional[float]:
    """How far a segment deviates from level, in degrees. 0 = perfectly horizontal."""
    vec = segment_vector(frame, a, b, use_world=True)
    if vec is None:
        return None
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        return None
    vertical = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    cosine = float(np.dot(vec / norm, vertical))
    # angle from vertical, then complement gives deviation from horizontal
    from_vertical = float(np.degrees(np.arccos(np.clip(abs(cosine), -1.0, 1.0))))
    return abs(90.0 - from_vertical)


def distance(frame: Any, a: LM, b: LM, use_world: bool = True) -> Optional[float]:
    vec = segment_vector(frame, a, b, use_world)
    if vec is None:
        return None
    return float(np.linalg.norm(vec))


def midpoint(frame: Any, a: LM, b: LM, use_world: bool = True) -> Optional[np.ndarray]:
    frame = _pose(frame)
    la, lb = frame.get(a), frame.get(b)
    if la is None or lb is None:
        return None
    return (la.world + lb.world) / 2.0 if use_world else (la.px + lb.px) / 2.0


def torso_scale(frame: Any) -> Optional[float]:
    """Shoulder-to-hip distance. Used to express offsets in body-relative units
    so the same rule works for a tall and a short user."""
    frame = _pose(frame)
    sh = midpoint(frame, LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER)
    hp = midpoint(frame, LM.LEFT_HIP, LM.RIGHT_HIP)
    if sh is None or hp is None:
        return None
    scale = float(np.linalg.norm(sh - hp))
    return scale if scale > 1e-3 else None


def horizontal_offset(frame: Any, a: LM, b: LM) -> Optional[float]:
    """Signed left/right offset between two joints, normalized by torso length.

    Positive means a is to the subject's right of b. Used for knee-tracking rules.
    """
    frame = _pose(frame)
    la, lb = frame.get(a), frame.get(b)
    scale = torso_scale(frame)
    if la is None or lb is None or scale is None:
        return None
    return float((la.wx - lb.wx) / scale)
