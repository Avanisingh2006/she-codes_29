"""Ghost Coach — the reference overlay.

The reference is a **visual movement guide**, not a claim that one posture is
universally correct. It is fitted to the user: scaled to their torso length and
anchored on their own hips, so a tall user and a short user each see a guide
built to their proportions rather than a stock skeleton they are told to match.

For rep-based movements the ghost follows the user through the rep, so at the
bottom of a squat it shows the bottom of a squat — comparing a descending user
against a standing reference would be meaningless.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np

from .landmarks import LM, PoseFrame
from .synthetic import (CURL_UP, SQUAT_BOTTOM, STANDING, TREE, WARRIOR_2,
                        Joints, _lerp)

Point = Tuple[int, int]

# Which joints the ghost draws. Face landmarks are noise at this size.
GHOST_BONES = [
    (LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER),
    (LM.LEFT_SHOULDER, LM.LEFT_ELBOW), (LM.LEFT_ELBOW, LM.LEFT_WRIST),
    (LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW), (LM.RIGHT_ELBOW, LM.RIGHT_WRIST),
    (LM.LEFT_SHOULDER, LM.LEFT_HIP), (LM.RIGHT_SHOULDER, LM.RIGHT_HIP),
    (LM.LEFT_HIP, LM.RIGHT_HIP),
    (LM.LEFT_HIP, LM.LEFT_KNEE), (LM.LEFT_KNEE, LM.LEFT_ANKLE),
    (LM.RIGHT_HIP, LM.RIGHT_KNEE), (LM.RIGHT_KNEE, LM.RIGHT_ANKLE),
]


def _squat_reference(progress: float) -> Joints:
    return _lerp(STANDING, SQUAT_BOTTOM, progress)


def _curl_reference(progress: float) -> Joints:
    return _lerp(STANDING, CURL_UP, progress)


def _tree_reference(_progress: float) -> Joints:
    return TREE


def _warrior_reference(_progress: float) -> Joints:
    return WARRIOR_2


REFERENCES: Dict[str, Callable[[float], Joints]] = {
    "squat": _squat_reference,
    "bicep_curl": _curl_reference,
    "tree_pose": _tree_reference,
    "warrior_2": _warrior_reference,
}


def _mirror(joints: Joints) -> Joints:
    """Flip the reference left-for-right, for poses that have a lead side."""
    swap = {
        LM.LEFT_SHOULDER: LM.RIGHT_SHOULDER, LM.LEFT_ELBOW: LM.RIGHT_ELBOW,
        LM.LEFT_WRIST: LM.RIGHT_WRIST, LM.LEFT_HIP: LM.RIGHT_HIP,
        LM.LEFT_KNEE: LM.RIGHT_KNEE, LM.LEFT_ANKLE: LM.RIGHT_ANKLE,
        LM.LEFT_HEEL: LM.RIGHT_HEEL, LM.LEFT_FOOT_INDEX: LM.RIGHT_FOOT_INDEX,
        LM.LEFT_EYE: LM.RIGHT_EYE, LM.LEFT_EAR: LM.RIGHT_EAR,
    }
    pairs = dict(swap)
    pairs.update({v: k for k, v in swap.items()})

    out: Joints = {}
    for lm, (x, y, z) in joints.items():
        target = pairs.get(lm, lm)
        out[target] = (-x, y, z)
    return out


def _pixel(lm, shape) -> Optional[np.ndarray]:
    if lm is None:
        return None
    h, w = shape[:2]
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)


def fit_reference(
    exercise_key: str,
    pose: PoseFrame,
    image_shape,
    progress: float = 0.0,
    mirror: bool = False,
) -> Optional[Dict[LM, Point]]:
    """Place the reference skeleton onto the user, in image pixels.

    Returns None when there is not enough of the user visible to anchor it —
    a ghost floating in the wrong place is worse than no ghost.
    """
    builder = REFERENCES.get(exercise_key)
    if builder is None:
        return None

    joints = builder(max(0.0, min(1.0, progress)))
    if mirror:
        joints = _mirror(joints)

    # Anchor and scale come from the user, so the guide matches their build.
    ls, rs = pose.get(LM.LEFT_SHOULDER), pose.get(LM.RIGHT_SHOULDER)
    lh, rh = pose.get(LM.LEFT_HIP), pose.get(LM.RIGHT_HIP)
    if not all((ls, rs, lh, rh)):
        return None

    shoulder_px = (_pixel(ls, image_shape) + _pixel(rs, image_shape)) / 2.0
    hip_px = (_pixel(lh, image_shape) + _pixel(rh, image_shape)) / 2.0
    user_torso_px = float(np.linalg.norm(shoulder_px - hip_px))
    if user_torso_px < 20.0:
        return None

    ref_shoulder = np.array([
        (joints[LM.LEFT_SHOULDER][0] + joints[LM.RIGHT_SHOULDER][0]) / 2.0,
        (joints[LM.LEFT_SHOULDER][1] + joints[LM.RIGHT_SHOULDER][1]) / 2.0,
    ], dtype=np.float32)
    ref_hip = np.array([
        (joints[LM.LEFT_HIP][0] + joints[LM.RIGHT_HIP][0]) / 2.0,
        (joints[LM.LEFT_HIP][1] + joints[LM.RIGHT_HIP][1]) / 2.0,
    ], dtype=np.float32)
    ref_torso = float(np.linalg.norm(ref_shoulder - ref_hip))
    if ref_torso < 1e-4:
        return None

    scale = user_torso_px / ref_torso

    out: Dict[LM, Point] = {}
    for lm, (x, y, _z) in joints.items():
        offset = (np.array([x, y], dtype=np.float32) - ref_hip) * scale
        px = hip_px + offset
        out[lm] = (int(px[0]), int(px[1]))
    return out


def correction_arrow(
    pose: PoseFrame,
    ghost: Dict[LM, Point],
    landmark: LM,
    image_shape,
    min_pixels: float = 18.0,
) -> Optional[Tuple[Point, Point]]:
    """Arrow from where a joint is to where the guide puts it.

    Derived from the ghost rather than hand-written per error, so any joint the
    reference covers gets a correct arrow for free. Returns None when the joint
    is already close enough that an arrow would be visual noise.
    """
    user = pose.get(landmark)
    if user is None or landmark not in ghost:
        return None

    start = _pixel(user, image_shape)
    end = np.array(ghost[landmark], dtype=np.float32)
    if float(np.linalg.norm(end - start)) < min_pixels:
        return None
    return (int(start[0]), int(start[1])), (int(end[0]), int(end[1]))


def mean_deviation(pose: PoseFrame, ghost: Dict[LM, Point], image_shape) -> Optional[float]:
    """Average joint offset from the guide, as a fraction of torso length.

    A single number for "how close am I to the reference", used only as a
    display hint — the per-joint rules, not this, decide what is actually wrong.
    """
    ls, rs = pose.get(LM.LEFT_SHOULDER), pose.get(LM.RIGHT_SHOULDER)
    lh, rh = pose.get(LM.LEFT_HIP), pose.get(LM.RIGHT_HIP)
    if not all((ls, rs, lh, rh)):
        return None

    shoulder_px = (_pixel(ls, image_shape) + _pixel(rs, image_shape)) / 2.0
    hip_px = (_pixel(lh, image_shape) + _pixel(rh, image_shape)) / 2.0
    torso = float(np.linalg.norm(shoulder_px - hip_px))
    if torso < 20.0:
        return None

    offsets = []
    for lm in (LM.LEFT_ELBOW, LM.RIGHT_ELBOW, LM.LEFT_WRIST, LM.RIGHT_WRIST,
               LM.LEFT_KNEE, LM.RIGHT_KNEE, LM.LEFT_ANKLE, LM.RIGHT_ANKLE):
        user = pose.get(lm)
        if user is None or lm not in ghost:
            continue
        offsets.append(float(np.linalg.norm(
            _pixel(user, image_shape) - np.array(ghost[lm], dtype=np.float32))))

    if not offsets:
        return None
    return float(np.mean(offsets) / torso)


# Mean deviation (torso-lengths) below which the user counts as on the guide,
# and the looser band where they are nearly there. Above both they are "off".
ALIGNED_BELOW = 0.16
CLOSE_BELOW = 0.32


def alignment_status(pose: PoseFrame, ghost: Dict[LM, Point], image_shape) -> Optional[str]:
    """Coarse "am I on the guide?" label for the UI: aligned / close / off.

    A display hint layered on mean_deviation — it never drives coaching, it just
    lets the overlay glow subtly when the user matches the ghost. Returns None
    when the deviation cannot be measured (torso not visible, no shared joints).
    """
    dev = mean_deviation(pose, ghost, image_shape)
    if dev is None:
        return None
    if dev < ALIGNED_BELOW:
        return "aligned"
    if dev < CLOSE_BELOW:
        return "close"
    return "off"
