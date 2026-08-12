"""Human-readable body groups.

Landmark indices are the wrong vocabulary for a person deciding whether the
system can see them. Calibration talks about arms, legs and hips; this module is
the translation layer, and the single place that defines which landmarks make up
each part of the body.

Everything here is None-safe on the body map: before calibration there is no map
yet, and "no map" means "no restriction" rather than "nothing works". That keeps
the guards in the exercise profiles to a single readable call.
"""
from __future__ import annotations

from typing import (TYPE_CHECKING, Iterable, List, Mapping, Optional, Sequence,
                    Tuple)

from .landmarks import LM, MetricStatus

if TYPE_CHECKING:                      # pragma: no cover - typing only
    from .bodymap import BodyMap
    from .scoring import Metric


# --- group names, used verbatim in the UI ----------------------------------
HEAD = "Head"
SHOULDERS = "Shoulders"
LEFT_ARM = "Left arm"
RIGHT_ARM = "Right arm"
TORSO = "Torso"
HIPS = "Hips"
LEFT_LEG = "Left leg"
RIGHT_LEG = "Right leg"


# A group is available only when every landmark in it is trackable, so each list
# is deliberately the minimum that makes the group mean something.
GROUPS: List[Tuple[str, Tuple[LM, ...]]] = [
    (HEAD,      (LM.NOSE,)),
    (SHOULDERS, (LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER)),
    (LEFT_ARM,  (LM.LEFT_SHOULDER, LM.LEFT_ELBOW, LM.LEFT_WRIST)),
    (RIGHT_ARM, (LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW, LM.RIGHT_WRIST)),
    (TORSO,     (LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER, LM.LEFT_HIP, LM.RIGHT_HIP)),
    (HIPS,      (LM.LEFT_HIP, LM.RIGHT_HIP)),
    (LEFT_LEG,  (LM.LEFT_HIP, LM.LEFT_KNEE, LM.LEFT_ANKLE)),
    (RIGHT_LEG, (LM.RIGHT_HIP, LM.RIGHT_KNEE, LM.RIGHT_ANKLE)),
]

GROUP_NAMES: Tuple[str, ...] = tuple(name for name, _ in GROUPS)
_BY_NAME = {name: lms for name, lms in GROUPS}

# Pairs that only mean anything when both halves exist.
MIRRORED: Tuple[Tuple[str, str], ...] = ((LEFT_ARM, RIGHT_ARM), (LEFT_LEG, RIGHT_LEG))
SYMMETRY_GROUPS: Tuple[str, ...] = (LEFT_ARM, RIGHT_ARM, LEFT_LEG, RIGHT_LEG)

# U+2713 / U+2014. Overridable because not every console can render them.
MARK_OK = "✓"
MARK_MISSING = "—"


# --- queries ---------------------------------------------------------------
def landmarks_for(name: str) -> Tuple[LM, ...]:
    """The landmarks a named group is made of. Empty tuple for unknown names."""
    return _BY_NAME.get(name, ())


def tracked(body_map: Optional["BodyMap"], *lms: LM) -> bool:
    """True when this body includes every one of these landmarks.

    No body map means calibration has not run, so nothing is ruled out yet.
    """
    if body_map is None:
        return True
    return bool(body_map.all_trackable(*lms))


def has_group(body_map: Optional["BodyMap"], name: str) -> bool:
    lms = landmarks_for(name)
    if not lms:
        return False
    return tracked(body_map, *lms)


def group_status(body_map: Optional["BodyMap"]) -> List[Tuple[str, bool]]:
    """[("Head", True), ("Left leg", False), ...] in display order."""
    return [(name, tracked(body_map, *lms)) for name, lms in GROUPS]


def available_groups(body_map: Optional["BodyMap"]) -> List[str]:
    return [name for name, ok in group_status(body_map) if ok]


def missing_groups(body_map: Optional["BodyMap"]) -> List[str]:
    return [name for name, ok in group_status(body_map) if not ok]


def summary_lines(body_map: Optional["BodyMap"],
                  ok: str = MARK_OK, missing: str = MARK_MISSING) -> List[str]:
    """["Head ✓", "Left leg —", ...] — one line per group, for the calibration screen."""
    return [f"{name} {ok if is_ok else missing}" for name, is_ok in group_status(body_map)]


def symmetry_available(body_map: Optional["BodyMap"]) -> bool:
    """True when left/right comparison is meaningful for this body.

    A user with one arm or one leg is not asymmetric — the comparison simply does
    not exist, and every bilateral metric built on it has to stand down.
    """
    return all(has_group(body_map, name) for name in SYMMETRY_GROUPS)


def trackable_subset(body_map: Optional["BodyMap"],
                     lms: Iterable[LM]) -> Tuple[LM, ...]:
    """Filter a requirement list down to what this body actually has.

    Used to stop an exercise declaring itself unusable because it asked for a
    landmark this user was never going to have.
    """
    if body_map is None:
        return tuple(lms)
    return tuple(lm for lm in lms if body_map.is_trackable(lm))


# --- metric gating ---------------------------------------------------------
def mark_unavailable(metrics: Sequence["Metric"], body_map: Optional["BodyMap"],
                     requires: Mapping[str, Sequence[LM]]) -> List["Metric"]:
    """Force metrics whose landmarks this body lacks to NOT_APPLICABLE.

    `requires` maps metric key -> the landmarks that metric is built from. A
    metric that is not applicable carries no value and is dropped from scoring
    entirely, which is the difference between "we can't judge this" and "you did
    it badly". Mutates and returns the metrics for convenience.
    """
    out = list(metrics)
    if body_map is None:
        return out
    for metric in out:
        needed = requires.get(metric.key)
        if needed and not tracked(body_map, *needed):
            metric.status = MetricStatus.NOT_APPLICABLE
            metric.value = None
    return out
