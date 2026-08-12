"""Exercise registry and auto-detection.

Adding a fifth exercise later means writing one profile file and adding it to
PROFILE_CLASSES. Nothing in core/ changes.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple, Type

from core import config
from core.engine import EngineFrame

from .base import ExerciseProfile
from .bicep_curl import BicepCurlProfile
from .squat import SquatProfile
from .tree_pose import TreePoseProfile
from .warrior2 import WarriorTwoProfile

PROFILE_CLASSES: Tuple[Type[ExerciseProfile], ...] = (
    WarriorTwoProfile,
    TreePoseProfile,
    SquatProfile,
    BicepCurlProfile,
)


class ExerciseRegistry:
    """Owns one live instance of each profile."""

    def __init__(self) -> None:
        self._profiles: Dict[str, ExerciseProfile] = {
            cls.key: cls() for cls in PROFILE_CLASSES
        }

    @property
    def keys(self) -> List[str]:
        return list(self._profiles)

    def all(self) -> List[ExerciseProfile]:
        return list(self._profiles.values())

    def get(self, key: str) -> Optional[ExerciseProfile]:
        return self._profiles.get(key)

    def names(self) -> Dict[str, str]:
        return {k: p.name for k, p in self._profiles.items()}

    def reset_all(self) -> None:
        for profile in self._profiles.values():
            profile.reset()


@dataclass
class Recognition:
    """Outcome of one auto-detect attempt."""
    key: Optional[str]
    name: Optional[str]
    score: float
    margin: float
    confident: bool
    ranking: List[Tuple[str, float]]

    @property
    def message(self) -> str:
        if self.confident and self.name:
            return f"Detected: {self.name} ({self.score:.0%} confidence)"
        if self.name:
            return f"Not sure — closest match is {self.name}. Please choose manually."
        return "Can't tell yet. Get into position, or choose manually."


class ExerciseRecognizer:
    """Votes over a short window so a single odd frame can't flip the exercise.

    Deliberately heuristic: each profile scores its own pose signature and the
    registry picks the winner, which keeps auto-detection to the four supported
    exercises without training a custom model.
    """

    def __init__(self, registry: ExerciseRegistry,
                 window: int = config.AUTODETECT_WINDOW) -> None:
        self.registry = registry
        self._history: Deque[Dict[str, float]] = deque(maxlen=window)

    def reset(self) -> None:
        self._history.clear()

    @property
    def samples(self) -> int:
        return len(self._history)

    def observe(self, frame: EngineFrame) -> Recognition:
        scores: Dict[str, float] = {}
        for profile in self.registry.all():
            try:
                scores[profile.key] = float(profile.recognition_score(frame))
            except Exception:
                # A misbehaving profile must not break detection for the others.
                scores[profile.key] = 0.0
        self._history.append(scores)

        return self._decide()

    def _decide(self) -> Recognition:
        if not self._history:
            return Recognition(None, None, 0.0, 0.0, False, [])

        # Mean score per exercise across the voting window.
        totals: Dict[str, float] = {k: 0.0 for k in self.registry.keys}
        for sample in self._history:
            for key, value in sample.items():
                totals[key] = totals.get(key, 0.0) + value
        averaged = {k: v / len(self._history) for k, v in totals.items()}

        ranking = sorted(averaged.items(), key=lambda kv: kv[1], reverse=True)
        best_key, best_score = ranking[0]
        runner_up = ranking[1][1] if len(ranking) > 1 else 0.0
        margin = best_score - runner_up

        confident = (
            len(self._history) >= self._history.maxlen
            and best_score >= config.AUTODETECT_MIN_SCORE
            and margin >= config.AUTODETECT_MIN_MARGIN
        )

        profile = self.registry.get(best_key)
        return Recognition(
            key=best_key if best_score > 0 else None,
            name=profile.name if profile and best_score > 0 else None,
            score=best_score,
            margin=margin,
            confident=confident,
            ranking=ranking,
        )
