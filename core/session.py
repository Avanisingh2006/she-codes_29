"""What actually happened between Start and Stop.

The recorder sits inside the render loop, so `update()` is deliberately a
handful of adds and dict lookups — no I/O, no allocation of anything large, and
absolutely no database write per frame. Everything expensive (means, thirds,
ranking) happens once, in `finish()`.

The one rule that matters here is the same one the scoring layer lives by: a
metric the system could not measure, or that the body map says is not
applicable to this user, is left out of the report entirely. It is never
recorded as a zero. Scoring someone down for a joint the camera never saw is
the fastest way to make a coach untrustworthy.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

from .errors import FormError, Priority
from .landmarks import MetricStatus
from .phases import RepRecord

if TYPE_CHECKING:                       # pragma: no cover - typing only
    from exercises.base import ExerciseResult


def humanize(key: str) -> str:
    """'left_knee_valgus' -> 'Left knee valgus'. Stable enough to group on."""
    if not key:
        return ""
    return key.replace("_", " ").strip().capitalize()


@dataclass
class Improvement:
    """One metric that got better over the course of a session."""
    metric: str                 # human readable name
    key: str                    # stable metric key
    delta: float                # points gained, first third -> last third

    @property
    def label(self) -> str:
        sign = "+" if self.delta >= 0 else ""
        return f"{self.metric} {sign}{self.delta:.0f} pts"

    def to_dict(self) -> dict:
        return {"metric": self.metric, "key": self.key, "delta": round(float(self.delta), 1)}

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> Optional["Improvement"]:
        if not isinstance(data, dict):
            return None
        try:
            return cls(metric=str(data.get("metric", "")),
                       key=str(data.get("key", "")),
                       delta=float(data.get("delta", 0.0)))
        except (TypeError, ValueError):
            return None


@dataclass
class SessionSummary:
    """The record of one session — the thing that gets shown and stored."""
    exercise: str
    exercise_name: str
    started_at: float
    duration: float
    movement_score: float                       # 0-100
    reps: int
    good_reps: int
    hold_duration: float
    corrections: int
    successful_corrections: int
    metric_scores: Dict[str, float] = field(default_factory=dict)
    main_issue: str = ""                        # human readable
    most_common_error: str = ""                 # raw error key
    biggest_improvement: Optional[Improvement] = None
    variation_used: Optional[str] = None
    adaptive_mode: bool = False
    # -- movement control (added later; old records simply lack these) -----
    control_score: Optional[float] = None       # 0-100 mean smoothness, None if unmeasured
    unstable_events: int = 0                    # jerky/unstable moments detected
    control_improved: Optional[bool] = None     # first third vs last third; None if too few samples

    # -- derived ----------------------------------------------------------
    @property
    def clean_rate(self) -> Optional[float]:
        if self.reps <= 0:
            return None
        return 100.0 * self.good_reps / self.reps

    @property
    def correction_rate(self) -> Optional[float]:
        if self.corrections <= 0:
            return None
        return 100.0 * self.successful_corrections / self.corrections

    @property
    def date_label(self) -> str:
        try:
            return time.strftime("%d %b %H:%M", time.localtime(self.started_at))
        except (ValueError, OSError, OverflowError):
            return ""

    def to_dict(self) -> dict:
        return {
            "exercise": self.exercise,
            "exercise_name": self.exercise_name,
            "started_at": float(self.started_at),
            "date_label": self.date_label,
            "duration": round(float(self.duration), 2),
            "movement_score": round(float(self.movement_score), 1),
            "reps": int(self.reps),
            "good_reps": int(self.good_reps),
            "hold_duration": round(float(self.hold_duration), 2),
            "corrections": int(self.corrections),
            "successful_corrections": int(self.successful_corrections),
            "metric_scores": {k: round(float(v), 1) for k, v in self.metric_scores.items()},
            "main_issue": self.main_issue,
            "most_common_error": self.most_common_error,
            "biggest_improvement": (self.biggest_improvement.to_dict()
                                    if self.biggest_improvement else None),
            "variation_used": self.variation_used,
            "adaptive_mode": bool(self.adaptive_mode),
            "control_score": (round(float(self.control_score), 1)
                              if self.control_score is not None else None),
            "unstable_events": int(self.unstable_events),
            "control_improved": (bool(self.control_improved)
                                 if self.control_improved is not None else None),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SessionSummary":
        """Rebuild a summary from stored JSON. Tolerant of missing keys."""
        raw_metrics = data.get("metric_scores") or {}
        metrics: Dict[str, float] = {}
        if isinstance(raw_metrics, dict):
            for key, value in raw_metrics.items():
                try:
                    metrics[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue
        def num(key: str, default: float = 0.0) -> float:
            try:
                return float(data.get(key, default))
            except (TypeError, ValueError):
                return default
        # Control fields arrived later: old records simply do not have them,
        # and that must read back as "not measured", never as zero.
        raw_control = data.get("control_score")
        control_score: Optional[float] = None
        if raw_control is not None and not isinstance(raw_control, bool):
            try:
                control_score = float(raw_control)
            except (TypeError, ValueError):
                control_score = None
        raw_improved = data.get("control_improved")
        control_improved = bool(raw_improved) if isinstance(raw_improved, bool) else None
        return cls(
            exercise=str(data.get("exercise", "")),
            exercise_name=str(data.get("exercise_name", "")),
            started_at=num("started_at"),
            duration=num("duration"),
            movement_score=num("movement_score"),
            reps=int(num("reps")),
            good_reps=int(num("good_reps")),
            hold_duration=num("hold_duration"),
            corrections=int(num("corrections")),
            successful_corrections=int(num("successful_corrections")),
            metric_scores=metrics,
            main_issue=str(data.get("main_issue") or ""),
            most_common_error=str(data.get("most_common_error") or ""),
            biggest_improvement=Improvement.from_dict(data.get("biggest_improvement")),
            variation_used=data.get("variation_used") or None,
            adaptive_mode=bool(data.get("adaptive_mode", False)),
            control_score=control_score,
            unstable_events=max(0, int(num("unstable_events", 0))),
            control_improved=control_improved,
        )

    def headline_lines(self) -> List[str]:
        """The end-of-session block, one line per fact worth reading."""
        name = self.exercise_name or humanize(self.exercise) or "Session"
        lines: List[str] = [
            f"{name} — {self._duration_text()}",
            f"Movement score    {self.movement_score:.0f}/100",
        ]
        if self.control_score is not None:
            lines.append(f"Movement control  {self.control_score:.0f}/100")

        if self.reps > 0:
            lines.append(f"Reps              {self.reps} ({self.good_reps} clean)")
        if self.hold_duration > 0.5:
            lines.append(f"Best hold         {self.hold_duration:.0f}s")

        if self.metric_scores:
            best = max(self.metric_scores.items(), key=lambda kv: kv[1])
            worst = min(self.metric_scores.items(), key=lambda kv: kv[1])
            lines.append(f"Strongest         {humanize(best[0])} {best[1]:.0f}/100")
            if worst[0] != best[0]:
                lines.append(f"Weakest           {humanize(worst[0])} {worst[1]:.0f}/100")

        if self.main_issue:
            lines.append(f"Main issue        {self.main_issue}")
        if self.corrections > 0:
            lines.append(f"Corrections       {self.successful_corrections}/{self.corrections} fixed")
        if self.unstable_events > 0:
            lines.append(f"Unstable moments  {self.unstable_events}")
        if self.control_improved:
            lines.append("Control trend     smoother by the final repetitions")
        if self.biggest_improvement is not None:
            lines.append(f"Biggest gain      {self.biggest_improvement.label}")
        if self.variation_used:
            lines.append(f"Variation         {self.variation_used}")
        if self.adaptive_mode:
            lines.append("Adaptive mode — untracked joints were left out, not marked down.")
        return lines

    def _duration_text(self) -> str:
        seconds = max(0.0, float(self.duration))
        if seconds < 60:
            return f"{seconds:.0f}s"
        return f"{int(seconds // 60)}m {int(seconds % 60):02d}s"


class SessionRecorder:
    """Accumulates one exercise session in memory.

    Feed it every frame's ExerciseResult; ask it for a SessionSummary at the
    end. Optionally pass the phase machine's RepRecords so clean reps are read
    from the reps themselves rather than inferred.
    """

    #: A metric needs at least this many samples before a first-third /
    #: last-third comparison means anything.
    MIN_TREND_SAMPLES = 6
    #: Gain smaller than this is noise, not improvement.
    MIN_TREND_DELTA = 2.0

    def __init__(self, exercise: str = "", exercise_name: str = "",
                 adaptive_mode: bool = False, started_at: Optional[float] = None) -> None:
        self.exercise = exercise
        self.exercise_name = exercise_name
        self.adaptive_mode = bool(adaptive_mode)
        self.started_at = float(started_at) if started_at is not None else time.time()
        self.reset()

    # -- lifecycle ---------------------------------------------------------
    def reset(self) -> None:
        self.frames = 0
        self._score_sum = 0.0
        self._score_n = 0
        self._metric_samples: Dict[str, List[float]] = {}
        self._error_counts: Dict[str, int] = {}
        self._error_seen: Dict[str, FormError] = {}
        self._first_now: Optional[float] = None
        self._last_now: Optional[float] = None
        self.reps = 0
        self.good_reps = 0
        self.hold_duration = 0.0
        self.corrections = 0
        self.successful_corrections = 0
        self.variation_used: Optional[str] = None
        self._rep_records: List[RepRecord] = []
        self._clean_from_records = False
        self._control_samples: List[float] = []
        self.unstable_events = 0

    # -- per frame ---------------------------------------------------------
    def update(self, result: "ExerciseResult", now: float,
               reps: Optional[Sequence[RepRecord]] = None) -> None:
        """Fold one frame into the running totals. Cheap by design."""
        if result is None:
            return

        now = float(now)
        if self._first_now is None:
            self._first_now = now
        self._last_now = max(now, self._last_now if self._last_now is not None else now)

        self.frames += 1
        if not self.exercise:
            self.exercise = getattr(result, "exercise", "") or ""
        if not self.exercise_name:
            self.exercise_name = getattr(result, "exercise_name", "") or ""

        score = getattr(result, "score", None)
        if score is not None:
            self._score_sum += float(score)
            self._score_n += 1

        # --- movement control: read defensively — the analysis layer may or
        # may not expose these fields yet, and either has to work. -----------
        control = getattr(result, "control", None)
        if control is not None:
            try:
                self._control_samples.append(float(control))
            except (TypeError, ValueError):
                pass
        unstable = getattr(result, "unstable_events", 0)
        try:
            unstable = int(unstable) if unstable is not None else 0
        except (TypeError, ValueError):
            unstable = 0
        if unstable > self.unstable_events:
            self.unstable_events = unstable

        # --- metrics: only what was genuinely measured and scorable --------
        for metric in getattr(result, "metrics", None) or ():
            if getattr(metric, "status", None) is MetricStatus.NOT_APPLICABLE:
                self.adaptive_mode = True          # the body map dropped this joint
                continue
            value = metric.score()
            if value is None:                      # unmeasured, or no target band
                continue
            self._metric_samples.setdefault(metric.key, []).append(float(value))

        # --- errors: how often each one was actually active ----------------
        for error in getattr(result, "errors", None) or ():
            self._error_counts[error.key] = self._error_counts.get(error.key, 0) + 1
            self._error_seen[error.key] = error

        # --- reps ----------------------------------------------------------
        previous = self.reps
        count = int(getattr(result, "rep_count", 0) or 0)
        if count > self.reps:
            self.reps = count

        records = reps if reps is not None else getattr(result, "rep_records", None)
        if records:
            self._rep_records = list(records)
            self._clean_from_records = True
            self.reps = max(self.reps, len(self._rep_records))
        elif self.reps > previous and not self._clean_from_records:
            # No rep records available: a rep that completed with nothing
            # active against it counts as clean.
            if not (getattr(result, "errors", None) or ()):
                self.good_reps += (self.reps - previous)

        hold = float(getattr(result, "hold_duration", 0.0) or 0.0)
        if hold > self.hold_duration:
            self.hold_duration = hold

    # -- annotations from the layers above ---------------------------------
    def note_correction(self, attempted: bool, succeeded: bool) -> None:
        """The coaching layer asked for a fix; did the user manage it."""
        if attempted:
            self.corrections += 1
        if succeeded:
            self.successful_corrections += 1

    def note_variation(self, name: str) -> None:
        """An easier/alternative version of the movement was offered and used."""
        if name:
            self.variation_used = str(name)

    def note_adaptive(self, adaptive: bool = True) -> None:
        self.adaptive_mode = bool(adaptive)

    # -- reporting ---------------------------------------------------------
    @property
    def duration(self) -> float:
        if self._first_now is None or self._last_now is None:
            return 0.0
        return max(0.0, self._last_now - self._first_now)

    @property
    def movement_score(self) -> float:
        if self._score_n <= 0:
            return 0.0
        return float(max(0.0, min(100.0, self._score_sum / self._score_n)))

    def metric_scores(self) -> Dict[str, float]:
        """Mean score per metric — measured, applicable metrics only."""
        return {key: float(sum(values) / len(values))
                for key, values in self._metric_samples.items() if values}

    def _good_reps(self) -> int:
        if self._clean_from_records:
            return sum(1 for r in self._rep_records if r.clean)
        return min(self.good_reps, self.reps)

    def _main_error(self) -> Tuple[str, str]:
        """(raw key, human readable). Frequency first, then how serious it is."""
        if not self._error_counts:
            return "", ""

        def rank(item: Tuple[str, int]) -> tuple:
            key, count = item
            error = self._error_seen.get(key)
            priority = int(error.priority) if error else int(Priority.REFINEMENT)
            weight = error.severity.weight if error else 1.0
            return (-count, priority, -weight)

        key = sorted(self._error_counts.items(), key=rank)[0][0]
        return key, humanize(key)

    def _biggest_improvement(self) -> Optional[Improvement]:
        """Compare the first third of the session with the last third."""
        best: Optional[Improvement] = None
        for key, values in self._metric_samples.items():
            if len(values) < self.MIN_TREND_SAMPLES:
                continue
            third = max(1, len(values) // 3)
            early = sum(values[:third]) / third
            late = sum(values[-third:]) / third
            delta = late - early
            if delta < self.MIN_TREND_DELTA:
                continue
            if best is None or delta > best.delta:
                best = Improvement(metric=humanize(key), key=key, delta=float(delta))
        return best

    def _control_trend(self) -> Tuple[Optional[float], Optional[bool]]:
        """(mean control score, did it improve first third -> last third).

        None score means control was never measured this session — it is not
        recorded as zero. The improvement verdict needs MIN_TREND_SAMPLES
        samples before a thirds comparison means anything; below that it is
        None, not False.
        """
        values = self._control_samples
        if not values:
            return None, None
        score = float(sum(values) / len(values))
        if len(values) < self.MIN_TREND_SAMPLES:
            return score, None
        third = max(1, len(values) // 3)
        early = sum(values[:third]) / third
        late = sum(values[-third:]) / third
        return score, bool((late - early) >= self.MIN_TREND_DELTA)

    def finish(self) -> SessionSummary:
        """Close the session and produce its summary. Safe to call repeatedly."""
        error_key, issue = self._main_error()
        control_score, control_improved = self._control_trend()
        return SessionSummary(
            exercise=self.exercise,
            exercise_name=self.exercise_name or humanize(self.exercise),
            started_at=self.started_at,
            duration=self.duration,
            movement_score=self.movement_score,
            reps=self.reps,
            good_reps=self._good_reps(),
            hold_duration=self.hold_duration,
            corrections=self.corrections,
            successful_corrections=self.successful_corrections,
            metric_scores=self.metric_scores(),
            main_issue=issue,
            most_common_error=error_key,
            biggest_improvement=self._biggest_improvement(),
            variation_used=self.variation_used,
            adaptive_mode=self.adaptive_mode,
            control_score=control_score,
            unstable_events=self.unstable_events,
            control_improved=control_improved,
        )
