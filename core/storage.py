"""Local progress storage — one JSON file, no accounts, no cloud.

Progress tracking is only useful if it is free to the user: no sign-up, no
upload of anything a camera saw. Sessions are appended to a plain JSON file
next to the code, so the whole history is inspectable and deletable by hand.

Reads never raise. A missing file, an empty file, a half-written file from a
crash, or a file someone edited badly all resolve to "no history yet" — a
progress panel is not worth taking the app down for. Writes are atomic
(temp file then replace) so a crash mid-save cannot corrupt what is already
there.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from collections import Counter
from datetime import date as _Date, timedelta as _TimeDelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .session import SessionSummary, humanize

#: data/ lives beside the package, resolved from this file — never hard-coded.
DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data"
DEFAULT_FILENAME = "sessions.json"


def default_path() -> Path:
    return DATA_DIR / DEFAULT_FILENAME


class ProgressStore:
    """Append-only history of finished sessions."""

    def __init__(self, path: Optional[Union[str, Path]] = None) -> None:
        self.path = Path(path) if path is not None else default_path()

    # -- io ----------------------------------------------------------------
    def _read(self) -> List[dict]:
        """Every stored session, or [] for anything we cannot make sense of."""
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (FileNotFoundError, NotADirectoryError, PermissionError, OSError):
            return []
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            return []       # corrupt or half-written: treat as empty history
        if not isinstance(data, list):
            return []
        return [item for item in data if isinstance(item, dict)]

    def _write(self, sessions: Sequence[dict]) -> bool:
        """Atomic write: temp file in the same directory, then replace."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            handle = tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=str(self.path.parent),
                prefix=self.path.name + ".", suffix=".tmp", delete=False)
            temp_name = handle.name
            try:
                with handle:
                    json.dump(list(sessions), handle, indent=2)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp_name, str(self.path))
            except BaseException:
                try:
                    os.unlink(temp_name)
                except OSError:
                    pass
                raise
        except (OSError, TypeError, ValueError):
            return False
        return True

    # -- api ---------------------------------------------------------------
    def save(self, summary: Union[SessionSummary, dict]) -> None:
        """Append one finished session."""
        if summary is None:
            return
        record = summary.to_dict() if hasattr(summary, "to_dict") else dict(summary)
        sessions = self._read()
        sessions.append(record)
        self._write(sessions)

    def all(self) -> List[dict]:
        return self._read()

    def for_exercise(self, key: str) -> List[dict]:
        return [s for s in self._read() if s.get("exercise") == key]

    def count(self) -> int:
        return len(self._read())

    def clear(self) -> None:
        """Delete the history file. The user owns this data; make it easy to drop."""
        try:
            self.path.unlink()
        except (FileNotFoundError, OSError):
            pass

    def score_series(self, exercise: Optional[str] = None) -> List[Tuple[str, float]]:
        """(date label, movement score) in the order the sessions happened."""
        sessions = self.for_exercise(exercise) if exercise else self._read()
        series: List[Tuple[str, float]] = []
        for index, session in enumerate(sessions, start=1):
            score = _number(session.get("movement_score"))
            if score is None:
                continue
            series.append((_label(session, index), float(score)))
        return series

    def last(self, exercise: Optional[str] = None) -> Optional[dict]:
        sessions = self.for_exercise(exercise) if exercise else self._read()
        return sessions[-1] if sessions else None

    # -- the last week -------------------------------------------------------
    def last_7_days(self, exercise: Optional[str] = None) -> List[dict]:
        """Exactly 7 entries, oldest first, ending today.

        A day with no sessions keeps score/control as None — an empty day is
        empty, never zero and never interpolated. Grouping uses each record's
        own started_at in local time.
        """
        sessions = self.for_exercise(exercise) if exercise else self._read()
        by_day: Dict[_Date, List[dict]] = {}
        for session in sessions:
            day = _day_of(_number(session.get("started_at")))
            if day is None:
                continue
            by_day.setdefault(day, []).append(session)

        today = _Date.today()
        out: List[dict] = []
        for offset in range(6, -1, -1):
            day = today - _TimeDelta(days=offset)
            todays = by_day.get(day, [])
            scores = [v for v in (_number(x.get("movement_score")) for x in todays)
                      if v is not None]
            controls = [v for v in (_number(x.get("control_score")) for x in todays)
                        if v is not None]
            out.append({
                "label": day.strftime("%a"),
                "date": _day_label(day),
                "score": round(sum(scores) / len(scores), 1) if scores else None,
                "control": round(sum(controls) / len(controls), 1) if controls else None,
                "sessions": len(todays),
            })
        return out

    def sessions_list(self) -> List[dict]:
        """Every stored session as a pickable row, newest first.

        `index` indexes into `all()` so a row can be resolved back to its full
        stored record via `get_session()`.
        """
        out: List[dict] = []
        for index, session in enumerate(self._read()):
            day = _day_of(_number(session.get("started_at")))
            name = str(session.get("exercise_name")
                       or humanize(str(session.get("exercise") or ""))
                       or "Session")
            out.append({
                "index": index,
                "date": _day_label(day) if day is not None else "--",
                "exercise_name": name,
                "score": _number(session.get("movement_score")),
                "control": _number(session.get("control_score")),
            })
        out.reverse()
        return out

    def get_session(self, index) -> Optional[dict]:
        """One stored record by its index into all(). None if out of range."""
        try:
            index = int(index)
        except (TypeError, ValueError):
            return None
        sessions = self._read()
        if 0 <= index < len(sessions):
            return sessions[index]
        return None

    def exercise_progress(self) -> Dict[str, dict]:
        """Per-exercise aggregates. Exercises are never merged — a squat
        average says nothing about a tree pose and must never blend into one.
        """
        buckets: Dict[str, dict] = {}
        for session in self._read():
            key = str(session.get("exercise") or "")
            if not key:
                continue
            bucket = buckets.setdefault(key, {"name": "", "scores": [], "controls": []})
            if not bucket["name"]:
                bucket["name"] = str(session.get("exercise_name") or humanize(key))
            score = _number(session.get("movement_score"))
            if score is not None:
                bucket["scores"].append(score)
            control = _number(session.get("control_score"))
            if control is not None:
                bucket["controls"].append(control)
            bucket["sessions"] = bucket.get("sessions", 0) + 1

        out: Dict[str, dict] = {}
        for key, bucket in buckets.items():
            scores, controls = bucket["scores"], bucket["controls"]
            out[key] = {
                "name": bucket["name"] or humanize(key),
                "sessions": int(bucket.get("sessions", 0)),
                "avg_score": round(sum(scores) / len(scores), 1) if scores else None,
                "avg_control": round(sum(controls) / len(controls), 1) if controls else None,
                "best_score": round(max(scores), 1) if scores else None,
            }
        return out

    def improvement(self, exercise: Optional[str] = None) -> Optional[dict]:
        """Earliest session vs latest session: {"accuracy": (a, b), "control": (a, b)}.

        Either comparison is None when one of its ends was not measured, and
        the whole answer is None below two sessions — one data point is not a
        trend.
        """
        sessions = self.for_exercise(exercise) if exercise else self._read()
        if len(sessions) < 2:
            return None
        first, last = sessions[0], sessions[-1]

        def pair(key: str) -> Optional[Tuple[float, float]]:
            a, b = _number(first.get(key)), _number(last.get(key))
            if a is None or b is None:
                return None
            return (a, b)

        return {"accuracy": pair("movement_score"), "control": pair("control_score")}

    def weekly_summary(self) -> dict:
        """The last 7 calendar days, summed up honestly.

        The note only claims an improvement the stored numbers actually show;
        with fewer than two sessions in the window it says more sessions are
        required instead of guessing. Fitness wording only — never a claim
        about health, injury or recovery.
        """
        today = _Date.today()
        start = today - _TimeDelta(days=6)
        window: List[dict] = []
        for session in self._read():
            day = _day_of(_number(session.get("started_at")))
            if day is None or day < start or day > today:
                continue
            window.append(session)

        scores = [v for v in (_number(x.get("movement_score")) for x in window)
                  if v is not None]
        controls = [v for v in (_number(x.get("control_score")) for x in window)
                    if v is not None]
        avg_score = round(sum(scores) / len(scores), 1) if scores else None
        avg_control = round(sum(controls) / len(controls), 1) if controls else None

        improved = self._most_improved(window)
        most_improved = improved["metric"] if improved else None

        totals: Dict[str, List[float]] = {}
        for session in window:
            for key, value in _metric_items(session):
                totals.setdefault(key, []).append(value)
        means = {k: sum(v) / len(v) for k, v in totals.items() if v}
        focus = humanize(min(means, key=means.get)) if means else None

        if len(window) < 2:
            note = ("More sessions are required to build a weekly summary — "
                    "complete a few sessions this week to see your trend.")
        else:
            note = _weekly_note(window)

        return {
            "sessions": len(window),
            "avg_score": avg_score,
            "avg_control": avg_control,
            "most_improved": most_improved,
            "focus": focus,
            "note": note,
        }

    # -- the long view -----------------------------------------------------
    def profile(self) -> dict:
        """What the history says about this user, across every session."""
        sessions = self._read()
        blank = {
            "overall": None,
            "strengths": [],
            "needs_attention": None,
            "most_improved": None,
            "best_exercise": None,
            "persistent_issue": None,
            "sessions": 0,
        }
        if not sessions:
            return blank

        scores = [s for s in (_number(x.get("movement_score")) for x in sessions)
                  if s is not None]
        overall = round(sum(scores) / len(scores), 1) if scores else None

        # --- metric means. Absent metrics stay absent; they never score 0. --
        totals: Dict[str, List[float]] = {}
        for session in sessions:
            for key, value in _metric_items(session):
                totals.setdefault(key, []).append(value)
        means = {key: sum(values) / len(values) for key, values in totals.items() if values}
        ranked = sorted(means.items(), key=lambda kv: kv[1], reverse=True)

        strengths = [humanize(key) for key, _v in ranked[:2]]
        needs_attention = humanize(ranked[-1][0]) if ranked else None

        # --- improvement: earlier half of the history vs the later half -----
        most_improved = self._most_improved(sessions)

        # --- best exercise --------------------------------------------------
        per_exercise: Dict[str, List[float]] = {}
        names: Dict[str, str] = {}
        for session in sessions:
            key = str(session.get("exercise") or "")
            score = _number(session.get("movement_score"))
            if not key or score is None:
                continue
            per_exercise.setdefault(key, []).append(score)
            names.setdefault(key, str(session.get("exercise_name") or humanize(key)))
        best_exercise = None
        if per_exercise:
            key, values = max(per_exercise.items(),
                              key=lambda kv: sum(kv[1]) / len(kv[1]))
            best_exercise = {"exercise": key, "name": names.get(key, humanize(key)),
                             "score": round(sum(values) / len(values), 1)}

        issues = Counter(str(s.get("main_issue")) for s in sessions if s.get("main_issue"))
        persistent_issue = issues.most_common(1)[0][0] if issues else None

        return {
            "overall": overall,
            "strengths": strengths,
            "needs_attention": needs_attention,
            "most_improved": most_improved,
            "best_exercise": best_exercise,
            "persistent_issue": persistent_issue,
            "sessions": len(sessions),
        }

    def _most_improved(self, sessions: Sequence[dict]) -> Optional[dict]:
        """Metric with the biggest gain from the earlier half to the later half."""
        if len(sessions) < 2:
            return None
        split = max(1, len(sessions) // 2)
        early, late = sessions[:split], sessions[split:]

        def means(group: Iterable[dict]) -> Dict[str, float]:
            bucket: Dict[str, List[float]] = {}
            for session in group:
                for key, value in _metric_items(session):
                    bucket.setdefault(key, []).append(value)
            return {k: sum(v) / len(v) for k, v in bucket.items() if v}

        first, last = means(early), means(late)
        best: Optional[dict] = None
        for key, late_mean in last.items():
            if key not in first:
                continue
            delta = late_mean - first[key]
            if delta <= 0.5:
                continue
            if best is None or delta > best["delta"]:
                best = {"metric": humanize(key), "key": key, "delta": round(float(delta), 1)}
        return best

    def summary_lines(self) -> List[str]:
        """The progress panel, as plain lines."""
        profile = self.profile()
        if not profile["sessions"]:
            return ["No sessions recorded yet."]
        lines = [f"Sessions          {profile['sessions']}"]
        if profile["overall"] is not None:
            lines.append(f"Overall           {profile['overall']:.0f}/100")
        if profile["strengths"]:
            lines.append("Strengths         " + ", ".join(profile["strengths"]))
        if profile["needs_attention"]:
            lines.append(f"Needs attention   {profile['needs_attention']}")
        if profile["most_improved"]:
            improved = profile["most_improved"]
            lines.append(f"Most improved     {improved['metric']} +{improved['delta']:.0f} pts")
        if profile["best_exercise"]:
            best = profile["best_exercise"]
            lines.append(f"Best exercise     {best['name']} {best['score']:.0f}/100")
        if profile["persistent_issue"]:
            lines.append(f"Recurring issue   {profile['persistent_issue']}")
        return lines


# -- helpers ---------------------------------------------------------------
def _number(value) -> Optional[float]:
    try:
        if value is None or isinstance(value, bool):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric_items(session: dict) -> List[Tuple[str, float]]:
    """Usable (metric key, score) pairs from one stored session."""
    raw = session.get("metric_scores")
    if not isinstance(raw, dict):
        return []
    items: List[Tuple[str, float]] = []
    for key, value in raw.items():
        score = _number(value)
        if score is not None:
            items.append((str(key), score))
    return items


def _day_of(started: Optional[float]) -> Optional[_Date]:
    """Local calendar day of a timestamp, or None for anything unusable."""
    if started is None:
        return None
    try:
        return _Date.fromtimestamp(started)
    except (OverflowError, OSError, ValueError):
        return None


def _day_label(day: _Date) -> str:
    """'12 Aug' — no zero padding, no locale surprises worth having."""
    return f"{day.day} {day.strftime('%b')}"


def _weekly_note(window: Sequence[dict]) -> str:
    """One honest sentence about the week. Control first, accuracy second.

    A percentage is only claimed when both ends of the week measured the same
    thing; otherwise the note stays neutral. No medical language, ever.
    """
    def ends(key: str) -> Optional[Tuple[float, float]]:
        values = [v for v in (_number(x.get(key)) for x in window) if v is not None]
        if len(values) < 2:
            return None
        return values[0], values[-1]

    control = ends("control_score")
    if control and control[0] > 0:
        pct = 100.0 * (control[1] - control[0]) / control[0]
        if pct >= 1.0:
            return f"Your movement control improved by {pct:.0f}% this week."

    accuracy = ends("movement_score")
    if accuracy and accuracy[0] > 0:
        pct = 100.0 * (accuracy[1] - accuracy[0]) / accuracy[0]
        if pct >= 1.0:
            return f"Your accuracy improved by {pct:.0f}% this week."
        if pct <= -1.0:
            return ("Your scores were a little lower this week — normal "
                    "variation between sessions.")

    return "Your performance held steady this week."


def _label(session: dict, index: int) -> str:
    label = session.get("date_label")
    if isinstance(label, str) and label:
        return label
    started = _number(session.get("started_at"))
    if started:
        try:
            return time.strftime("%d %b %H:%M", time.localtime(started))
        except (ValueError, OSError, OverflowError):
            pass
    return f"#{index}"
