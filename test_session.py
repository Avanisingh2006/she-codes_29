"""Session recording and progress storage, driven by synthetic results.

No camera, no pose model — just ExerciseResults built by hand, so the
accumulation rules (especially "never score an unmeasured metric") are checked
directly rather than inferred from a video.
"""
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.errors import FormError, Priority, Severity
from core.landmarks import MetricStatus
from core.phases import RepRecord
from core.scoring import Metric
from core.session import SessionRecorder, SessionSummary, Improvement, humanize
from core.storage import ProgressStore, default_path
from exercises.base import ExerciseResult

failures = []


def check(name, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + name + ("  " + str(extra) if extra else ""))
    if not cond:
        failures.append(name)


def metric(key, value, status=MetricStatus.MEASURED, lo=None, hi=None, falloff=30.0):
    return Metric(key=key, label=key.replace("_", " ").title(), value=value,
                  status=status, lo=lo, hi=hi, falloff=falloff)


def error(key, severity=Severity.MINOR, priority=Priority.ALIGNMENT):
    return FormError(key=key, message=f"{key} message", cue=key.upper(),
                     severity=severity, priority=priority)


def result(score=80.0, metrics=(), errors=(), reps=0, hold=0.0):
    r = ExerciseResult(exercise="squat", exercise_name="Squat", movement="dynamic",
                       ready=True, score=score, rep_count=reps, hold_duration=hold)
    r.metrics = list(metrics)
    r.errors = list(errors)
    r.primary_error = r.errors[0] if r.errors else None
    return r


TMP = Path(tempfile.mkdtemp(prefix="movewise3_session_"))

try:
    # ------------------------------------------------------- accumulation
    rec = SessionRecorder(exercise="squat", exercise_name="Squat")
    for i in range(30):
        rec.update(result(score=60.0 + i,
                          metrics=[metric("knee_angle", 100.0, lo=90, hi=170)],
                          hold=i * 0.1),
                   now=i * 0.1)
    s = rec.finish()
    check("summary is a SessionSummary", isinstance(s, SessionSummary))
    check("duration from frame timestamps", abs(s.duration - 2.9) < 1e-6, s.duration)
    check("movement score is the rolling mean",
          abs(s.movement_score - 74.5) < 1e-6, s.movement_score)
    check("exercise identity carried through",
          s.exercise == "squat" and s.exercise_name == "Squat")
    check("max hold duration tracked", abs(s.hold_duration - 2.9) < 1e-6, s.hold_duration)
    check("started_at is a wall clock time", s.started_at > 1_600_000_000, s.started_at)
    check("no errors -> no main issue", s.main_issue == "" and s.most_common_error == "")

    # empty session must not explode
    empty = SessionRecorder(exercise="squat").finish()
    check("empty session finishes cleanly",
          empty.movement_score == 0.0 and empty.duration == 0.0 and empty.reps == 0)
    check("empty session has no metric scores", empty.metric_scores == {})

    # unscorable frames (score None) must not drag the mean down
    rec = SessionRecorder(exercise="squat")
    rec.update(result(score=None), now=0.0)
    rec.update(result(score=90.0, metrics=[metric("depth", 80.0, hi=100)]), now=0.1)
    check("unscorable frames excluded from the mean",
          abs(rec.finish().movement_score - 90.0) < 1e-6, rec.finish().movement_score)

    # ------------------------------------------- fairness: NA / unmeasured
    rec = SessionRecorder(exercise="squat")
    for i in range(12):
        rec.update(result(score=70.0, metrics=[
            metric("knee_angle", 120.0, lo=90, hi=170),                       # measured
            metric("hip_angle", None, status=MetricStatus.UNMEASURED, lo=60, hi=180),
            metric("left_knee_track", 5.0, status=MetricStatus.NOT_APPLICABLE, hi=10),
            metric("stability", 100.0),                                        # no band -> None
        ]), now=i * 0.1)
    s = rec.finish()
    check("measured metric is scored", "knee_angle" in s.metric_scores, s.metric_scores)
    check("unmeasured metric excluded, not zeroed", "hip_angle" not in s.metric_scores)
    check("not-applicable metric excluded, not zeroed", "left_knee_track" not in s.metric_scores)
    check("band-less metric excluded", "stability" not in s.metric_scores)
    check("no metric ever recorded as 0 for being unseen",
          all(v > 0 for v in s.metric_scores.values()), s.metric_scores)
    check("N/A metric flips the session to adaptive mode", s.adaptive_mode is True)

    # ---------------------------------------------------------- good reps
    rec = SessionRecorder(exercise="squat")
    reps = []
    for i in range(20):
        if i == 5:
            reps.append(RepRecord(index=1, duration=1.2, peak=88.0))                 # clean
        if i == 10:
            reps.append(RepRecord(index=2, duration=1.3, peak=95.0,
                                  error_keys=("back_angle",)))                       # not clean
        if i == 15:
            reps.append(RepRecord(index=3, duration=1.1, peak=87.0))                 # clean
        rec.update(result(score=75.0, reps=len(reps),
                          metrics=[metric("depth", 90.0, hi=100)]),
                   now=i * 0.1, reps=list(reps))
    s = rec.finish()
    check("reps counted", s.reps == 3, s.reps)
    check("good reps = clean RepRecords", s.good_reps == 2, s.good_reps)

    # without RepRecords, clean reps are inferred from active errors
    rec = SessionRecorder(exercise="squat")
    rec.update(result(score=80.0, reps=0), now=0.0)
    rec.update(result(score=80.0, reps=1), now=0.1)                                  # clean
    rec.update(result(score=80.0, reps=2, errors=[error("back_angle")]), now=0.2)    # dirty
    s = rec.finish()
    check("reps inferred without records", s.reps == 2, s.reps)
    check("good reps inferred from active errors", s.good_reps == 1, s.good_reps)

    # ------------------------------------------------------------- errors
    rec = SessionRecorder(exercise="squat")
    for i in range(20):
        errs = [error("back_angle", Severity.MAJOR, Priority.ALIGNMENT)] if i < 15 else []
        if i < 4:
            errs.append(error("asymmetry"))
        rec.update(result(score=70.0, errors=errs), now=i * 0.1)
    s = rec.finish()
    check("most common error key found", s.most_common_error == "back_angle", s.most_common_error)
    check("main issue is human readable", s.main_issue == "Back angle", s.main_issue)

    # ------------------------------------------------------- corrections
    rec = SessionRecorder(exercise="squat")
    rec.update(result(score=70.0), now=0.0)
    rec.note_correction(attempted=True, succeeded=False)
    rec.note_correction(attempted=True, succeeded=True)
    rec.note_correction(attempted=True, succeeded=True)
    rec.note_variation("chair-supported squat")
    s = rec.finish()
    check("corrections tracked", s.corrections == 3, s.corrections)
    check("successful corrections tracked", s.successful_corrections == 2,
          s.successful_corrections)
    check("correction rate derived", abs((s.correction_rate or 0) - 66.6667) < 0.01,
          s.correction_rate)
    check("variation recorded", s.variation_used == "chair-supported squat", s.variation_used)

    # ------------------------------------------------------- improvement
    rec = SessionRecorder(exercise="squat")
    for i in range(30):
        back = 70.0 - i                    # starts bad (over the 45 limit), ends good
        rec.update(result(score=70.0, metrics=[
            metric("back_angle", back, hi=45.0, falloff=30.0),
            metric("knee_angle", 120.0, lo=90, hi=170),      # flat: never improves
        ]), now=i * 0.1)
    s = rec.finish()
    check("improvement detected", s.biggest_improvement is not None)
    check("improvement names the metric that improved",
          s.biggest_improvement is not None and s.biggest_improvement.key == "back_angle",
          s.biggest_improvement)
    check("improvement delta is positive",
          s.biggest_improvement is not None and s.biggest_improvement.delta > 10,
          s.biggest_improvement.delta if s.biggest_improvement else None)

    # a flat session claims no improvement
    rec = SessionRecorder(exercise="squat")
    for i in range(30):
        rec.update(result(score=70.0, metrics=[metric("knee_angle", 120.0, lo=90, hi=170)]),
                   now=i * 0.1)
    check("no improvement claimed when nothing changed",
          rec.finish().biggest_improvement is None)

    # a metric that got worse is not reported as a gain
    rec = SessionRecorder(exercise="squat")
    for i in range(30):
        rec.update(result(score=70.0, metrics=[
            metric("back_angle", 40.0 + i, hi=45.0, falloff=30.0)]), now=i * 0.1)
    check("regression is not reported as improvement",
          rec.finish().biggest_improvement is None)

    # ------------------------------------------------------- presentation
    rec = SessionRecorder(exercise="squat", exercise_name="Squat")
    for i in range(30):
        rec.update(result(score=80.0, reps=1 if i > 10 else 0,
                          errors=[error("back_angle")],
                          metrics=[metric("knee_angle", 120.0, lo=90, hi=170)]),
                   now=i * 0.1)
    rec.note_correction(True, True)
    s = rec.finish()
    lines = s.headline_lines()
    check("headline lines produced", isinstance(lines, list) and len(lines) >= 4, len(lines))
    check("headline names the exercise", lines[0].startswith("Squat"), lines[0])
    check("headline reports the score",
          any("Movement score" in ln for ln in lines))
    check("headline reports the main issue",
          any("Back angle" in ln for ln in lines))
    check("headline lines are all strings", all(isinstance(ln, str) for ln in lines))

    payload = s.to_dict()
    check("to_dict is JSON serialisable", isinstance(json.dumps(payload), str))
    check("to_dict keeps the metric scores",
          payload["metric_scores"].get("knee_angle") is not None, payload["metric_scores"])
    check("to_dict carries a date label", bool(payload.get("date_label")))
    check("from_dict round-trips the summary",
          SessionSummary.from_dict(payload).movement_score == payload["movement_score"])

    # ------------------------------------------------------------ storage
    store = ProgressStore(TMP / "nested" / "sessions.json")
    check("missing file reads as empty", store.all() == [])
    check("missing file profile is empty", store.profile()["sessions"] == 0)
    check("missing file score series is empty", store.score_series() == [])

    store.save(s)
    check("save creates the file", (TMP / "nested" / "sessions.json").exists())
    check("saved session round-trips", len(store.all()) == 1)
    stored = store.all()[0]
    check("stored score matches the summary",
          abs(stored["movement_score"] - round(s.movement_score, 1)) < 1e-6, stored["movement_score"])
    check("stored file is valid JSON on disk",
          isinstance(json.loads((TMP / "nested" / "sessions.json").read_text("utf-8")), list))
    check("no stray temp files left behind",
          [p.name for p in (TMP / "nested").iterdir()] == ["sessions.json"],
          [p.name for p in (TMP / "nested").iterdir()])

    store.save(s)
    check("save appends rather than replaces", len(store.all()) == 2)

    # ---------------------------------------------------- several sessions
    store = ProgressStore(TMP / "profile.json")
    base = time.time() - 86400 * 5

    def session(exercise, name, score, metrics, issue, started, reps=(6, 5)):
        summary = SessionSummary(exercise=exercise, exercise_name=name, started_at=started,
                                 duration=120.0, movement_score=score, reps=reps[0],
                                 good_reps=reps[1], hold_duration=0.0, corrections=2,
                                 successful_corrections=1, metric_scores=metrics,
                                 main_issue=issue, most_common_error=issue.lower().replace(" ", "_"))
        store.save(summary)

    session("squat", "Squat", 60.0, {"back_angle": 50.0, "knee_angle": 90.0},
            "Back angle", base + 0)
    session("squat", "Squat", 65.0, {"back_angle": 55.0, "knee_angle": 92.0},
            "Back angle", base + 3600)
    session("tree_pose", "Tree pose", 88.0, {"stability": 95.0, "knee_angle": 91.0},
            "Hip drop", base + 7200)
    session("squat", "Squat", 80.0, {"back_angle": 78.0, "knee_angle": 93.0},
            "Back angle", base + 10800)

    check("all() returns every session", len(store.all()) == 4)
    check("for_exercise filters", len(store.for_exercise("squat")) == 3)
    check("for_exercise on unknown key is empty", store.for_exercise("nope") == [])

    series = store.score_series("squat")
    check("score series has one point per session", len(series) == 3, series)
    check("score series is (label, score)",
          all(isinstance(a, str) and isinstance(b, float) for a, b in series), series)
    check("score series is in session order",
          [b for _a, b in series] == [60.0, 65.0, 80.0], series)
    check("score series labels are dates", all(a and not a.startswith("#") for a, _b in series),
          series)

    profile = store.profile()
    check("profile counts sessions", profile["sessions"] == 4, profile["sessions"])
    check("profile overall is the mean score",
          abs(profile["overall"] - 73.25) < 0.05, profile["overall"])
    check("profile picks two strengths", len(profile["strengths"]) == 2, profile["strengths"])
    check("profile strength is the best metric",
          profile["strengths"][0] == "Stability", profile["strengths"])
    check("profile needs_attention is the worst metric",
          profile["needs_attention"] == "Back angle", profile["needs_attention"])
    check("profile most_improved found",
          profile["most_improved"] is not None and profile["most_improved"]["key"] == "back_angle",
          profile["most_improved"])
    check("profile most_improved has a positive delta",
          profile["most_improved"]["delta"] > 0, profile["most_improved"])
    check("profile best exercise is the highest scoring one",
          profile["best_exercise"]["exercise"] == "tree_pose", profile["best_exercise"])
    check("profile persistent issue is the most frequent one",
          profile["persistent_issue"] == "Back angle", profile["persistent_issue"])
    check("profile summary lines render",
          len(store.summary_lines()) >= 4, store.summary_lines())

    # a metric only some sessions measured is still averaged over those sessions
    check("partially measured metric still ranked",
          "Stability" in profile["strengths"] or profile["needs_attention"] == "Stability",
          profile)

    # -------------------------------------------------- broken files
    corrupt = TMP / "corrupt.json"
    corrupt.write_text('{"this": "is not a list", ', encoding="utf-8")
    broken = ProgressStore(corrupt)
    check("corrupt JSON tolerated by all()", broken.all() == [])
    check("corrupt JSON tolerated by profile()", broken.profile()["sessions"] == 0)
    check("corrupt JSON tolerated by score_series()", broken.score_series() == [])
    check("corrupt JSON tolerated by for_exercise()", broken.for_exercise("squat") == [])
    broken.save(s)
    check("save recovers over a corrupt file", len(broken.all()) == 1, broken.all())

    empty_file = TMP / "empty.json"
    empty_file.write_text("", encoding="utf-8")
    check("empty file tolerated", ProgressStore(empty_file).all() == [])

    wrong_shape = TMP / "wrong.json"
    wrong_shape.write_text('{"sessions": 3}', encoding="utf-8")
    check("wrong top-level type tolerated", ProgressStore(wrong_shape).all() == [])

    mixed = TMP / "mixed.json"
    mixed.write_text('[{"exercise": "squat", "movement_score": 70}, 5, null, "x"]',
                     encoding="utf-8")
    check("non-dict entries skipped", len(ProgressStore(mixed).all()) == 1,
          ProgressStore(mixed).all())
    check("garbage values do not break profile()",
          ProgressStore(mixed).profile()["sessions"] == 1)

    bad_values = TMP / "badvalues.json"
    bad_values.write_text(
        '[{"exercise": "squat", "movement_score": "eighty", '
        '"metric_scores": {"a": "x", "b": 70}, "main_issue": null}]', encoding="utf-8")
    p = ProgressStore(bad_values).profile()
    check("unparseable score ignored, not crashed", p["overall"] is None, p["overall"])
    check("unparseable metric value ignored", p["needs_attention"] == "B", p["needs_attention"])
    check("score series skips unparseable scores", ProgressStore(bad_values).score_series() == [])

    # --------------------------------------------------------- default path
    expected = Path(__file__).resolve().parent / "data" / "sessions.json"
    check("default path is data/sessions.json beside the code",
          default_path() == expected, default_path())
    check("default path is derived from the module, not hard-coded",
          "core" not in str(default_path()) and default_path().is_absolute())
    existed = expected.parent.exists()
    default_store = ProgressStore()
    check("default store points at the default path", default_store.path == expected)
    check("constructing the default store writes nothing to disk",
          expected.parent.exists() == existed and default_store.all() == [] or existed)

    # --------------------------------------------------------- integration
    store = ProgressStore(TMP / "integration.json")
    rec = SessionRecorder(exercise="squat", exercise_name="Squat")
    reps = []
    for i in range(40):
        if i in (10, 20, 30):
            reps.append(RepRecord(index=len(reps) + 1, duration=1.2, peak=88.0,
                                  error_keys=() if i != 20 else ("back_angle",)))
        rec.update(result(score=70.0 + i * 0.5, reps=len(reps), errors=[error("back_angle")],
                          metrics=[metric("back_angle", 60.0 - i, hi=45.0, falloff=30.0),
                                   metric("hip_angle", None, status=MetricStatus.UNMEASURED,
                                          lo=60, hi=180)]),
                   now=i * 0.05, reps=list(reps))
    rec.note_correction(True, True)
    summary = rec.finish()
    store.save(summary)
    reloaded = store.all()[0]
    check("end to end: session survives storage",
          reloaded["exercise"] == "squat" and reloaded["reps"] == 3
          and reloaded["good_reps"] == 2, reloaded)
    check("end to end: unmeasured metric never stored",
          "hip_angle" not in reloaded["metric_scores"], reloaded["metric_scores"])
    check("end to end: improvement stored",
          reloaded["biggest_improvement"] is not None
          and reloaded["biggest_improvement"]["key"] == "back_angle",
          reloaded["biggest_improvement"])

    # ------------------------------------------------------------- helpers
    check("humanize makes keys readable", humanize("left_knee_valgus") == "Left knee valgus")
    check("humanize tolerates empty input", humanize("") == "")
    check("improvement label reads naturally",
          Improvement("Back angle", "back_angle", 12.4).label == "Back angle +12 pts")

finally:
    shutil.rmtree(TMP, ignore_errors=True)

print()
if failures:
    print("FAILURES: " + ", ".join(failures))
    sys.exit(1)
print("FAILURES: none")
