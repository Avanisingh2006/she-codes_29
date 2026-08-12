"""Persistent 7-day progress tracking, checked against a temp store.

Everything here runs on a throwaway directory — never the real data dir — and
the persistence test proves the restart story by re-opening a brand-new
ProgressStore instance on the same path. The empty days of a week must stay
None: the store never fabricates a score for a day nobody trained.
"""
import json
import shutil
import sys
import tempfile
import time
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.session import SessionRecorder, SessionSummary
from core.storage import ProgressStore

failures = []


def check(name, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + name + ("  " + str(extra) if extra else ""))
    if not cond:
        failures.append(name)


TODAY = date.today()


def midday(days_ago):
    """Wall-clock timestamp at 12:00 local, N days ago — DST-proof for tests."""
    d = TODAY - timedelta(days=days_ago)
    return datetime(d.year, d.month, d.day, 12, 0).timestamp()


def summary(exercise="squat", name="Squat", score=70.0, started=None, control=None,
            metrics=None, unstable=0, improved=None):
    return SessionSummary(
        exercise=exercise, exercise_name=name,
        started_at=started if started is not None else time.time(),
        duration=60.0, movement_score=score, reps=5, good_reps=4, hold_duration=0.0,
        corrections=1, successful_corrections=1, metric_scores=metrics or {},
        control_score=control, unstable_events=unstable, control_improved=improved)


class BareResult:
    """A frame result from an analysis layer that knows nothing about control."""
    def __init__(self, score=80.0):
        self.score = score
        self.metrics = []
        self.errors = []
        self.rep_count = 0
        self.hold_duration = 0.0


class ControlResult(BareResult):
    """A frame result from an analysis layer that does measure control."""
    def __init__(self, score=80.0, control=None, unstable=0):
        super().__init__(score)
        self.control = control
        self.unstable_events = unstable


TMP = Path(tempfile.mkdtemp(prefix="movewise3_progress_"))

try:
    # ------------------------------------------------------- last_7_days
    store = ProgressStore(TMP / "seven.json")
    days = store.last_7_days()
    check("empty store still yields exactly 7 days", len(days) == 7, len(days))
    check("empty days carry None scores, never zero",
          all(d["score"] is None and d["control"] is None for d in days))
    check("empty days count zero sessions", all(d["sessions"] == 0 for d in days))
    expected_labels = [(TODAY - timedelta(days=o)).strftime("%a") for o in range(6, -1, -1)]
    check("day labels run oldest to today", [d["label"] for d in days] == expected_labels,
          [d["label"] for d in days])
    check("last entry is today's date",
          days[-1]["date"] == f"{TODAY.day} {TODAY.strftime('%b')}", days[-1]["date"])

    store.save(summary(score=80.0, control=75.0, started=midday(0)))
    store.save(summary(score=90.0, control=85.0, started=midday(0)))
    store.save(summary(exercise="tree_pose", name="Tree pose", score=60.0,
                       started=midday(2)))
    store.save(summary(score=50.0, started=midday(6)))
    store.save(summary(score=99.0, started=midday(8)))       # outside the window

    days = store.last_7_days()
    check("still exactly 7 entries with data", len(days) == 7)
    check("today averages its two sessions",
          days[-1]["score"] == 85.0 and days[-1]["sessions"] == 2, days[-1])
    check("today averages measured control", days[-1]["control"] == 80.0, days[-1])
    check("day without control stays None for control",
          days[4]["score"] == 60.0 and days[4]["control"] is None, days[4])
    check("oldest day of the window included", days[0]["score"] == 50.0, days[0])
    check("session older than 7 days excluded",
          all(d["score"] != 99.0 for d in days if d["score"] is not None))
    check("untrained days in a busy week stay None",
          days[1]["score"] is None and days[2]["score"] is None
          and days[3]["score"] is None and days[5]["score"] is None)

    filtered = store.last_7_days(exercise="tree_pose")
    check("exercise filter keeps only that exercise",
          filtered[4]["score"] == 60.0 and filtered[-1]["score"] is None, filtered[4])

    # --------------------------------------------- persistence / restart
    reopened = ProgressStore(TMP / "seven.json")     # a fresh instance = restart
    check("history survives a new store instance", len(reopened.all()) == 5)
    check("7-day view identical after restart",
          reopened.last_7_days() == store.last_7_days())
    check("control fields survive the restart",
          reopened.all()[0].get("control_score") == 75.0, reopened.all()[0].get("control_score"))

    # ------------------------------------- sessions_list / get_session
    listed = store.sessions_list()
    check("sessions_list covers every session", len(listed) == 5)
    check("sessions_list is newest first",
          listed[0]["index"] == 4 and listed[-1]["index"] == 0,
          [i["index"] for i in listed])
    check("list rows carry name, score and control",
          listed[-1]["exercise_name"] == "Squat" and listed[-1]["score"] == 80.0
          and listed[-1]["control"] == 75.0, listed[-1])
    check("list rows carry a readable date",
          listed[-1]["date"] == f"{TODAY.day} {TODAY.strftime('%b')}", listed[-1]["date"])
    picked = store.get_session(listed[-1]["index"])
    check("get_session resolves a listed row",
          picked is not None and picked["movement_score"] == 80.0
          and picked["exercise_name"] == "Squat", picked)
    check("get_session out of range is None", store.get_session(999) is None)
    check("get_session negative index is None", store.get_session(-1) is None)
    check("get_session garbage index is None", store.get_session("nope") is None)

    # --------------------------------------------------- exercise_progress
    prog = store.exercise_progress()
    check("one entry per exercise, never merged",
          set(prog) == {"squat", "tree_pose"}, set(prog))
    sq = prog["squat"]
    check("squat sessions counted separately", sq["sessions"] == 4, sq)
    check("squat average is squat-only",
          abs(sq["avg_score"] - (80 + 90 + 50 + 99) / 4.0) < 0.05, sq["avg_score"])
    check("squat best score found", sq["best_score"] == 99.0, sq["best_score"])
    check("squat avg control ignores unmeasured sessions",
          sq["avg_control"] == 80.0, sq["avg_control"])
    tp = prog["tree_pose"]
    check("tree pose stands alone",
          tp["sessions"] == 1 and tp["avg_score"] == 60.0 and tp["best_score"] == 60.0, tp)
    check("exercise with no control data reads None, not zero",
          tp["avg_control"] is None, tp["avg_control"])

    # -------------------------------------------------------- improvement
    single = ProgressStore(TMP / "single.json")
    single.save(summary(score=70.0))
    check("improvement needs at least two sessions", single.improvement() is None)

    both = ProgressStore(TMP / "both.json")
    both.save(summary(score=60.0, control=50.0, started=midday(3)))
    both.save(summary(score=80.0, control=70.0, started=midday(0)))
    imp = both.improvement()
    check("improvement compares earliest to latest",
          imp is not None and imp["accuracy"] == (60.0, 80.0), imp)
    check("control improvement carried when both ends measured",
          imp is not None and imp["control"] == (50.0, 70.0), imp)

    mixed = ProgressStore(TMP / "mixed_imp.json")
    mixed.save(summary(score=60.0, control=None, started=midday(3)))
    mixed.save(summary(score=80.0, control=70.0, started=midday(0)))
    imp = mixed.improvement()
    check("control is None when one end is unmeasured",
          imp is not None and imp["control"] is None and imp["accuracy"] == (60.0, 80.0), imp)
    check("improvement filter respects the exercise",
          store.improvement(exercise="tree_pose") is None)

    # ------------------------------------------------------ weekly_summary
    empty_wk = ProgressStore(TMP / "wk_empty.json").weekly_summary()
    check("weekly summary with 0 sessions counts 0", empty_wk["sessions"] == 0, empty_wk)
    check("weekly summary with 0 sessions asks for more sessions",
          "more sessions" in empty_wk["note"].lower(), empty_wk["note"])
    check("weekly summary with 0 sessions claims no improvement",
          "improved" not in empty_wk["note"].lower(), empty_wk["note"])
    check("weekly averages honest at 0 sessions",
          empty_wk["avg_score"] is None and empty_wk["avg_control"] is None)

    one = ProgressStore(TMP / "wk_one.json")
    one.save(summary(score=88.0, control=90.0, started=midday(1)))
    one_wk = one.weekly_summary()
    check("weekly summary with 1 session asks for more sessions",
          one_wk["sessions"] == 1 and "more sessions" in one_wk["note"].lower(), one_wk["note"])

    two = ProgressStore(TMP / "wk_two.json")
    two.save(summary(score=60.0, control=50.0, started=midday(4),
                     metrics={"back_angle": 50.0, "knee_angle": 90.0}))
    two.save(summary(score=70.0, control=60.0, started=midday(0),
                     metrics={"back_angle": 70.0, "knee_angle": 90.0}))
    wk = two.weekly_summary()
    check("weekly summary counts window sessions", wk["sessions"] == 2, wk)
    check("weekly averages computed", wk["avg_score"] == 65.0 and wk["avg_control"] == 55.0, wk)
    check("weekly note claims the improvement the data shows",
          "movement control improved by 20%" in wk["note"].lower(), wk["note"])
    check("weekly note makes no medical claims",
          not any(w in wk["note"].lower()
                  for w in ("injur", "pain", "diagnos", "heal", "therap", "medical")),
          wk["note"])
    check("weekly focus is the weakest metric", wk["focus"] == "Back angle", wk["focus"])
    check("weekly most improved found", wk["most_improved"] == "Back angle",
          wk["most_improved"])

    flat = ProgressStore(TMP / "wk_flat.json")
    flat.save(summary(score=70.0, started=midday(3)))
    flat.save(summary(score=70.0, started=midday(0)))
    check("flat week claims no improvement",
          "improved" not in flat.weekly_summary()["note"].lower(),
          flat.weekly_summary()["note"])

    # ------------------------------------------------- old-format records
    old_path = TMP / "old.json"
    old_record = {"exercise": "squat", "exercise_name": "Squat", "started_at": midday(1),
                  "duration": 60.0, "movement_score": 72.0, "reps": 4, "good_reps": 3,
                  "hold_duration": 0.0, "corrections": 1, "successful_corrections": 1,
                  "metric_scores": {"knee_angle": 80.0}, "main_issue": "Back angle"}
    old_path.write_text(json.dumps([old_record]), encoding="utf-8")
    old = ProgressStore(old_path)
    check("old record loads without control keys", len(old.all()) == 1)
    check("old record shows None control in the week",
          old.last_7_days()[-2]["score"] == 72.0
          and old.last_7_days()[-2]["control"] is None, old.last_7_days()[-2])
    check("old record lists with None control",
          old.sessions_list()[0]["control"] is None, old.sessions_list()[0])
    check("old record aggregates with None control",
          old.exercise_progress()["squat"]["avg_control"] is None)
    revived = SessionSummary.from_dict(old_record)
    check("old record rebuilds as a summary",
          revived.control_score is None and revived.unstable_events == 0
          and revived.control_improved is None)
    check("old record weekly summary does not invent control",
          old.weekly_summary()["avg_control"] is None)

    # -------------------------------------------- control field round trip
    rt = summary(score=81.0, control=77.7, unstable=3, improved=True)
    payload = rt.to_dict()
    check("control fields serialise to JSON", isinstance(json.dumps(payload), str))
    back = SessionSummary.from_dict(payload)
    check("control score round-trips", back.control_score == 77.7, back.control_score)
    check("unstable events round-trip", back.unstable_events == 3, back.unstable_events)
    check("control improved round-trips", back.control_improved is True)
    check("headline mentions movement control",
          any("Movement control" in ln for ln in back.headline_lines()))
    rt_store = ProgressStore(TMP / "roundtrip.json")
    rt_store.save(rt)
    stored = ProgressStore(TMP / "roundtrip.json").all()[0]
    check("control fields survive storage and restart",
          stored["control_score"] == 77.7 and stored["unstable_events"] == 3
          and stored["control_improved"] is True, stored)

    # ------------------------------------------------------- corrupt file
    corrupt = TMP / "corrupt.json"
    corrupt.write_text('{"definitely": "not a list"', encoding="utf-8")
    broken = ProgressStore(corrupt)
    check("corrupt file: last_7_days still 7 empty days",
          len(broken.last_7_days()) == 7
          and all(d["score"] is None for d in broken.last_7_days()))
    check("corrupt file: sessions_list empty", broken.sessions_list() == [])
    check("corrupt file: get_session None", broken.get_session(0) is None)
    check("corrupt file: exercise_progress empty", broken.exercise_progress() == {})
    check("corrupt file: improvement None", broken.improvement() is None)
    wk = broken.weekly_summary()
    check("corrupt file: weekly summary honest",
          wk["sessions"] == 0 and "more sessions" in wk["note"].lower(), wk)

    # ----------------------------------------- recorder without the fields
    rec = SessionRecorder(exercise="squat", exercise_name="Squat")
    for i in range(10):
        rec.update(BareResult(score=70.0), now=i * 0.1)
    s = rec.finish()
    check("recorder tolerates results without control fields",
          s.control_score is None and s.unstable_events == 0
          and s.control_improved is None, (s.control_score, s.unstable_events))

    # -------------------------------------------- recorder with the fields
    rec = SessionRecorder(exercise="squat", exercise_name="Squat")
    for i in range(12):
        rec.update(ControlResult(score=70.0, control=40.0 + i * 4,
                                 unstable=min(i, 3)), now=i * 0.1)
    s = rec.finish()
    check("recorder means the control samples",
          s.control_score is not None and abs(s.control_score - 62.0) < 1e-6,
          s.control_score)
    check("recorder keeps the max unstable count", s.unstable_events == 3,
          s.unstable_events)
    check("recorder detects control improving", s.control_improved is True)

    rec = SessionRecorder(exercise="squat")
    for i in range(12):
        rec.update(ControlResult(score=70.0, control=80.0 - i * 4), now=i * 0.1)
    check("declining control is not called improvement",
          rec.finish().control_improved is False)

    rec = SessionRecorder(exercise="squat")
    for i in range(4):
        rec.update(ControlResult(score=70.0, control=60.0 + i * 10), now=i * 0.1)
    s = rec.finish()
    check("under 6 control samples: verdict withheld, mean still reported",
          s.control_improved is None and s.control_score is not None,
          (s.control_improved, s.control_score))

    rec = SessionRecorder(exercise="squat")
    rec.update(ControlResult(score=70.0, control="wobbly", unstable="lots"), now=0.0)
    s = rec.finish()
    check("garbage control values ignored, not crashed",
          s.control_score is None and s.unstable_events == 0)

finally:
    shutil.rmtree(TMP, ignore_errors=True)

print()
if failures:
    print("FAILURES: " + ", ".join(failures))
    sys.exit(1)
print("FAILURES: none")
