"""Phase 2 — exercise analysis layer, tested per exercise and independently.

Drives each analyzer through scripted clips and hand-built poses, and checks the
structured result, phase machines, rep/hold detection, error priority, debounce,
scoring and the ghost overlay.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.engine import PoseEngine
from core.errors import ErrorTracker, FormError, Priority, Severity
from core.landmarks import LM, MetricStatus
from core.phases import HoldStateMachine, PhaseNames, RepPhaseMachine
from core.reference import correction_arrow, fit_reference
from core.scoring import Metric, StabilityTracker, aggregate_score
from core.synthetic import CLIPS_BY_KEY, SyntheticSource
from exercises.registry import ExerciseRegistry

failures = []


def check(name, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + name + ("  " + str(extra) if extra else ""))
    if not cond:
        failures.append(name)


def play(clip_key, exercise_key, frames=None):
    """Run a clip through engine + profile; return (final result, seen state)."""
    engine = PoseEngine()
    engine.calibrator.duration = 0.4
    reg = ExerciseRegistry()
    profile = reg.get(exercise_key)

    src = SyntheticSource(CLIPS_BY_KEY[clip_key], loop=False)
    src.open()
    last, phases, err_keys, scores, primaries = None, set(), set(), [], set()
    n = 0
    while True:
        item = src.read()
        if item is None or item.finished:
            break
        n += 1
        if frames and n > frames:
            break
        f = engine.process_pose(item.pose)
        if f.calibrating:
            continue
        r = profile.analyse(f)
        if not r.ready:
            continue
        last = r
        phases.add(r.phase)
        for e in r.errors:
            err_keys.add(e.key)
        if r.primary_error:
            primaries.add(r.primary_error.key)
        if r.score is not None:
            scores.append(r.score)
    return dict(last=last, phases=phases, errors=err_keys, scores=scores,
                primaries=primaries, profile=profile, engine=engine, frames=n)


# =====================================================================
print("=== structured result contract ===")
REQUIRED = ["exercise", "phase", "score", "rep_count", "hold_duration",
            "metrics", "errors", "primary_error", "confidence"]
for key, ex in [("squat_good", "squat"), ("curl", "bicep_curl"),
                ("tree", "tree_pose"), ("warrior", "warrior_2")]:
    out = play(key, ex)
    r = out["last"]
    check(f"{ex}: returns a result", r is not None)
    missing = [f for f in REQUIRED if not hasattr(r, f)]
    check(f"{ex}: has all required fields", not missing, missing)
    check(f"{ex}: exercise field matches", r.exercise == ex, r.exercise)
    check(f"{ex}: confidence in 0..1", 0.0 <= r.confidence <= 1.0, r.confidence)
    check(f"{ex}: score in 0..100", r.score is None or 0 <= r.score <= 100, r.score)


# =====================================================================
print()
print("=== SQUAT — five-phase cycle, reps, phase-aware errors ===")
g = play("squat_good", "squat")
check("squat visits all four phases",
      {"standing", "descending", "bottom", "ascending"} <= g["phases"], sorted(g["phases"]))
check("squat counts 3 reps", g["last"].rep_count == 3, g["last"].rep_count)
check("squat clean clip has no valgus", not any("valgus" in k for k in g["errors"]),
      sorted(g["errors"]))
check("squat clean clip scores well", max(g["scores"]) >= 80, round(max(g["scores"]), 1))
check("squat reports consistency after reps", g["last"].quality is not None, g["last"].quality)
check("squat exposes depth metric", g["last"].metric("depth") is not None)
check("squat exposes back angle metric", g["last"].metric("back_angle") is not None)
check("squat exposes symmetry metric", g["last"].metric("symmetry") is not None)

v = play("squat_valgus", "squat")
check("squat valgus clip flags knee tracking",
      any("valgus" in k for k in v["errors"]), sorted(v["errors"]))
check("valgus is the primary error (SAFETY outranks the rest)",
      any("valgus" in k for k in v["primaries"]), sorted(v["primaries"]))
# Compare means, not maxima: both clips score 100 at the top of the rep where
# the knees are fine, so only the average reflects the fault.
mean_v = sum(v["scores"]) / len(v["scores"])
mean_g = sum(g["scores"]) / len(g["scores"])
check("valgus clip scores lower than clean clip", mean_v < mean_g - 5,
      (round(mean_v, 1), round(mean_g, 1)))

# valgus must only fire under load, never while standing
sq = v["profile"]
standing_errors = [e for e in (v["last"].errors or []) if e.phase == "standing"
                   and "valgus" in e.key]
check("valgus never attributed to the standing phase", not standing_errors)


# =====================================================================
print()
print("=== BICEP CURL — start/curl/peak/return, per-arm reps ===")
c = play("curl", "bicep_curl")
check("curl visits all four phases",
      {"start", "curl", "peak", "return"} <= c["phases"], sorted(c["phases"]))
check("curl counts 4 reps", c["last"].rep_count == 4, c["last"].rep_count)
check("curl counts each arm independently",
      c["last"].metric("left_reps").value == 4 and c["last"].metric("right_reps").value == 4,
      (c["last"].metric("left_reps").value, c["last"].metric("right_reps").value))
check("curl exposes range-of-motion metric", c["last"].metric("rom") is not None)
check("curl exposes elbow stability metric", c["last"].metric("elbow_stability") is not None)
check("curl reports consistency", c["last"].quality is not None, c["last"].quality)


# =====================================================================
print()
print("=== TREE POSE — hold detection, stability, tolerant of small movement ===")
t = play("tree", "tree_pose")
check("tree reaches holding phase", "holding" in t["phases"], sorted(t["phases"]))
check("tree accrues hold duration", t["last"].hold_duration > 3.0,
      round(t["last"].hold_duration, 1))
check("tree reports hold quality", t["last"].quality is not None, t["last"].quality)
check("tree reports stability", t["last"].stability is not None, t["last"].stability)
check("tree exposes hip + shoulder level metrics",
      t["last"].metric("hip_tilt") is not None and t["last"].metric("shoulder_tilt") is not None)
check("tree exposes standing-leg metric", t["last"].metric("standing_leg") is not None)

# The clip sways progressively; small early movement must NOT be punished.
early = play("tree", "tree_pose", frames=90)
check("small natural sway is not flagged early",
      "unstable" not in early["errors"], sorted(early["errors"]))


# =====================================================================
print()
print("=== WARRIOR II — multi-axis hold ===")
w = play("warrior", "warrior_2")
check("warrior reaches holding phase", "holding" in w["phases"], sorted(w["phases"]))
check("warrior accrues hold duration", w["last"].hold_duration >= 0.0)
check("warrior catches the arm droop", "arms_dropping" in w["errors"], sorted(w["errors"]))
check("warrior exposes all alignment metrics",
      all(w["last"].metric(k) is not None for k in
          ("front_knee", "back_leg", "arm_level", "arm_height",
           "shoulder_tilt", "hip_tilt", "torso_lean", "stability")))
check("warrior reports hold quality", w["last"].quality is not None, w["last"].quality)


# =====================================================================
print()
print("=== error priority + debounce ===")
tracker = ErrorTracker(on_frames=5, off_frames=10)
minor = FormError(key="tidy", message="m", cue="M", severity=Severity.MINOR,
                  priority=Priority.REFINEMENT)
safety = FormError(key="knee", message="s", cue="S", severity=Severity.MAJOR,
                   priority=Priority.SAFETY)

for i in range(4):
    active = tracker.update([minor, safety], float(i))
check("error suppressed below the debounce threshold", active == [], active)

for i in range(4, 8):
    active = tracker.update([minor, safety], float(i))
check("error surfaces once it persists", len(active) == 2, len(active))
check("SAFETY outranks REFINEMENT", tracker.primary(active).key == "knee",
      tracker.primary(active).key)

for i in range(8, 14):
    active = tracker.update([], float(i))
check("error still held 6 frames after it vanished (slow to forget)",
      len(active) == 2, len(active))
for i in range(14, 22):
    active = tracker.update([], float(i))
check("error clears once off_frames is exceeded", active == [], active)

# a single-frame blip must never reach the user
tracker2 = ErrorTracker(on_frames=5, off_frames=10)
tracker2.update([safety], 0.0)
for i in range(1, 6):
    a2 = tracker2.update([], float(i))
check("one-frame blip never surfaces", a2 == [], a2)


# =====================================================================
print()
print("=== scoring ===")
inside = Metric("a", "A", 90.0, MetricStatus.MEASURED, lo=80, hi=100)
outside = Metric("b", "B", 120.0, MetricStatus.MEASURED, lo=80, hi=100, falloff=40)
far = Metric("e", "E", 140.0, MetricStatus.MEASURED, lo=80, hi=100, falloff=40)
unseen = Metric("c", "C", None, MetricStatus.UNMEASURED, lo=80, hi=100)
na = Metric("d", "D", None, MetricStatus.NOT_APPLICABLE, lo=80, hi=100)

check("metric inside band scores 100", inside.score() == 100.0, inside.score())
check("metric outside band scores partial", outside.score() == 50.0, outside.score())
check("metric a full falloff past the band scores 0", far.score() == 0.0, far.score())
check("unmeasured metric is unscored", unseen.score() is None)
check("not-applicable metric is unscored", na.score() is None)
check("aggregate ignores unscorable metrics",
      aggregate_score([inside, unseen, na]) == 100.0, aggregate_score([inside, unseen, na]))
check("aggregate applies penalties",
      aggregate_score([inside], penalties=20.0) == 80.0,
      aggregate_score([inside], penalties=20.0))
check("aggregate of nothing scorable is None", aggregate_score([unseen, na]) is None)


# =====================================================================
print()
print("=== phase machines in isolation ===")
pm = RepPhaseMachine(160, 115, PhaseNames("top", "down", "bottom", "up"),
                     min_rep_seconds=0.0)
t0 = 0.0
for value in [175, 170, 150, 130, 110, 100, 110, 130, 150, 170, 175]:
    pm.update(float(value), t0)
    t0 += 0.1
check("rep machine counts one full cycle", pm.rep_count == 1, pm.rep_count)
check("rep machine records depth", pm.last_rep.peak <= 100.0, pm.last_rep.peak)
check("depth progress is 0 at the top", pm.depth_progress < 0.05, pm.depth_progress)

# partial rep (never reaches bottom) must not count
pm2 = RepPhaseMachine(160, 115, PhaseNames("top", "down", "bottom", "up"),
                      min_rep_seconds=0.0)
t0 = 0.0
for value in [175, 165, 150, 140, 150, 165, 175]:
    pm2.update(float(value), t0)
    t0 += 0.1
check("partial rep is not counted", pm2.rep_count == 0, pm2.rep_count)

hm = HoldStateMachine(enter_frames=3, exit_frames=5)
t0 = 0.0
for _ in range(6):
    hm.update(True, t0); t0 += 0.1
check("hold engages after enter_frames", hm.phase == "holding", hm.phase)
check("hold duration accrues", hm.duration(t0) > 0.2, round(hm.duration(t0), 2))
for _ in range(3):
    hm.update(False, t0); t0 += 0.1
check("brief wobble does not break the hold", hm.phase == "holding", hm.phase)
for _ in range(6):
    hm.update(False, t0); t0 += 0.1
check("sustained loss does break the hold", hm.phase == "setting up", hm.phase)


# =====================================================================
print()
print("=== ghost coach / reference overlay ===")
engine = PoseEngine()
engine.calibrator.duration = 0.4
src = SyntheticSource(CLIPS_BY_KEY["squat_good"], loop=False)
src.open()
shape = (720, 720, 3)
ghost_top = ghost_bottom = None
pose_bottom = None
for _ in range(120):
    item = src.read()
    if item is None or item.finished:
        break
    f = engine.process_pose(item.pose)
    if f.calibrating:
        continue
    if ghost_top is None:
        ghost_top = fit_reference("squat", f.pose, shape, progress=0.0)
    ghost_bottom = fit_reference("squat", f.pose, shape, progress=1.0)
    pose_bottom = f.pose

check("ghost is produced", ghost_top is not None and ghost_bottom is not None)
check("ghost covers the main joints",
      all(lm in ghost_top for lm in (LM.LEFT_KNEE, LM.RIGHT_KNEE, LM.LEFT_SHOULDER)))
check("ghost tracks the rep (top differs from bottom)",
      ghost_top[LM.LEFT_KNEE] != ghost_bottom[LM.LEFT_KNEE],
      (ghost_top[LM.LEFT_KNEE], ghost_bottom[LM.LEFT_KNEE]))

for ex in ("squat", "bicep_curl", "tree_pose", "warrior_2"):
    gh = fit_reference(ex, pose_bottom, shape, progress=0.5)
    check(f"ghost available for {ex}", gh is not None and len(gh) > 10)

check("no ghost for an unknown exercise",
      fit_reference("nope", pose_bottom, shape) is None)

# ghost must scale to the user, not sit at a fixed size
import copy
from core.landmarks import Landmark, PoseFrame
small = PoseFrame(landmarks={
    i: Landmark(idx=i, x=0.5 + (lm.x - 0.5) * 0.5, y=0.5 + (lm.y - 0.5) * 0.5,
                z=lm.z, visibility=lm.visibility, wx=lm.wx, wy=lm.wy, wz=lm.wz)
    for i, lm in pose_bottom.landmarks.items()}, detected=True)
big_ghost = fit_reference("squat", pose_bottom, shape, progress=0.5)
small_ghost = fit_reference("squat", small, shape, progress=0.5)


def span(g):
    return abs(g[LM.LEFT_SHOULDER][1] - g[LM.LEFT_ANKLE][1])


check("ghost scales to the user's size", span(small_ghost) < span(big_ghost),
      (span(small_ghost), span(big_ghost)))

arrow = correction_arrow(pose_bottom, ghost_bottom, LM.LEFT_KNEE, shape, min_pixels=0.0)
check("correction arrow is produced", arrow is not None)
check("arrow ends at the reference joint",
      arrow is not None and arrow[1] == ghost_bottom[LM.LEFT_KNEE])
near = correction_arrow(pose_bottom, ghost_bottom, LM.LEFT_KNEE, shape, min_pixels=100000)
check("no arrow when already close enough", near is None)


# =====================================================================
print()
print("=== independence: analyzers do not leak state into each other ===")
reg = ExerciseRegistry()
sq, cu = reg.get("squat"), reg.get("bicep_curl")
eng = PoseEngine()
eng.calibrator.duration = 0.4
src = SyntheticSource(CLIPS_BY_KEY["squat_good"], loop=False)
src.open()
while True:
    item = src.read()
    if item is None or item.finished:
        break
    f = eng.process_pose(item.pose)
    if f.calibrating:
        continue
    sq.analyse(f)
    cu.analyse(f)
check("squat counted reps on the squat clip", sq.reps >= 2, sq.reps)
check("curl did not count squat reps as curls", cu.reps == 0, cu.reps)

sq.reset()
check("reset clears reps", sq.reps == 0)
check("reset clears the error tracker", sq.tracker.active_keys() == [])


print()
print("FAILURES:", failures if failures else "none")
sys.exit(1 if failures else 0)
