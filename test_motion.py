"""Stability & Controlled Movement Coach — tested without a camera.

Drives MotionAnalyzer through scripted clips and hand-built frames, and checks
the control score, the candidate errors, the confidence discipline (no strong
judgment from low-visibility frames or single-frame glitches), episode counting
and the improvement read-out. Also proves the movement_control metric never
changes the pre-existing accuracy score.
"""
import math
import sys
from dataclasses import fields as dc_fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.engine import EngineFrame, PoseEngine
from core.landmarks import LM, Landmark, PoseFrame
from core.motion import PROFILES, MotionAnalyzer
from core.synthetic import CLIPS_BY_KEY, STANDING, SyntheticSource, to_pose_frame
from exercises.base import ExerciseResult
from exercises.registry import ExerciseRegistry

failures = []

MOTION_KEYS = {"too_fast", "jerky_movement", "torso_sway"}


def check(name, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + name + ("  " + str(extra) if extra else ""))
    if not cond:
        failures.append(name)


def play(clip_key, exercise_key):
    """Run a clip through engine + profile; collect everything the tests need."""
    engine = PoseEngine()
    engine.calibrator.duration = 0.4
    reg = ExerciseRegistry()
    profile = reg.get(exercise_key)

    src = SyntheticSource(CLIPS_BY_KEY[clip_key], loop=False)
    src.open()
    last, err_keys, scores = None, set(), []
    while True:
        item = src.read()
        if item is None or item.finished:
            break
        f = engine.process_pose(item.pose)
        if f.calibrating:
            continue
        r = profile.analyse(f)
        if not r.ready:
            continue
        last = r
        for e in r.errors:
            err_keys.add(e.key)
        if r.score is not None:
            scores.append(r.score)
    return dict(last=last, errors=err_keys, scores=scores, profile=profile)


def frame_at(joints, i, t):
    """EngineFrame straight from ground-truth joints — MotionAnalyzer needs no angles."""
    return EngineFrame(pose=to_pose_frame(joints, i, t))


def shifted(dx):
    return {lm: (x + dx, y, z) for lm, (x, y, z) in STANDING.items()}


# =====================================================================
print("=== 1. controlled clip: high control score, no stability errors ===")
good = play("squat_good", "squat")
check("squat_good produces a control score", good["last"].control is not None,
      good["last"].control)
check("squat_good control score >= 75", (good["last"].control or 0) >= 75,
      round(good["last"].control or -1, 1))
check("squat_good surfaces no stability errors",
      not (good["errors"] & MOTION_KEYS), sorted(good["errors"] & MOTION_KEYS))


# =====================================================================
print()
print("=== 2. rushed clip: lower control, too_fast/jerky_movement surfaces ===")
fast = play("squat_fast", "squat")
check("squat_fast produces a control score", fast["last"].control is not None,
      fast["last"].control)
check("squat_fast control clearly lower than squat_good",
      fast["last"].control is not None
      and fast["last"].control <= (good["last"].control or 0) - 15,
      (round(fast["last"].control or -1, 1), round(good["last"].control or -1, 1)))
check("squat_fast surfaces too_fast or jerky_movement (through the debounce)",
      bool(fast["errors"] & {"too_fast", "jerky_movement"}), sorted(fast["errors"]))


# =====================================================================
print()
print("=== 3. one-frame teleport: window means, never single frames ===")
ma = MotionAnalyzer("squat")
fired = []
n_still = 3 * PROFILES["squat"].window
for i in range(n_still):
    ma.update(frame_at(STANDING, i, i / 30.0))
    fired.extend(ma.candidates("standing", i / 30.0))
check("still pose fills the window", ma.window_full, len(ma._speeds))
# whole body jumps 0.4 m (~0.8 torso-lengths) for exactly one frame
ma.update(frame_at(shifted(0.40), n_still, n_still / 30.0))
fired.extend(ma.candidates("standing", n_still / 30.0))
for i in range(n_still + 1, n_still + 40):
    ma.update(frame_at(STANDING, i, i / 30.0))
    fired.extend(ma.candidates("standing", i / 30.0))
check("one-frame teleport never fires a candidate", fired == [],
      [e.key for e in fired])
check("teleport does not wreck the control score",
      ma.control_score() is not None and ma.control_score() >= 90,
      ma.control_score())


# =====================================================================
print()
print("=== 4. low visibility: no strong judgment ===")
ma = MotionAnalyzer("squat")
cands = []
for i in range(80):
    t = i / 30.0
    # violent whole-body oscillation, but every landmark is low-confidence
    joints = shifted(0.15 * math.sin(2.0 * math.pi * 2.5 * t))
    lms = {int(lm): Landmark(idx=int(lm), x=0.5 + x, y=0.5 + y, z=z,
                             visibility=0.3, wx=x, wy=y, wz=z)
           for lm, (x, y, z) in joints.items()}
    f = EngineFrame(pose=PoseFrame(landmarks=lms, frame_index=i, timestamp=t,
                                   detected=True))
    ma.update(f)
    cands.extend(ma.candidates("standing", t))
check("low-visibility frames produce no candidates", cands == [],
      [e.key for e in cands])
check("low-visibility frames leave control unmeasured (None)",
      ma.control_score() is None, ma.control_score())


# =====================================================================
print()
print("=== 5. at most one motion candidate per frame ===")
engine = PoseEngine()
engine.calibrator.duration = 0.4
ma = MotionAnalyzer("squat")
src = SyntheticSource(CLIPS_BY_KEY["squat_fast"], loop=False)
src.open()
max_per_frame, frames_with_candidate = 0, 0
while True:
    item = src.read()
    if item is None or item.finished:
        break
    f = engine.process_pose(item.pose)
    if f.calibrating:
        continue
    ma.update(f)
    got = ma.candidates("descending", f.timestamp)
    max_per_frame = max(max_per_frame, len(got))
    if got:
        frames_with_candidate += 1
check("candidates fire on the rushed clip at all", frames_with_candidate > 0,
      frames_with_candidate)
check("never more than one candidate per frame", max_per_frame <= 1, max_per_frame)


# =====================================================================
print()
print("=== 6. static profile tolerates its clip's natural sway ===")
tree = play("tree", "tree_pose")
check("tree pose clip never flagged too_fast", "too_fast" not in tree["errors"],
      sorted(tree["errors"]))
check("tree pose keeps a decent control score",
      tree["last"].control is None or tree["last"].control >= 75,
      tree["last"].control)


# =====================================================================
print()
print("=== 7. movement_control metric: visible, weightless, score untouched ===")
last = good["last"]
mc = last.metric("movement_control")
check("movement_control metric is present", mc is not None)
check("movement_control sits at the front of the metrics list",
      last.metrics and last.metrics[0].key == "movement_control",
      [m.key for m in last.metrics[:3]])
check("movement_control carries weight 0 (never moves the accuracy score)",
      mc is not None and mc.weight == 0.0, None if mc is None else mc.weight)
mean_good = sum(good["scores"]) / len(good["scores"])
check("squat_good clean-clip mean score unchanged (88.5 +/- 2)",
      abs(mean_good - 88.5) <= 2.0, round(mean_good, 2))


# =====================================================================
print()
print("=== 8. unstable episodes counted for the session summary ===")
check("squat_fast counts at least one unstable episode",
      fast["last"].unstable_events > 0, fast["last"].unstable_events)
check("squat_good counts zero unstable episodes",
      good["last"].unstable_events == 0, good["last"].unstable_events)


# =====================================================================
print()
print("=== 9. improved(): first third vs last third ===")
ma = MotionAnalyzer("squat")
check("improved() is None without enough data", ma.improved() is None)
for i in range(240):
    t = i / 30.0
    if i < 120:      # rushed: big fast whole-body oscillation
        dx = 0.15 * math.sin(2.0 * math.pi * 2.5 * t)
    else:            # settled: barely moving
        dx = 0.01 * math.sin(2.0 * math.pi * 0.5 * t)
    ma.update(frame_at(shifted(dx), i, t))
check("starts-fast-ends-slow session reads as improved", ma.improved() is True,
      ma.improved())

ma2 = MotionAnalyzer("squat")
for i in range(240):
    t = i / 30.0
    if i < 120:      # settled first...
        dx = 0.01 * math.sin(2.0 * math.pi * 0.5 * t)
    else:            # ...then falls apart
        dx = 0.15 * math.sin(2.0 * math.pi * 2.5 * t)
    ma2.update(frame_at(shifted(dx), i, t))
check("starts-slow-ends-fast session does NOT read as improved",
      ma2.improved() is False, ma2.improved())


# =====================================================================
print()
print("=== 10. ExerciseResult carries the new fields ===")
names = {f.name for f in dc_fields(ExerciseResult)}
check("ExerciseResult has a control field", "control" in names)
check("ExerciseResult has an unstable_events field", "unstable_events" in names)
blank = ExerciseResult(exercise="x")
check("control defaults to None", blank.control is None, blank.control)
check("unstable_events defaults to 0", blank.unstable_events == 0,
      blank.unstable_events)
check("reset gives every profile a fresh MotionAnalyzer",
      all(isinstance(p.motion, MotionAnalyzer) for p in ExerciseRegistry().all()))


print()
print("FAILURES:", failures if failures else "none")
sys.exit(1 if failures else 0)
