"""Verify every sample clip drives the analysis stack to the behaviour it advertises."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.engine import PoseEngine
from core.synthetic import CLIPS, SyntheticSource
from exercises.registry import ExerciseRegistry, ExerciseRecognizer

failures = []
def check(name, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + name + ("  " + str(extra) if extra else ""))
    if not cond: failures.append(name)

def run(clip, exercise_key=None, auto=False):
    """Play a clip once through the engine + profile. Returns collected stats."""
    engine = PoseEngine()
    engine.calibrator.duration = 0.5
    reg = ExerciseRegistry()
    rec = ExerciseRecognizer(reg)
    profile = reg.get(exercise_key) if exercise_key else None

    src = SyntheticSource(clip, loop=False)
    src.open()
    devs, phases, last, detected_key = set(), set(), None, None
    frames = 0
    while True:
        item = src.read()
        if item is None or item.finished:
            break
        frames += 1
        f = engine.process_pose(item.pose)
        if auto:
            r = rec.observe(f)
            if r.confident and r.key and detected_key is None:
                detected_key = r.key
        if profile is None or f.calibrating:
            continue
        res = profile.analyse(f)
        if res.ready:
            last = res
            phases.add(res.phase)
            for d in res.errors:
                devs.add(d.key)
    return dict(frames=frames, devs=devs, phases=phases, last=last, detected=detected_key)

print("=== clip playback ===")
for clip in CLIPS:
    out = run(clip, clip.exercise_key)
    ok = out["last"] is not None and out["frames"] > 50
    check(f"{clip.key}: plays and analyses", ok,
          f"{out['frames']} frames, phases={sorted(out['phases'])}")

print()
print("=== each clip does what its caption claims ===")

# squat_good: counts reps, and must NOT report knees caving
g = run(next(c for c in CLIPS if c.key=="squat_good"), "squat")
check("squat_good counts 2+ reps", g["last"].rep_count >= 2, g["last"].rep_count)
check("squat_good reaches bottom phase", "bottom" in g["phases"], sorted(g["phases"]))
check("squat_good does NOT flag valgus",
      not any("valgus" in d for d in g["devs"]), sorted(g["devs"]))

# squat_valgus: same reps, but valgus IS flagged
v = run(next(c for c in CLIPS if c.key=="squat_valgus"), "squat")
check("squat_valgus counts 2+ reps", v["last"].rep_count >= 2, v["last"].rep_count)
check("squat_valgus flags knee valgus",
      any("valgus" in d for d in v["devs"]), sorted(v["devs"]))

# curl: counts reps on both arms
c = run(next(c for c in CLIPS if c.key=="curl"), "bicep_curl")
lr = c["last"].metric("left_reps").value, c["last"].metric("right_reps").value
check("curl counts 3+ reps", c["last"].rep_count >= 3, c["last"].rep_count)
check("curl counts both arms", lr[0] >= 3 and lr[1] >= 3, lr)

# tree: enters balancing phase, and sway grows enough to be flagged late
t = run(next(c for c in CLIPS if c.key=="tree"), "tree_pose")
check("tree reaches holding phase", "holding" in t["phases"], sorted(t["phases"]))
check("tree accrues hold time", t["last"].hold_duration > 3.0, round(t["last"].hold_duration,1))

# warrior: holds, then the arm droop is caught
w = run(next(c for c in CLIPS if c.key=="warrior"), "warrior_2")
check("warrior reaches holding phase", "holding" in w["phases"], sorted(w["phases"]))
check("warrior catches the arm droop", "arms_dropping" in w["devs"], sorted(w["devs"]))

print()
print("=== auto-detect on the clips ===")
for clip in CLIPS:
    out = run(clip, clip.exercise_key, auto=True)
    check(f"auto-detect {clip.key} -> {clip.exercise_key}",
          out["detected"] == clip.exercise_key, f"got {out['detected']}")

print()
print("FAILURES:", failures if failures else "none")
sys.exit(1 if failures else 0)
