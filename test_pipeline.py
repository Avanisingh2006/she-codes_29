"""Smoke test: drive the whole pipeline with synthetic poses, no camera."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.landmarks import LM, Landmark, PoseFrame, MetricStatus
from core.engine import PoseEngine, EngineFrame, COMMON_ANGLES
from core.smoothing import PoseSmoother
from core.bodymap import BodyMapCalibrator, BodyMode, BodyMap
from core.geometry import joint_angle, angle_to_vertical, angle_to_horizontal
from exercises.registry import ExerciseRegistry, ExerciseRecognizer

failures = []
def check(name, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + name + ("  " + str(extra) if extra else ""))
    if not cond: failures.append(name)

def mk(points, vis=0.95, t=0.0):
    """points: {LM: (wx,wy,wz)} in metres, y down."""
    lms = {}
    for lm, (wx, wy, wz) in points.items():
        lms[int(lm)] = Landmark(idx=int(lm), x=0.5+wx, y=0.5+wy, z=wz,
                                visibility=vis, wx=wx, wy=wy, wz=wz)
    return PoseFrame(landmarks=lms, frame_index=1, timestamp=t, detected=True)

# ---------------------------------------------------------------- geometry
# Straight leg: hip(0,-0.4) knee(0,0) ankle(0,0.4) -> 180 deg
f = mk({LM.LEFT_HIP:(0,-0.4,0), LM.LEFT_KNEE:(0,0,0), LM.LEFT_ANKLE:(0,0.4,0)})
a = joint_angle(f, LM.LEFT_HIP, LM.LEFT_KNEE, LM.LEFT_ANKLE)
check("straight leg = 180", a is not None and abs(a-180) < 1, f"got {a}")

# Right angle: hip(0,-0.4) knee(0,0) ankle(0.4,0) -> 90 deg
f = mk({LM.LEFT_HIP:(0,-0.4,0), LM.LEFT_KNEE:(0,0,0), LM.LEFT_ANKLE:(0.4,0,0)})
a = joint_angle(f, LM.LEFT_HIP, LM.LEFT_KNEE, LM.LEFT_ANKLE)
check("bent knee = 90", a is not None and abs(a-90) < 1, f"got {a}")

# Missing landmark -> None, not a crash
f = mk({LM.LEFT_HIP:(0,-0.4,0), LM.LEFT_KNEE:(0,0,0)})
check("missing landmark -> None", joint_angle(f, LM.LEFT_HIP, LM.LEFT_KNEE, LM.LEFT_ANKLE) is None)

# vertical / horizontal
f = mk({LM.LEFT_SHOULDER:(0,-0.5,0), LM.LEFT_HIP:(0,0,0)})
check("upright torso ~0 off vertical", abs(angle_to_vertical(f, LM.LEFT_SHOULDER, LM.LEFT_HIP)) < 1)
f = mk({LM.LEFT_WRIST:(-0.7,0,0), LM.RIGHT_WRIST:(0.7,0,0)})
check("level arms ~0 off horizontal", abs(angle_to_horizontal(f, LM.LEFT_WRIST, LM.RIGHT_WRIST)) < 1)

# ---------------------------------------------------------------- smoothing
sm = PoseSmoother()
for i in range(5):
    sm.apply(mk({LM.LEFT_KNEE:(0,0,0)}, t=i/30))
# now the landmark vanishes -> should coast, not disappear
coasted = sm.apply(PoseFrame(landmarks={}, frame_index=99, timestamp=6/30, detected=True))
check("coasts through dropout", int(LM.LEFT_KNEE) in coasted.landmarks)
for i in range(20):
    coasted = sm.apply(PoseFrame(landmarks={}, frame_index=i, timestamp=(7+i)/30, detected=True))
check("gives up after MAX_STALE_FRAMES", int(LM.LEFT_KNEE) not in coasted.landmarks)

# ---------------------------------------------------------------- body map
FULL = {
 LM.LEFT_SHOULDER:(-0.2,-0.5,0), LM.RIGHT_SHOULDER:(0.2,-0.5,0),
 LM.LEFT_ELBOW:(-0.22,-0.2,0),   LM.RIGHT_ELBOW:(0.22,-0.2,0),
 LM.LEFT_WRIST:(-0.24,0.1,0),    LM.RIGHT_WRIST:(0.24,0.1,0),
 LM.LEFT_HIP:(-0.15,0,0),        LM.RIGHT_HIP:(0.15,0,0),
 LM.LEFT_KNEE:(-0.15,0.45,0),    LM.RIGHT_KNEE:(0.15,0.45,0),
 LM.LEFT_ANKLE:(-0.15,0.9,0),    LM.RIGHT_ANKLE:(0.15,0.9,0),
}
cal = BodyMapCalibrator(duration=0.2)
for i in range(30):
    cal.observe(mk(FULL, t=i*0.02))
bm = cal.build()
check("full body -> STANDARD", bm.mode is BodyMode.STANDARD, bm.mode)

partial = {k:v for k,v in FULL.items() if k not in (LM.LEFT_KNEE, LM.LEFT_ANKLE)}
cal2 = BodyMapCalibrator(duration=0.2)
for i in range(30):
    cal2.observe(mk(partial, t=i*0.02))
bm2 = cal2.build()
check("missing leg -> ADAPTIVE", bm2.mode is BodyMode.ADAPTIVE, bm2.mode)
check("adaptive marks metric N/A",
      bm2.status_for(LM.LEFT_HIP, LM.LEFT_KNEE, LM.LEFT_ANKLE) is MetricStatus.NOT_APPLICABLE)

# ---------------------------------------------------------------- engine angles
eng = PoseEngine(auto_calibrate=False)
eng.force_body_map(bm)
ef = EngineFrame(pose=mk(FULL), angles=eng._compute_angles(mk(FULL)), body_map=bm)
check("engine computes all common angles", len(ef.angles) == len(COMMON_ANGLES), len(ef.angles))
check("straight knee measured", ef.status("left_knee") is MetricStatus.MEASURED)

# adaptive engine must mark the untrackable leg N/A, never 'fine'
eng2 = PoseEngine(auto_calibrate=False); eng2.force_body_map(bm2)
ang2 = eng2._compute_angles(mk(FULL))
check("N/A when body map lacks joint", ang2["left_knee"].status is MetricStatus.NOT_APPLICABLE)
check("N/A angle is not usable", not ang2["left_knee"].usable)

# low visibility -> UNMEASURED, not measured
ang3 = eng._compute_angles(mk(FULL, vis=0.2))
check("low visibility -> UNMEASURED", ang3["left_knee"].status is MetricStatus.UNMEASURED)

# ---------------------------------------------------------------- profiles
reg = ExerciseRegistry()
check("four exercises registered", len(reg.all()) == 4, reg.keys)

def engframe(points, vis=0.95, t=0.0, body_map=bm):
    p = mk(points, vis=vis, t=t)
    e = PoseEngine(auto_calibrate=False); e.force_body_map(body_map)
    return EngineFrame(pose=p, angles=e._compute_angles(p), body_map=body_map)

# --- squat: stand -> bottom -> stand should count 1 rep
squat = reg.get("squat")
stand = dict(FULL)
bottom = dict(FULL); bottom[LM.LEFT_KNEE]=(-0.3,0.45,0.25); bottom[LM.RIGHT_KNEE]=(0.3,0.45,0.25)
bottom[LM.LEFT_HIP]=(-0.15,0.25,0); bottom[LM.RIGHT_HIP]=(0.15,0.25,0)
# Timestamps must advance: the phase machine rejects reps faster than
# min_rep_seconds, which is what stops jitter double-counting.
for i, pts in enumerate((stand, bottom, bottom, stand)):
    r = squat.analyse(engframe(pts, t=i * 0.5))
check("squat analyses without crash", r.ready)
check("squat counted a rep", squat.reps >= 1, squat.reps)

# --- warrior II
w2 = reg.get("warrior_2")
# Arms wide and level, wide stance, front knee ~104 deg, back leg straight.
w2pts = dict(FULL)
w2pts[LM.LEFT_WRIST]=(-0.85,-0.5,0); w2pts[LM.RIGHT_WRIST]=(0.85,-0.5,0)
w2pts[LM.LEFT_HIP]=(-0.15,0.15,0);   w2pts[LM.RIGHT_HIP]=(0.15,0.15,0)
w2pts[LM.LEFT_KNEE]=(-0.55,0.25,0);  w2pts[LM.LEFT_ANKLE]=(-0.55,0.9,0)
w2pts[LM.RIGHT_KNEE]=(0.45,0.52,0);  w2pts[LM.RIGHT_ANKLE]=(0.75,0.9,0)
r = w2.analyse(engframe(w2pts))
check("warrior II analyses", r.ready)
check("warrior II gives metrics", len(r.metrics) >= 6, len(r.metrics))

# --- tree pose
tree = reg.get("tree_pose")
tpts = dict(FULL); tpts[LM.RIGHT_ANKLE]=(0.05,0.45,0)   # right foot raised
for i in range(15):
    r = tree.analyse(engframe(tpts, t=i*0.03))
check("tree pose analyses", r.ready)
check("tree pose detects balance phase", r.phase in ("holding","setting up"), r.phase)

# --- bicep curl
curl = reg.get("bicep_curl")
down = dict(FULL)
up = dict(FULL); up[LM.LEFT_WRIST]=(-0.26,-0.45,0); up[LM.RIGHT_WRIST]=(0.26,-0.45,0)
for i, pts in enumerate((down, up, up, down)):
    r = curl.analyse(engframe(pts, t=i * 0.5))
check("bicep curl analyses", r.ready)
check("bicep curl counted a rep", curl.reps >= 1, curl.reps)

# --- every profile survives an empty frame
empty = EngineFrame(pose=PoseFrame(detected=False), angles={}, body_map=bm)
ok = True
for p in reg.all():
    try:
        res = p.analyse(empty)
        if res.ready: ok = False
    except Exception as e:
        ok = False; print("   crash in", p.key, e)
check("all profiles survive empty frame", ok)

# --- profiles survive an adaptive body map (missing left leg)
ok = True
for p in reg.all():
    try: p.analyse(engframe(partial, body_map=bm2))
    except Exception as e:
        ok = False; print("   crash in", p.key, e)
check("all profiles survive adaptive body map", ok)

# ---------------------------------------------------------------- recognizer
rec = ExerciseRecognizer(reg)
for i in range(25):
    out = rec.observe(engframe(w2pts))
check("recognizer ranks all four", len(out.ranking) == 4)
check("recognizer picks Warrior II for W2 pose", out.key == "warrior_2", out.ranking)

rec2 = ExerciseRecognizer(reg)
for i in range(25):
    out2 = rec2.observe(engframe(tpts))
check("recognizer picks Tree Pose for tree pose", out2.key == "tree_pose", out2.ranking)

rec3 = ExerciseRecognizer(reg)
for i in range(25):
    out3 = rec3.observe(engframe(bottom))
check("recognizer picks Squat for squat", out3.key == "squat", out3.ranking)

rec4 = ExerciseRecognizer(reg)
for i in range(25):
    out4 = rec4.observe(engframe(up))
check("recognizer picks Bicep Curl for curl", out4.key == "bicep_curl", out4.ranking)

# ambiguous / no person -> not confident
rec5 = ExerciseRecognizer(reg)
for i in range(25):
    out5 = rec5.observe(empty)
check("no person -> not confident", not out5.confident)

print()
print("FAILURES:", failures if failures else "none")
sys.exit(1 if failures else 0)
