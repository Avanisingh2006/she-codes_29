"""Robustness tests — landmark outlier rejection, coasting, and ghost alignment.

Covers the two additions layered onto the engine:
  * PoseSmoother rejects single-frame landmark "teleports" without ever
    fighting sustained motion or breaking the coasting contract.
  * reference.alignment_status gives the UI a coarse aligned/close/off label.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.landmarks import LM, Landmark, PoseFrame
from core.reference import GHOST_BONES, alignment_status, fit_reference
from core.smoothing import PoseSmoother

failures = []


def check(name, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + name + ("  " + str(extra) if extra else ""))
    if not cond:
        failures.append(name)


def mk(points, vis=0.95, t=0.0):
    """points: {LM: (wx,wy,wz)} in metres, y down. Same helper as test_pipeline."""
    lms = {}
    for lm, (wx, wy, wz) in points.items():
        lms[int(lm)] = Landmark(idx=int(lm), x=0.5 + wx, y=0.5 + wy, z=wz,
                                visibility=vis, wx=wx, wy=wy, wz=wz)
    return PoseFrame(landmarks=lms, frame_index=1, timestamp=t, detected=True)


# Shoulder mid (0,-0.5) to hip mid (0,0) -> torso scale is exactly 0.5 m.
TORSO = {
    LM.LEFT_SHOULDER: (-0.20, -0.50, 0.0), LM.RIGHT_SHOULDER: (0.20, -0.50, 0.0),
    LM.LEFT_HIP: (-0.15, 0.00, 0.0), LM.RIGHT_HIP: (0.15, 0.00, 0.0),
}
SCALE = 0.5
HOME = (-0.24, 0.10, 0.0)          # wrist resting position


def frame_with_wrist(pos, t):
    pts = dict(TORSO)
    pts[LM.LEFT_WRIST] = pos
    return mk(pts, t=t)


def wrist(frame):
    return frame.landmarks.get(int(LM.LEFT_WRIST))


# =====================================================================
print("=== 1. single-frame teleport is rejected ===")
sm = PoseSmoother()
for i in range(5):
    sm.apply(frame_with_wrist(HOME, t=i / 30))

# The wrist "teleports" 2 torso-lengths away for exactly one frame.
tele = (HOME[0] + 2.0 * SCALE, HOME[1], 0.0)
out = sm.apply(frame_with_wrist(tele, t=5 / 30))
w = wrist(out)
check("teleport frame: output stays near original position",
      abs(w.wx - HOME[0]) < 0.05, round(w.wx, 3))
check("teleport frame: confidence decays like coasting",
      w.visibility < 0.95, round(w.visibility, 3))

# Next normal frame resumes tracking at full confidence.
out = sm.apply(frame_with_wrist(HOME, t=6 / 30))
w = wrist(out)
check("next normal frame resumes tracking", abs(w.wx - HOME[0]) < 0.05, round(w.wx, 3))
check("confidence restored after resume", w.visibility == 0.95, w.visibility)


# =====================================================================
print()
print("=== 2. sustained fast movement is accepted (no lag-lock) ===")
sm = PoseSmoother()
for i in range(5):
    sm.apply(frame_with_wrist(HOME, t=i / 30))

far = (HOME[0] + 1.5 * SCALE, HOME[1], 0.0)   # jumps 1.5 torso-lengths and STAYS
outs, viss = [], []
for j in range(15):
    out = sm.apply(frame_with_wrist(far, t=(5 + j) / 30))
    outs.append(wrist(out).wx)
    viss.append(wrist(out).visibility)

# Frame 1 at the new spot may be rejected; by frame 2-3 it must be accepted
# and the output visibly on its way to the new position.
check("accepted within 2-3 frames (output moving by frame 3)",
      abs(outs[2] - HOME[0]) > 0.10, [round(x, 3) for x in outs[:4]])
check("output converges to the new position",
      abs(outs[-1] - far[0]) < 0.10, round(outs[-1], 3))
check("full confidence once motion is accepted", viss[-1] == 0.95, viss[-1])


# =====================================================================
print()
print("=== 3. normal jitter is smoothed, never rejected ===")
sm = PoseSmoother()
rng = random.Random(42)
raw, smooth, viss = [], [], []
for i in range(60):
    jx = HOME[0] + rng.uniform(-0.01, 0.01)
    jy = HOME[1] + rng.uniform(-0.01, 0.01)
    out = sm.apply(frame_with_wrist((jx, jy, 0.0), t=i / 30))
    raw.append(jx)
    smooth.append(wrist(out).wx)
    viss.append(wrist(out).visibility)


def var(xs):
    m = sum(xs) / len(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)


check("smoothed variance < raw variance", var(smooth) < var(raw),
      (round(var(smooth), 7), round(var(raw), 7)))
check("no rejection triggered on jitter (confidence never decayed)",
      all(v == 0.95 for v in viss))


# =====================================================================
print()
print("=== 4. coasting behaviour is unchanged ===")
sm = PoseSmoother()
for i in range(5):
    sm.apply(mk({LM.LEFT_KNEE: (0, 0, 0)}, t=i / 30))
coasted = sm.apply(PoseFrame(landmarks={}, frame_index=99, timestamp=6 / 30, detected=True))
check("coasts through dropout", int(LM.LEFT_KNEE) in coasted.landmarks)
for i in range(20):
    coasted = sm.apply(PoseFrame(landmarks={}, frame_index=i, timestamp=(7 + i) / 30,
                                 detected=True))
check("gives up after MAX_STALE_FRAMES", int(LM.LEFT_KNEE) not in coasted.landmarks)

# After a genuine dropout the last position is old, so a large jump is
# plausible and must be accepted immediately — rejection only vouches for
# landmarks it was tracking the frame before.
sm = PoseSmoother()
for i in range(5):
    sm.apply(frame_with_wrist(HOME, t=i / 30))
for i in range(3):
    sm.apply(mk(TORSO, t=(5 + i) / 30))            # wrist genuinely missing
out = sm.apply(frame_with_wrist(far, t=8 / 30))
w = wrist(out)
check("far position after real dropout is accepted, not rejected",
      w.visibility == 0.95 and abs(w.wx - HOME[0]) > 0.10,
      (round(w.wx, 3), w.visibility))


# =====================================================================
print()
print("=== 5. ghost alignment status ===")
from core.synthetic import STANDING

shape = (720, 720, 3)
pose = mk(STANDING)
ghost = fit_reference("squat", pose, shape, progress=0.0)
check("ghost produced for standing user", ghost is not None)
check("matching pose -> aligned",
      alignment_status(pose, ghost, shape) == "aligned",
      alignment_status(pose, ghost, shape))

DEV_JOINTS = (LM.LEFT_ELBOW, LM.RIGHT_ELBOW, LM.LEFT_WRIST, LM.RIGHT_WRIST,
              LM.LEFT_KNEE, LM.RIGHT_KNEE, LM.LEFT_ANKLE, LM.RIGHT_ANKLE)


def shifted(dx):
    """STANDING with every scored limb joint pushed sideways by dx metres."""
    return mk({lm: ((x + dx, y, z) if lm in DEV_JOINTS else (x, y, z))
               for lm, (x, y, z) in STANDING.items()})


check("slightly-off pose -> close",
      alignment_status(shifted(0.10), ghost, shape) == "close",
      alignment_status(shifted(0.10), ghost, shape))
check("far-off pose -> off",
      alignment_status(shifted(0.40), ghost, shape) == "off",
      alignment_status(shifted(0.40), ghost, shape))

no_torso = mk({lm: v for lm, v in STANDING.items()
               if lm not in (LM.LEFT_HIP, LM.RIGHT_HIP)})
check("torso missing -> None", alignment_status(no_torso, ghost, shape) is None)


# =====================================================================
print()
print("=== 6. ghost skeleton has no facial landmarks ===")
# MediaPipe Pose indices 0..10 are all face: nose, eyes, ears, mouth.
bone_lms = {lm for bone in GHOST_BONES for lm in bone}
check("GHOST_BONES contains no facial landmarks",
      all(int(lm) > 10 for lm in bone_lms),
      sorted(int(lm) for lm in bone_lms))


# =====================================================================
print()
print("=== 7. fit_reference regression: all four exercises ===")
for ex in ("squat", "bicep_curl", "tree_pose", "warrior_2"):
    gh = fit_reference(ex, pose, shape, progress=0.5)
    check(f"fit_reference returns a ghost for {ex}",
          gh is not None and len(gh) > 10,
          None if gh is None else len(gh))


print()
print("FAILURES:", failures if failures else "none")
sys.exit(1 if failures else 0)
