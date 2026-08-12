"""Adaptive Mode — MoveWise must not assume a standard symmetrical body.

A landmark this camera cannot see on this person is a fact about the setup, not
a posture error. This suite drives the analysis stack with synthetic poses for
two users doing the same movement — one tracked in full, one missing a limb —
and checks the second is never told off for the limb they do not have, and never
scores lower for it.

No camera, no MediaPipe: poses are built by hand.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:                                    # the tick below is not in every codepage
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                       # pragma: no cover - older interpreters
    pass

from core.bodygroups import (GROUPS, GROUP_NAMES, HEAD, LEFT_ARM, LEFT_LEG,
                             MARK_MISSING, MARK_OK, RIGHT_ARM, RIGHT_LEG,
                             group_status, summary_lines, symmetry_available,
                             trackable_subset)
from core.bodymap import BodyMap, BodyMapCalibrator, BodyMode
from core.engine import EngineFrame, PoseEngine
from core.landmarks import LM, Landmark, MetricStatus, PoseFrame
from core.scoring import Metric, aggregate_score
from exercises.registry import ExerciseRegistry

failures = []


def check(name, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + name + ("  " + str(extra) if extra else ""))
    if not cond:
        failures.append(name)


def mk(points, vis=0.95, t=0.0):
    """points: {LM: (wx,wy,wz)} in metres, y down."""
    lms = {}
    for lm, (wx, wy, wz) in points.items():
        lms[int(lm)] = Landmark(idx=int(lm), x=0.5 + wx, y=0.5 + wy, z=wz,
                                visibility=vis, wx=wx, wy=wy, wz=wz)
    return PoseFrame(landmarks=lms, frame_index=1, timestamp=t, detected=True)


def calibrate(points, frames=30):
    cal = BodyMapCalibrator(duration=0.2)
    for i in range(frames):
        cal.observe(mk(points, t=i * 0.02))
    return cal.build()


# One engine for the whole suite: it is only used for its angle maths, and each
# instance otherwise spins up a detector we never feed an image to.
_ENGINE = PoseEngine(auto_calibrate=False)


def engframe(points, body_map, t=0.0, vis=0.95):
    pose = mk(points, vis=vis, t=t)
    _ENGINE.force_body_map(body_map)
    return EngineFrame(pose=pose, angles=_ENGINE._compute_angles(pose), body_map=body_map)


def without(points, *lms):
    return {k: v for k, v in points.items() if k not in lms}


def run(profile, poses, body_map, dt=0.5):
    """Play a list of poses through one profile; collect results and every error seen."""
    results, errors = [], []
    profile.reset()
    for i, pts in enumerate(poses):
        r = profile.analyse(engframe(pts, body_map, t=i * dt))
        results.append(r)
        errors.extend(r.errors)
    return results, errors


def mentions_left(error):
    return "left" in error.key.lower() or "left" in error.message.lower()


# --- one body, tracked in full --------------------------------------------
STANDING = {
    LM.NOSE:           (0.00, -0.75, 0.00),
    LM.LEFT_SHOULDER:  (-0.20, -0.50, 0.00), LM.RIGHT_SHOULDER: (0.20, -0.50, 0.00),
    LM.LEFT_ELBOW:     (-0.22, -0.20, 0.00), LM.RIGHT_ELBOW:    (0.22, -0.20, 0.00),
    LM.LEFT_WRIST:     (-0.24, 0.10, 0.00),  LM.RIGHT_WRIST:    (0.24, 0.10, 0.00),
    LM.LEFT_HIP:       (-0.15, 0.00, 0.00),  LM.RIGHT_HIP:      (0.15, 0.00, 0.00),
    LM.LEFT_KNEE:      (-0.15, 0.45, 0.00),  LM.RIGHT_KNEE:     (0.15, 0.45, 0.00),
    LM.LEFT_ANKLE:     (-0.15, 0.90, 0.00),  LM.RIGHT_ANKLE:    (0.15, 0.90, 0.00),
}

# A clean, deep, symmetric squat bottom — knees ~93 deg, torso barely leaning.
BOTTOM = dict(STANDING)
BOTTOM.update({
    LM.NOSE:           (0.00, -0.50, 0.05),
    LM.LEFT_SHOULDER:  (-0.20, -0.25, 0.05), LM.RIGHT_SHOULDER: (0.20, -0.25, 0.05),
    LM.LEFT_ELBOW:     (-0.22, 0.05, 0.05),  LM.RIGHT_ELBOW:    (0.22, 0.05, 0.05),
    LM.LEFT_WRIST:     (-0.24, 0.35, 0.05),  LM.RIGHT_WRIST:    (0.24, 0.35, 0.05),
    LM.LEFT_HIP:       (-0.15, 0.25, 0.00),  LM.RIGHT_HIP:      (0.15, 0.25, 0.00),
    LM.LEFT_KNEE:      (-0.15, 0.50, 0.30),  LM.RIGHT_KNEE:     (0.15, 0.50, 0.30),
})

# Same body, no left leg below the hip, and same body, no left forearm.
NO_LEFT_LEG = (LM.LEFT_KNEE, LM.LEFT_ANKLE)
NO_LEFT_ARM = (LM.LEFT_ELBOW, LM.LEFT_WRIST)


print("=== calibration: a standard body ===")
standard = calibrate(STANDING)
check("full body -> STANDARD mode", standard.mode is BodyMode.STANDARD, standard.mode)
check("full body -> symmetry applicable", standard.symmetry_applicable() is True)
check("full body -> no missing groups", standard.missing_groups() == [],
      standard.missing_groups())
check("full body -> all groups available",
      standard.available_groups() == list(GROUP_NAMES), standard.available_groups())
check("full body -> core landmarks all trackable", standard.missing_core() == [])


print()
print("=== calibration: a body with no left leg ===")
adaptive = calibrate(without(STANDING, *NO_LEFT_LEG))
check("missing left leg -> ADAPTIVE mode", adaptive.mode is BodyMode.ADAPTIVE, adaptive.mode)
check("missing left leg -> symmetry NOT applicable",
      adaptive.symmetry_applicable() is False)
check("'Left leg' is reported missing", LEFT_LEG in adaptive.missing_groups(),
      adaptive.missing_groups())
check("only the left leg is missing", adaptive.missing_groups() == [LEFT_LEG],
      adaptive.missing_groups())
check("right leg still available", RIGHT_LEG in adaptive.available_groups())
check("both arms still available",
      LEFT_ARM in adaptive.available_groups() and RIGHT_ARM in adaptive.available_groups())
check("left-leg metric is NOT_APPLICABLE, not a failure",
      adaptive.status_for(LM.LEFT_HIP, LM.LEFT_KNEE, LM.LEFT_ANKLE)
      is MetricStatus.NOT_APPLICABLE)
check("right-leg metric still MEASURED",
      adaptive.status_for(LM.RIGHT_HIP, LM.RIGHT_KNEE, LM.RIGHT_ANKLE)
      is MetricStatus.MEASURED)
check("applicable() agrees with status_for",
      adaptive.applicable(LM.RIGHT_KNEE) and not adaptive.applicable(LM.LEFT_KNEE))
check("adaptive summary names the mode, not a fault",
      "Adaptive" in adaptive.summary() and "not tracking" in adaptive.summary())

one_armed = calibrate(without(STANDING, *NO_LEFT_ARM))
check("missing left arm also disables symmetry",
      one_armed.symmetry_applicable() is False)
check("missing left arm reports 'Left arm'",
      one_armed.missing_groups() == [LEFT_ARM], one_armed.missing_groups())


print()
print("=== body groups: shape of the calibration read-out ===")
status = group_status(adaptive)
check("group_status returns one entry per group", len(status) == len(GROUPS), len(status))
check("group_status entries are (name, bool)",
      all(isinstance(n, str) and isinstance(ok, bool) for n, ok in status), status[:2])
check("group_status names match GROUPS order",
      [n for n, _ in status] == list(GROUP_NAMES))
check("group_status marks Head available", dict(status)[HEAD] is True)
check("group_status marks Left leg unavailable", dict(status)[LEFT_LEG] is False)

lines = summary_lines(adaptive)
check("summary_lines returns one line per group", len(lines) == len(GROUPS), len(lines))
check("summary_lines ticks what is tracked", f"{HEAD} {MARK_OK}" in lines, lines[0])
check("summary_lines dashes what is not", f"{LEFT_LEG} {MARK_MISSING}" in lines,
      [l for l in lines if l.startswith(LEFT_LEG)])
check("summary_lines are plain strings", all(isinstance(l, str) for l in lines))
check("no body map yet -> nothing ruled out",
      symmetry_available(None) and all(ok for _n, ok in group_status(None)))
check("required landmarks filter down to this body",
      trackable_subset(adaptive, (LM.LEFT_KNEE, LM.RIGHT_KNEE)) == (LM.RIGHT_KNEE,),
      trackable_subset(adaptive, (LM.LEFT_KNEE, LM.RIGHT_KNEE)))


print()
print("=== squat: the adaptive user is coached, not blamed ===")
reg = ExerciseRegistry()
CYCLE = [STANDING, BOTTOM, BOTTOM, STANDING] * 3 + [STANDING] * 6
adaptive_cycle = [without(p, *NO_LEFT_LEG) for p in CYCLE]

std_results, std_errors = run(reg.get("squat"), CYCLE, standard)
ada_results, ada_errors = run(reg.get("squat"), adaptive_cycle, adaptive)

check("adaptive user can still be analysed at all", ada_results[-1].ready)
check("adaptive user's reps are counted",
      ada_results[-1].rep_count == std_results[-1].rep_count >= 1,
      (ada_results[-1].rep_count, std_results[-1].rep_count))
check("no error mentions the untracked left side",
      not any(mentions_left(e) for e in ada_errors),
      sorted({e.key for e in ada_errors if mentions_left(e)}))
check("no asymmetry error for a body with one leg",
      not any(e.key == "asymmetry" for e in ada_errors),
      sorted({e.key for e in ada_errors}))
check("clean form produces no errors for either user",
      not std_errors and not ada_errors,
      (sorted({e.key for e in std_errors}), sorted({e.key for e in ada_errors})))

sym = ada_results[-1].metric("symmetry")
check("symmetry metric is absent or NOT_APPLICABLE for the adaptive user",
      sym is None or sym.status is MetricStatus.NOT_APPLICABLE,
      None if sym is None else (sym.status, sym.value))
check("symmetry metric is still there for the standard user",
      std_results[-1].metric("symmetry") is not None)
check("no left-side metric is scored against the adaptive user",
      all(m.score() is None for m in ada_results[-1].metrics
          if m.status is MetricStatus.NOT_APPLICABLE),
      [m.key for m in ada_results[-1].metrics
       if m.status is MetricStatus.NOT_APPLICABLE])
check("the leg the user does have is still measured",
      ada_results[-1].metric("knee_track") is not None
      and ada_results[-1].metric("knee_track").status is MetricStatus.MEASURED,
      ada_results[-1].metric("knee_track").status)


print()
print("=== scoring is not penalised for a body part that isn't there ===")
# Equivalent good form, held steady: every metric either reads perfect or drops
# out, so a fair aggregate has to land in the same place for both bodies.
HOLD = [STANDING] * 20
std_hold, _ = run(reg.get("squat"), HOLD, standard, dt=0.05)
ada_hold, _ = run(reg.get("squat"), [without(p, *NO_LEFT_LEG) for p in HOLD],
                  adaptive, dt=0.05)
std_score, ada_score = std_hold[-1].score, ada_hold[-1].score

check("standard user scores", std_score is not None, std_score)
check("adaptive user scores", ada_score is not None, ada_score)
check("adaptive user is not scored lower than the standard user",
      ada_score is not None and std_score is not None and ada_score >= std_score - 0.001,
      (round(ada_score or -1, 2), round(std_score or -1, 2)))
check("good form scores full marks for both",
      std_score == 100.0 and ada_score == 100.0, (std_score, ada_score))
check("the adaptive user is scored on fewer metrics, not worse ones",
      len(ada_hold[-1].scorable_metrics) < len(std_hold[-1].scorable_metrics)
      and all(m.score() == 100.0 for m in ada_hold[-1].scorable_metrics),
      (len(ada_hold[-1].scorable_metrics), len(std_hold[-1].scorable_metrics)))

# Same again across a full rep cycle: the best either body can do is the same.
std_best = max(r.score for r in std_results if r.score is not None)
ada_best = max(r.score for r in ada_results if r.score is not None)
check("adaptive user reaches the same peak score over a full squat",
      ada_best >= std_best - 0.001, (round(ada_best, 2), round(std_best, 2)))

shared = {m.key: m for m in std_results[-1].metrics}
same = [k for k, m in ((m.key, m) for m in ada_results[-1].metrics)
        if m.measured and k in shared and shared[k].measured
        and abs((m.value or 0) - (shared[k].value or 0)) < 1e-6]
check("shared metrics read identically for both users", len(same) >= 3, same)

# scoring.py's own contract, verified rather than changed.
na = Metric("na", "N/A", None, MetricStatus.NOT_APPLICABLE, lo=80, hi=100)
good = Metric("ok", "OK", 90.0, MetricStatus.MEASURED, lo=80, hi=100)
check("aggregate_score drops NOT_APPLICABLE metrics",
      aggregate_score([good, na]) == aggregate_score([good]) == 100.0,
      (aggregate_score([good, na]), aggregate_score([good])))
check("aggregate_score of nothing applicable is None", aggregate_score([na]) is None)


print()
print("=== bicep curl: one arm is a body, not a fault ===")
CURL_UP = dict(STANDING)
CURL_UP[LM.LEFT_WRIST] = (-0.26, -0.45, 0.0)
CURL_UP[LM.RIGHT_WRIST] = (0.26, -0.45, 0.0)
curl_cycle = [STANDING, CURL_UP, CURL_UP, STANDING] * 3 + [STANDING] * 6
one_arm_cycle = [without(p, *NO_LEFT_ARM) for p in curl_cycle]

curl_results, curl_errors = run(reg.get("bicep_curl"), one_arm_cycle, one_armed)
check("one-armed user can still be analysed", curl_results[-1].ready)
check("no curl error mentions the untracked arm",
      not any(mentions_left(e) for e in curl_errors),
      sorted({e.key for e in curl_errors if mentions_left(e)}))
left_elbow = curl_results[-1].metric("left_elbow")
left_reps = curl_results[-1].metric("left_reps")
check("untracked arm's angle metric is NOT_APPLICABLE",
      left_elbow is not None and left_elbow.status is MetricStatus.NOT_APPLICABLE,
      None if left_elbow is None else left_elbow.status)
check("untracked arm's rep count is NOT_APPLICABLE, not zero",
      left_reps is not None and left_reps.status is MetricStatus.NOT_APPLICABLE,
      None if left_reps is None else (left_reps.status, left_reps.value))
check("untracked arm reads 'not applicable' on screen",
      left_elbow.display == "not applicable", left_elbow.display)
check("tracked arm still counts reps", curl_results[-1].rep_count >= 1,
      curl_results[-1].rep_count)
check("one-armed user still gets a score", curl_results[-1].score is not None,
      curl_results[-1].score)


print()
print("=== yoga profiles degrade gracefully ===")
TREE = dict(STANDING)
TREE[LM.RIGHT_ANKLE] = (0.05, 0.45, 0.0)          # right foot raised onto the left leg
tree_poses = [TREE] * 20

for key, poses, bmap in (("tree_pose", tree_poses, adaptive),
                         ("warrior_2", [BOTTOM] * 20, adaptive)):
    profile = reg.get(key)
    crashed = None
    try:
        results, errs = run(profile, [without(p, *NO_LEFT_LEG) for p in poses],
                            bmap, dt=0.05)
    except Exception as exc:            # pragma: no cover - failure path
        crashed, results, errs = exc, [], []
    check(f"{key} survives a body with no left leg", crashed is None, crashed)
    check(f"{key} raises no error about the missing side",
          not any(mentions_left(e) for e in errs),
          sorted({e.key for e in errs if mentions_left(e)}))
    if results:
        na_keys = {m.key for m in results[-1].metrics
                   if m.status is MetricStatus.NOT_APPLICABLE}
        check(f"{key} scores nothing it cannot see",
              all(m.score() is None for m in results[-1].metrics
                  if m.status is not MetricStatus.MEASURED), sorted(na_keys))

# Bilateral level metrics stand down when a side is untracked.
half_map = calibrate(without(STANDING, LM.LEFT_HIP, LM.LEFT_KNEE, LM.LEFT_ANKLE))
tree_half, _ = run(reg.get("tree_pose"),
                   [without(p, LM.LEFT_HIP, LM.LEFT_KNEE, LM.LEFT_ANKLE)
                    for p in tree_poses], half_map, dt=0.05)
hip_metric = tree_half[-1].metric("hip_tilt") if tree_half and tree_half[-1].ready else None
check("hip level is NOT_APPLICABLE without both hips",
      hip_metric is None or hip_metric.status is MetricStatus.NOT_APPLICABLE,
      None if hip_metric is None else hip_metric.status)

# And an empty body map must not crash any of this.
blank = BodyMap()
check("an empty body map reports every group missing",
      blank.available_groups() == [] and len(blank.missing_groups()) == len(GROUPS))
check("an empty body map disables symmetry", not blank.symmetry_applicable())


print()
print("FAILURES:", failures if failures else "none")
sys.exit(1 if failures else 0)
