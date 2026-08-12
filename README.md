# MoveWise

**Smarter guidance for every movement.**

An adaptive AI movement coach: it understands which of four exercises you're doing, finds the invisible form mistake that matters most, shows the correction on a ghost reference fitted to your own body, adapts *how* it coaches when you don't improve, checks comfort before pushing, adapts its analysis to different body configurations, and measures whether you actually got better.

> Don't adapt your body to the AI. Let the AI adapt to you.

---

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Or double-click `run.bat` (Windows). One instance per machine — the pose reader uses the local webcam.

The pose model (~6 MB) is bundled in `models/`; no network needed on first run.

## Verify without a camera

Each suite prints PASS/FAIL lines and ends with `FAILURES: none`:

```bash
python test_pipeline.py       # pose engine, smoothing, body map
python test_clips.py          # sample clips behave as their captions claim
python test_analysis.py      # analyzers, phases, reps, scoring, ghost
python test_coaching.py      # escalation ladder, comfort check, variations
python test_session.py       # session recording and summaries
python test_accessibility.py # adaptive mode and fair scoring
python test_motion.py        # movement control / stability coach
python test_progress.py      # 7-day persistent progress
python test_robust.py        # outlier rejection, ghost alignment
```

## The demo path (no hardware needed)

Home → **Library** → Squat → set Input to **Sample clip** → **Run calibration** → **Continue to live analysis**.

- **"Squat — knees caving in"** shows one prioritised cue while three faults are live.
- **"Squat — held fault (coaching demo)"** climbs the full coaching ladder: notice → instruct → focus, text → arrow → ghost emphasis, then the comfort check and an easier variation.
- **"Squat — fast/jerky"** triggers the Movement Control coach.

Webcam and uploaded video work the same way — pick a different Input.

---

## Architecture

One pose engine, four exercise profiles, one structured result. The engine never knows what an exercise is; a profile never touches MediaPipe; the UI reads only the common `ExerciseResult`.

```
Camera / Video / Scripted clip                    core/source.py, synthetic.py
  └─ PoseDetector (MediaPipe, both API gens)      core/detector.py
       └─ PoseSmoother (One Euro + outlier        core/smoothing.py
          rejection + dropout coasting)
            └─ Personal Body Map                  core/bodymap.py, bodygroups.py
               Standard / Adaptive mode
                 └─ PoseEngine → EngineFrame      core/engine.py
                      ├─ Exercise profiles        exercises/*.py
                      │    phases · reps/holds    core/phases.py
                      │    errors + debounce      core/errors.py
                      │    scoring + quality      core/scoring.py
                      │    motion control         core/motion.py
                      ├─ Ghost Coach              core/reference.py
                      ├─ Adaptive coaching        core/coaching.py
                      │    escalate → modality →
                      │    comfort → variation
                      └─ Session → Progress       core/session.py, storage.py
```

### The honesty rule everything rests on

Every measurement is `MEASURED`, `UNMEASURED` (couldn't see it), or `NOT_APPLICABLE` (this user's body map doesn't include it). Nothing is ever reported correct because it wasn't seen, unmeasured metrics leave the score untouched rather than scoring zero, and a landmark the camera can't track is never a posture error. That single rule carries the accessibility story end to end.

### Coaching, not detection

Faults pass two gates before the user hears anything: a hysteresis debounce (5 frames to appear, 10 to clear — one bad frame never speaks) and a priority ladder (`SAFETY > ALIGNMENT > STABILITY > RANGE > REFINEMENT`) that lets exactly one cue through. If the same fault persists, the *wording* escalates (notice → instruct → focus) and the *teaching method* escalates (text → simplified → arrow → ghost emphasis). Still stuck → comfort check → predefined easier variation. Every correction is verified against the following reps and logged as fixed or failed.

### Movement Control (stability coach)

Landmark motion over time — velocity, jerk, sway, normalized by torso length — produces a 0–100 **Movement control** score and calm cues ("You're moving too quickly — slow down."). It infers intensity from pose motion; it does not and cannot measure physical force. Exercise-aware thresholds: a squat is allowed to move; a tree pose is not. Low-confidence frames are never judged.

### Progress

Sessions persist to local JSON (`data/sessions.json`, atomic writes, survives restarts — no accounts, no cloud). The Progress screen shows a 7-day view (empty days say so; nothing is fabricated), previous sessions with their saved summaries, per-exercise progress (never merged across exercises), earliest→latest improvement, and a weekly summary.

---

## Safety

MoveWise provides fitness and movement guidance only. It does not diagnose medical conditions, injuries, or instability, and it is not a substitute for a qualified physiotherapist or doctor. Reported pain stops the set and refers out. This disclaimer is shown in-app on every screen.

## Packaging

```bash
python package.py        # builds movewise.zip with the model bundled
python export_samples.py # writes the sample clips as .mp4 files
```
