"""Phase 3 — the adaptive coaching layer, tested without a camera.

Drives AdaptiveCoach with hand-built FormErrors and a synthetic clock, and checks
the two escalation ladders, correction verification, the comfort check, the
variation offer and reset. No MediaPipe, no engine, no frames: the coach only
ever sees an ExerciseResult, so that is all we build.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Coach messages carry emoji; a cp1252 console would otherwise abort the run.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover - older / redirected streams
    pass

from core.coaching import (AdaptiveCoach, CoachStage, Coaching, CorrectionOutcome,
                           Modality)
from core.errors import FormError, Priority, Severity
from exercises.base import ExerciseResult

failures = []


def check(name, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + name + ("  " + str(extra) if extra else ""))
    if not cond:
        failures.append(name)


# ---------------------------------------------------------------------------
# synthetic fixtures
# ---------------------------------------------------------------------------
def err(key="left_knee_valgus",
        message="Your left knee is collapsing inward — push it out over your foot.",
        cue="LEFT KNEE -> OUT",
        severity=Severity.MAJOR,
        priority=Priority.SAFETY,
        confidence=1.0):
    return FormError(key=key, message=message, cue=cue, severity=severity,
                     priority=priority, confidence=confidence)


def res(*errors, exercise="squat", rep_count=0):
    """An ExerciseResult carrying exactly these errors, primary picked by rank."""
    ordered = sorted(errors, key=lambda e: e.rank)
    return ExerciseResult(exercise=exercise, exercise_name=exercise,
                          movement="dynamic", phase="descending",
                          rep_count=rep_count, errors=list(ordered),
                          primary_error=ordered[0] if ordered else None)


def escalate_to(coach, attempts, error=None, t0=0.0, exercise="squat"):
    """Drive the coach until it has spent `attempts` attempts on `error`.

    Returns (last coaching, clock). Steps the clock past the dwell timer and
    feeds enough updates to clear the minimum-observations guard each rung.
    """
    error = error or err()
    now = t0
    coaching = coach.update(res(error, exercise=exercise), now)
    rounds = 0
    while coaching.attempts < attempts:
        rounds += 1
        if rounds > attempts + 5:            # guard: never hang a test run
            raise AssertionError(
                f"coach stalled at attempt {coaching.attempts}, wanted {attempts}")
        now += AdaptiveCoach.ESCALATE_SECONDS + 0.1
        for _ in range(AdaptiveCoach.MIN_UPDATES_PER_STAGE):
            coaching = coach.update(res(error, exercise=exercise), now)
            now += 0.05
    return coaching, now


# ---------------------------------------------------------------------------
# 1. message escalation through all three stages
# ---------------------------------------------------------------------------
print("--- stage escalation ---")
coach = AdaptiveCoach()
e = err()
c = coach.update(res(e), 0.0)
check("first sighting is NOTICE", c.stage is CoachStage.NOTICE, c.stage)
check("NOTICE uses the short cue", c.message == "⚠ LEFT KNEE -> OUT", c.message)
check("NOTICE is a fresh utterance", c.speak is True)
check("attempts starts at 1", c.attempts == 1, c.attempts)
check("error key is tracked", c.error_key == "left_knee_valgus", c.error_key)
check("outcome starts PENDING", c.outcome is CorrectionOutcome.PENDING, c.outcome)

# Same frame content, no time passed: must not escalate and must not re-speak.
c2 = coach.update(res(e), 0.05)
c3 = coach.update(res(e), 0.10)
check("no escalation on the next frame", c3.stage is CoachStage.NOTICE, c3.stage)
check("attempts unchanged within the dwell window", c3.attempts == 1, c3.attempts)
check("identical sentence is not spoken twice", (c2.speak, c3.speak) == (False, False))
check("message still rendered while held", c3.message == "⚠ LEFT KNEE -> OUT", c3.message)

# Time passes and the fault stays: climb to INSTRUCT.
c, now = escalate_to(coach, 2, e)
check("repeat escalates to INSTRUCT", c.stage is CoachStage.INSTRUCT, c.stage)
check("INSTRUCT wording", c.message == "Try push it out over your foot", c.message)
check("INSTRUCT is spoken once", c.attempts == 2, c.attempts)

c, now = escalate_to(coach, 3, e, t0=now)
check("continued fault escalates to FOCUS", c.stage is CoachStage.FOCUS, c.stage)
check("FOCUS wording",
      c.message == "🎯 Focus on ONE thing: push it out over your foot", c.message)

# FOCUS is the top of the wording ladder — it must not overflow.
c, now = escalate_to(coach, 5, e, t0=now)
check("stage saturates at FOCUS", c.stage is CoachStage.FOCUS, c.stage)
check("attempts keep counting past FOCUS", c.attempts >= 5, c.attempts)

# ---------------------------------------------------------------------------
# 2. modality ladder runs alongside the wording ladder
# ---------------------------------------------------------------------------
print("\n--- modality escalation ---")
coach = AdaptiveCoach()
seen = []
now = 0.0
for target in (1, 2, 3, 4, 5):
    c, now = escalate_to(coach, target, e, t0=now)
    seen.append(c.modality)
check("modality starts as TEXT", seen[0] is Modality.TEXT, seen[0])
check("second attempt is SIMPLE_TEXT", seen[1] is Modality.SIMPLE_TEXT, seen[1])
check("third attempt is ARROW", seen[2] is Modality.ARROW, seen[2])
check("fourth attempt is GHOST_EMPHASIS", seen[3] is Modality.GHOST_EMPHASIS, seen[3])
check("modality saturates at GHOST_EMPHASIS", seen[4] is Modality.GHOST_EMPHASIS, seen[4])
check("visual flag set once words are abandoned", c.visual is True)

# ---------------------------------------------------------------------------
# 3. never the identical sentence back to back, over a long run
# ---------------------------------------------------------------------------
print("\n--- anti-repetition ---")
coach = AdaptiveCoach()
spoken, prev, now = [], None, 0.0
for i in range(60):
    c = coach.update(res(e), now)
    if c.speak:
        spoken.append(c.message)
    now += 0.5
check("something was actually said", len(spoken) >= 3, spoken)
check("no two consecutive utterances are identical",
      all(a != b for a, b in zip(spoken, spoken[1:])), spoken)

# ---------------------------------------------------------------------------
# 4. correction verification
# ---------------------------------------------------------------------------
print("\n--- correction verification ---")
coach = AdaptiveCoach()
c, now = escalate_to(coach, 2, e)
c = coach.update(res(), now + 0.1)          # fault gone
check("clearing the fault is a SUCCESS", c.outcome is CorrectionOutcome.SUCCESS, c.outcome)
check("success is celebrated", c.message == "🎉 Correction successful!", c.message)
check("successes counter incremented", c.successes == 1, c.successes)
check("success reports the attempts it took", c.attempts == 2, c.attempts)

# The celebration holds for ~2s even if a fresh fault shows up underneath.
held = coach.update(res(err(key="back_angle", cue="STRAIGHTEN TORSO")), now + 1.0)
check("success message holds for ~2s", held.message == "🎉 Correction successful!",
      held.message)
check("held message is not re-spoken", held.speak is False)
after = coach.update(res(err(key="back_angle", cue="STRAIGHTEN TORSO")), now + 2.6)
check("coach moves on after the hold", after.error_key == "back_angle", after.error_key)
check("new correction starts at NOTICE", after.stage is CoachStage.NOTICE, after.stage)
check("session successes survive the new error", after.successes == 1, after.successes)

# Severity dropping is progress, not success.
coach = AdaptiveCoach()
major = err(severity=Severity.MAJOR)
minor = err(severity=Severity.MINOR)
c, now = escalate_to(coach, 2, major)
c = coach.update(res(minor), now + 0.1)
check("softening severity reads as IMPROVING",
      c.outcome is CorrectionOutcome.IMPROVING, c.outcome)
check("improving does not bump successes", c.successes == 0, c.successes)
# ...and improvement buys time rather than a harder cue.
for i in range(6):
    c = coach.update(res(minor), now + 0.2 + i * 0.5)
check("improvement pauses escalation", c.attempts == 2, c.attempts)
check("improvement suppresses the comfort check", c.show_comfort_check is False)

# ---------------------------------------------------------------------------
# 5. comfort check
# ---------------------------------------------------------------------------
print("\n--- comfort check ---")
coach = AdaptiveCoach()
c, now = escalate_to(coach, 3, e)
check("comfort check after 3 unimproved attempts", c.show_comfort_check is True)
check("comfort question wording",
      c.comfort_question == "This movement seems difficult to maintain. "
                            "How does this movement feel?", c.comfort_question)
check("comfort options are the three levels",
      tuple(c.comfort_options) == ("Comfortable", "Challenging", "Uncomfortable"),
      c.comfort_options)
check("no variation offered before the user answers", c.suggested_variation is None)

# The question sits alongside the coaching rather than replacing it.
c, now = escalate_to(coach, 4, e, t0=now)
check("question stays open until it is answered", c.show_comfort_check is True)
check("coaching keeps teaching while the question is open",
      c.modality is Modality.GHOST_EMPHASIS, c.modality)
check("no variation offered before an answer", c.suggested_variation is None,
      c.suggested_variation)

# ---------------------------------------------------------------------------
# 6. variation suggestion, accept and reject
# ---------------------------------------------------------------------------
print("\n--- variation: uncomfortable answer ---")
c = coach.answer_comfort("Uncomfortable")
check("answering clears the question", c.show_comfort_check is False)
check("uncomfortable offers a variation", c.suggested_variation is not None,
      c.suggested_variation)
check("variation is the squat mapping",
      c.suggested_variation == {"name": "Chair-assisted squat",
                                "hint": "This variation may be easier to perform."},
      c.suggested_variation)
check("variation_name convenience", c.variation_name == "Chair-assisted squat",
      c.variation_name)
check("failed correction is marked FAILED", c.outcome is CorrectionOutcome.FAILED,
      c.outcome)
check("variation message uses the approved wording",
      c.message == "Chair-assisted squat — This variation may be easier to perform.",
      c.message)
check("coach exposes the pending variation",
      coach.pending_variation == c.suggested_variation, coach.pending_variation)

accepted = coach.accept_variation()
check("accept returns the variation", accepted["name"] == "Chair-assisted squat", accepted)
check("accept records the switch", coach.variation_accepted is True)
check("accept clears the pending offer", coach.pending_variation is None)
c = coach.update(res(e), 100.0)
check("coaching restarts fresh after accepting", (c.attempts, c.stage) == (1, CoachStage.NOTICE),
      (c.attempts, c.stage))
check("accepting does not lose session successes", c.successes == 0, c.successes)

print("\n--- variation: persists after a non-uncomfortable answer ---")
coach = AdaptiveCoach()
c, now = escalate_to(coach, 3, e)
check("comfort check raised again on a fresh coach", c.show_comfort_check is True)
c = coach.answer_comfort("challenging")
check("challenging does not offer a variation yet", c.suggested_variation is None,
      c.suggested_variation)
check("challenging resumes coaching", c.show_comfort_check is False)
c, now = escalate_to(coach, 4, e, t0=now)
check("persisting after the comfort check offers a variation",
      c.suggested_variation is not None, c.suggested_variation)

coach.reject_variation()
c = coach.update(res(e), now + 0.1)
check("reject clears the offer", c.suggested_variation is None, c.suggested_variation)
check("reject is remembered by the coach", coach.pending_variation is None)
check("rejecting keeps the coach on the error", c.error_key == "left_knee_valgus",
      c.error_key)
c, now = escalate_to(coach, 7, e, t0=now + 0.2)
check("no re-nagging with the same variation", c.suggested_variation is None,
      c.suggested_variation)
check("no re-asking the comfort question", c.show_comfort_check is False)

# ---------------------------------------------------------------------------
# 7. the VARIATIONS table itself
# ---------------------------------------------------------------------------
print("\n--- variations table ---")
V = AdaptiveCoach.VARIATIONS
check("exactly the four exercises are covered",
      sorted(V) == ["bicep_curl", "squat", "tree_pose", "warrior_2"], sorted(V))
check("squat variation name", V["squat"]["name"] == "Chair-assisted squat")
check("curl variation name", V["bicep_curl"]["name"] == "Reduced-range curl")
check("tree pose variation is supported, not therapeutic",
      V["tree_pose"]["name"] == "Tree pose with wall support")
check("warrior II variation is a gentler stance",
      V["warrior_2"]["name"] == "Shorter stance Warrior II")
check("every hint uses the exact approved wording",
      all(v["hint"] == "This variation may be easier to perform." for v in V.values()))
check("every entry is name+hint only",
      all(set(v) == {"name", "hint"} for v in V.values()))

BANNED = ("pain", "injur", "rehab", "therap", "medical", "doctor", "physio",
          "diagnos", "treat", "symptom", "condition", "heal", "strain")
table_text = " ".join(v["name"] + " " + v["hint"] for v in V.values()).lower()
check("variation table makes no medical claims",
      not any(w in table_text for w in BANNED), table_text)

# Nothing the coach can ever say should stray into medical territory.
spoken_ever = [AdaptiveCoach.SUCCESS_MESSAGE, AdaptiveCoach.COMFORT_QUESTION,
               *AdaptiveCoach.COMFORT_OPTIONS, table_text]
check("no coach-authored string makes a medical claim",
      not any(w in " ".join(spoken_ever).lower() for w in BANNED), spoken_ever)

# A yoga error routes to the yoga variation, not the squat one.
coach = AdaptiveCoach()
wobble = err(key="unstable", message="You're wobbling — fix your gaze on one point.",
             cue="STEADY", severity=Severity.MINOR, priority=Priority.STABILITY)
c, now = escalate_to(coach, 3, wobble, exercise="tree_pose")
c = coach.answer_comfort("uncomfortable")
check("tree pose gets the tree pose variation",
      c.variation_name == "Tree pose with wall support", c.variation_name)

# An exercise with no variation on file must simply not offer one.
coach = AdaptiveCoach()
c, now = escalate_to(coach, 3, e, exercise="plank")
c = coach.answer_comfort("uncomfortable")
check("unknown exercise offers nothing rather than guessing",
      c.suggested_variation is None, c.suggested_variation)

try:
    AdaptiveCoach().answer_comfort("sore")
    check("invalid comfort level is rejected", False)
except ValueError:
    check("invalid comfort level is rejected", True)

# ---------------------------------------------------------------------------
# 8. instruction derivation from the error's own text
# ---------------------------------------------------------------------------
print("\n--- instruction derivation ---")
coach = AdaptiveCoach()
descriptive = err(key="asymmetry",
                  message="You're favouring one side — 8° difference between your knees.",
                  cue="EVEN OUT BOTH LEGS", severity=Severity.MINOR,
                  priority=Priority.ALIGNMENT)
c, now = escalate_to(coach, 2, descriptive)
check("descriptive tail falls back to the cue",
      c.message == "Try even out both legs", c.message)

coach = AdaptiveCoach()
plain = err(key="foot_too_low", message="Slide your foot higher up the standing leg.",
            cue="FOOT HIGHER", severity=Severity.MINOR, priority=Priority.ALIGNMENT)
c, now = escalate_to(coach, 2, plain)
check("dash-free message is used whole, in sentence flow",
      c.message == "Try slide your foot higher up the standing leg", c.message)

# ---------------------------------------------------------------------------
# 9. quiet frames and reset
# ---------------------------------------------------------------------------
print("\n--- idle and reset ---")
coach = AdaptiveCoach()
c = coach.update(res(), 0.0)
check("clean form says nothing", c.message == "", repr(c.message))
check("clean form has no error key", c.error_key is None)
check("clean form is silent", c.speak is False)

coach = AdaptiveCoach()
c, now = escalate_to(coach, 3, e)
coach.update(res(), now + 0.1)          # bank a success
coach.reset()
check("reset clears the tracked error", coach.error_key is None)
check("reset clears attempts", coach.attempts == 0, coach.attempts)
check("reset clears successes", coach.successes == 0, coach.successes)
check("reset clears the pending variation", coach.pending_variation is None)
check("reset clears the comfort answer", coach.comfort_answer is None)
check("reset clears the accepted flag", coach.variation_accepted is False)
check("reset clears the last coaching", coach.last.message == "", repr(coach.last.message))
c = coach.update(res(e), 0.0)
check("coach speaks again from scratch after reset",
      (c.stage, c.attempts, c.speak) == (CoachStage.NOTICE, 1, True),
      (c.stage, c.attempts, c.speak))


print()
print("FAILURES:", failures if failures else "none")
sys.exit(1 if failures else 0)
