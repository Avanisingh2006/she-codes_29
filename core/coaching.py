"""Adaptive coaching: how one correction is delivered, escalated and verified.

The analyzers already answer *what* is wrong (`FormError`) and *which one
matters most* (`ErrorTracker.primary`). This layer answers the harder question:
*how should we say it, how many times, and did it actually work?*

Three ideas drive the whole file:

1. **One correction at a time.** A user cannot act on four cues at once. The
   coach locks onto a single error key and stays on it until it clears. Whatever
   else is active is the analyzer's business, not the coach's.

2. **Escalate slowly, along two independent ladders.** Repeating the same
   sentence every frame is nagging, and shouting it louder is not teaching. So
   the *wording* climbs NOTICE -> INSTRUCT -> FOCUS while the *modality* climbs
   TEXT -> SIMPLE_TEXT -> ARROW -> GHOST_EMPHASIS. A cue that is not landing as
   words gets shown as geometry instead. Both ladders step on a dwell timer, not
   on frames.

3. **Verify, then back off.** Coaching that is never checked is just talking. We
   remember the severity at the moment coaching began; if the fault disappears
   that is a SUCCESS worth celebrating, if it softens that is IMPROVING and we
   stop escalating. If it survives three attempts unchanged we stop pushing
   harder and ask the user a *fitness* question — is this comfortable? — and, if
   not, offer an easier version of the same movement.

Scope note: this is fitness guidance only. Nothing here diagnoses pain, injury
or any medical condition, and the comfort check exists purely to pick an easier
variation of the exercise — never to interpret a symptom.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

from .errors import FormError

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps core/ free of exercises/
    from exercises.base import ExerciseResult


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
class CoachStage(str, Enum):
    """How firmly the *same* error is worded, in escalating order."""
    NOTICE = "notice"       # first mention: a short flag, no lecture
    INSTRUCT = "instruct"   # it came back: say what to actually do
    FOCUS = "focus"         # still there: drop everything else, one thing only


class Modality(str, Enum):
    """How the correction is *taught*, in escalating order.

    Words first, because words are cheapest. If words are not working we stop
    adding words and start showing the movement instead.
    """
    TEXT = "text"                       # full sentence
    SIMPLE_TEXT = "simple_text"         # short imperative, fewer words
    ARROW = "arrow"                     # draw the direction on the joint
    GHOST_EMPHASIS = "ghost_emphasis"   # highlight the reference pose itself


class CorrectionOutcome(str, Enum):
    """Did the coaching land?"""
    PENDING = "pending"       # coaching in progress, no change yet
    IMPROVING = "improving"   # same fault, but measurably softer
    SUCCESS = "success"       # fault cleared after we coached it
    FAILED = "failed"         # survived the full escalation; offer an easier version


# Stage ladder and modality ladder, indexed by attempt number (1-based).
_STAGE_LADDER: Sequence[CoachStage] = (
    CoachStage.NOTICE,
    CoachStage.INSTRUCT,
    CoachStage.FOCUS,
)
_MODALITY_LADDER: Sequence[Modality] = (
    Modality.TEXT,
    Modality.SIMPLE_TEXT,
    Modality.ARROW,
    Modality.GHOST_EMPHASIS,
)


# ---------------------------------------------------------------------------
# What the UI renders
# ---------------------------------------------------------------------------
@dataclass
class Coaching:
    """One frame's worth of coaching state. The UI renders exactly this.

    `message` is always the text that should currently be on screen; `speak` is
    True only on the frame where that text *changed*. That split is what lets the
    panel hold a sentence steadily while TTS / logging fire once — the coach
    never says the identical sentence twice in a row.
    """
    message: str = ""
    stage: CoachStage = CoachStage.NOTICE
    modality: Modality = Modality.TEXT
    error_key: Optional[str] = None
    outcome: CorrectionOutcome = CorrectionOutcome.PENDING
    show_comfort_check: bool = False
    suggested_variation: Optional[Dict[str, str]] = None
    attempts: int = 0
    successes: int = 0

    # -- supporting detail -------------------------------------------------
    speak: bool = False                 # True only when `message` just changed
    comfort_question: str = ""          # populated while show_comfort_check is True
    comfort_options: Sequence[str] = ()
    landmarks: Sequence = ()            # joints the ARROW / GHOST modality should mark

    @property
    def variation_name(self) -> Optional[str]:
        """Convenience for UIs that only want the variation's display name."""
        if not self.suggested_variation:
            return None
        return self.suggested_variation.get("name")

    @property
    def visual(self) -> bool:
        """True once words have been given up on in favour of geometry."""
        return self.modality in (Modality.ARROW, Modality.GHOST_EMPHASIS)


# ---------------------------------------------------------------------------
# The coach
# ---------------------------------------------------------------------------
class AdaptiveCoach:
    """Turns a stream of ExerciseResults into one escalating, verified correction."""

    # -- tuning ------------------------------------------------------------
    ESCALATE_SECONDS = 4.0        # dwell on a rung before climbing to the next
    MIN_UPDATES_PER_STAGE = 3     # ...and see the fault this many times too
    SUCCESS_HOLD_SECONDS = 2.0    # how long the celebration stays on screen
    COMFORT_AFTER_ATTEMPTS = 3    # attempts on one error before we ask about comfort
    # A drop this large in severity*confidence counts as genuine improvement.
    IMPROVE_MARGIN = 0.15

    COMFORT_QUESTION = ("This movement seems difficult to maintain. "
                        "How does this movement feel?")
    COMFORT_OPTIONS: Sequence[str] = ("Comfortable", "Challenging", "Uncomfortable")
    _COMFORT_LEVELS = {"comfortable", "challenging", "uncomfortable"}

    SUCCESS_MESSAGE = "🎉 Correction successful!"

    # Easier versions of the same movement. Fitness wording only: these are
    # *easier ways to do the exercise*, never treatments, and the hint is fixed
    # so no caller can slip a therapeutic claim in through this table.
    VARIATION_HINT = "This variation may be easier to perform."
    VARIATIONS: Dict[str, dict] = {
        "squat":      {"name": "Chair-assisted squat",     "hint": VARIATION_HINT},
        "bicep_curl": {"name": "Reduced-range curl",       "hint": VARIATION_HINT},
        "tree_pose":  {"name": "Tree pose with wall support", "hint": VARIATION_HINT},
        "warrior_2":  {"name": "Shorter stance Warrior II",   "hint": VARIATION_HINT},
    }

    def __init__(self) -> None:
        self.reset()

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear every scrap of per-session state. Called on Start / exercise switch."""
        self._exercise: str = ""

        # current correction
        self._key: Optional[str] = None
        self._error: Optional[FormError] = None
        self._stage: CoachStage = CoachStage.NOTICE
        self._modality: Modality = Modality.TEXT
        self._attempts: int = 0
        self._total_attempts: int = 0   # lifetime, never reset by a success
        self._outcome: CorrectionOutcome = CorrectionOutcome.PENDING
        self._stage_since: float = 0.0
        self._updates_in_stage: int = 0
        self._baseline: float = 0.0      # severity*confidence when coaching began
        self._improved: bool = False
        self._reps_at_start: int = 0

        # session counters
        self._successes: int = 0

        # celebration hold
        self._success_until: float = 0.0

        # comfort check / variation
        self._comfort_pending: bool = False
        self._comfort_answer: Optional[str] = None
        self._comfort_at_attempt: int = 0
        self._variation: Optional[Dict[str, str]] = None
        self._variation_accepted: bool = False
        self._variation_rejected: bool = False

        # anti-repetition
        self._last_spoken: str = ""
        self.last: Coaching = Coaching()

    # ------------------------------------------------------------------
    # read-only view
    # ------------------------------------------------------------------
    @property
    def error_key(self) -> Optional[str]:
        """The error currently being coached, if any."""
        return self._key

    @property
    def attempts(self) -> int:
        """Escalation attempts spent on the current (or most recent) error."""
        return self._attempts

    @property
    def total_attempts(self) -> int:
        """Every correction attempt this session.

        Distinct from `attempts`, which resets when an error is fixed — the
        session summary needs the lifetime count, otherwise a session that fixed
        three faults reports zero attempts and three successes.
        """
        return self._total_attempts

    @property
    def successes(self) -> int:
        """Corrections verified as cleared this session."""
        return self._successes

    @property
    def pending_variation(self) -> Optional[Dict[str, str]]:
        """The easier variation currently on offer, awaiting accept/reject."""
        return self._variation

    @property
    def variation_accepted(self) -> bool:
        return self._variation_accepted

    @property
    def comfort_answer(self) -> Optional[str]:
        return self._comfort_answer

    # ------------------------------------------------------------------
    # main entry point
    # ------------------------------------------------------------------
    def update(self, result: "ExerciseResult", now: float) -> Coaching:
        """Feed one analysed frame; get back what the UI should show."""
        if getattr(result, "exercise", ""):
            self._exercise = result.exercise

        primary = self._primary_of(result)
        active = {e.key for e in getattr(result, "errors", []) or []}
        if primary is not None:
            active.add(primary.key)

        # 1. Celebration takes the screen for a couple of seconds. Nothing —
        #    not even a fresh fault — interrupts a success the user earned.
        if now < self._success_until:
            return self._hold_last()

        # 2. Are we mid-correction?
        if self._key is not None:
            if self._key in active:
                return self._continue(result, primary, now)
            # The fault we were coaching is gone. That is the whole point.
            return self._resolve_success(now)

        # 3. Nothing in progress — adopt the current primary, if there is one.
        if primary is not None:
            return self._begin(primary, result, now)

        # 4. Clean frame, nothing to say.
        return self._emit(Coaching(successes=self._successes), message="")

    # ------------------------------------------------------------------
    # correction lifecycle
    # ------------------------------------------------------------------
    def _begin(self, error: FormError, result: "ExerciseResult", now: float) -> Coaching:
        """Lock onto a new error and deliver the gentlest form of the cue."""
        self._key = error.key
        self._error = error
        self._stage = CoachStage.NOTICE
        self._modality = Modality.TEXT
        self._attempts = 1
        self._total_attempts += 1
        self._outcome = CorrectionOutcome.PENDING
        self._stage_since = now
        self._updates_in_stage = 1
        self._baseline = self._intensity(error)
        self._improved = False
        self._reps_at_start = getattr(result, "rep_count", 0) or 0

        # A new error means the previous comfort/variation conversation is over.
        self._comfort_pending = False
        self._comfort_answer = None
        self._comfort_at_attempt = 0
        self._variation = None
        self._variation_rejected = False
        return self._render()

    def _continue(self, result: "ExerciseResult", primary: Optional[FormError],
                  now: float) -> Coaching:
        """The same fault is still there. Track it, and escalate on the timer."""
        # Refresh our copy so wording follows the latest severity/landmarks.
        current = self._find(result, self._key) or primary or self._error
        if current is not None:
            self._error = current
        self._updates_in_stage += 1

        # -- verification: has it softened? --------------------------------
        intensity = self._intensity(self._error) if self._error else self._baseline
        if intensity <= self._baseline - self.IMPROVE_MARGIN:
            self._improved = True
            self._outcome = CorrectionOutcome.IMPROVING
            # A softer fault is progress; re-baseline so further easing still reads
            # as improvement, and give them room rather than escalating over it.
            self._baseline = intensity
            self._stage_since = now
            self._updates_in_stage = 0

        # -- while an offer is on the table, stop pushing --------------------
        # An unanswered comfort *question* does not stop the coaching: it sits
        # alongside it, and the modality ladder still has rungs left to try. A
        # pending *variation* is a decision, so we wait for it.
        if self._variation is not None:
            return self._render()

        # -- escalation on a dwell timer, never per frame -------------------
        due = (now - self._stage_since) >= self.ESCALATE_SECONDS
        enough = self._updates_in_stage >= self.MIN_UPDATES_PER_STAGE
        if due and enough and not self._improved:
            self._escalate(now)

        return self._render()

    def _escalate(self, now: float) -> None:
        """Climb one rung of both ladders, then decide whether to ask about comfort."""
        self._attempts += 1
        self._total_attempts += 1
        self._stage = _STAGE_LADDER[min(self._attempts, len(_STAGE_LADDER)) - 1]
        self._modality = _MODALITY_LADDER[min(self._attempts, len(_MODALITY_LADDER)) - 1]
        self._stage_since = now
        self._updates_in_stage = 0

        # Words and pictures have both been tried and the fault has not budged.
        # Stop escalating the instruction and ask about the movement instead.
        if self._attempts >= self.COMFORT_AFTER_ATTEMPTS and not self._improved:
            if self._comfort_answer is None:
                # Ask, but never after they have already turned an easier
                # version down — that answer stands for the rest of the session.
                self._comfort_pending = not self._variation_rejected
            elif self._attempts > self._comfort_at_attempt:
                # They answered, we kept coaching, and it still persists.
                self._offer_variation()

    def _resolve_success(self, now: float) -> Coaching:
        """The coached fault is gone: bank it and celebrate, briefly."""
        self._successes += 1
        self._outcome = CorrectionOutcome.SUCCESS
        attempts = self._attempts
        key = self._key
        self._success_until = now + self.SUCCESS_HOLD_SECONDS

        coaching = Coaching(
            message=self.SUCCESS_MESSAGE,
            stage=self._stage,
            modality=Modality.TEXT,
            error_key=key,
            outcome=CorrectionOutcome.SUCCESS,
            attempts=attempts,
            successes=self._successes,
        )
        self._clear_correction()
        return self._emit(coaching, message=self.SUCCESS_MESSAGE)

    def _clear_correction(self) -> None:
        """Drop the current correction without touching session counters."""
        self._key = None
        self._error = None
        self._attempts = 0
        self._stage = CoachStage.NOTICE
        self._modality = Modality.TEXT
        self._improved = False
        self._baseline = 0.0
        self._updates_in_stage = 0
        self._comfort_pending = False
        self._comfort_answer = None
        self._comfort_at_attempt = 0
        self._variation = None

    # ------------------------------------------------------------------
    # comfort check and variations
    # ------------------------------------------------------------------
    def answer_comfort(self, level: str) -> Coaching:
        """Record the user's answer to the comfort check.

        `level` is one of COMFORT_OPTIONS, case-insensitive. "Uncomfortable" is
        read as *this version of the exercise is too demanding right now* — the
        response is an easier variation, never advice about a symptom.
        """
        value = (level or "").strip().lower()
        if value not in self._COMFORT_LEVELS:
            raise ValueError(f"comfort level must be one of {self.COMFORT_OPTIONS!r}")

        self._comfort_answer = value
        self._comfort_pending = False
        self._comfort_at_attempt = self._attempts

        if value == "uncomfortable":
            # Do not make them work for it — offer the easier version now.
            self._offer_variation()
        return self._render()

    def _offer_variation(self) -> None:
        """Put an easier version of the current exercise on the table, once."""
        if self._variation is not None or self._variation_rejected:
            return
        entry = self.VARIATIONS.get(self._exercise)
        if not entry:
            return
        self._variation = dict(entry)
        self._outcome = CorrectionOutcome.FAILED   # this correction did not land

    def accept_variation(self) -> Optional[Dict[str, str]]:
        """User takes the easier version: start the correction cycle over."""
        variation = self._variation
        if variation is None:
            return None
        self._variation_accepted = True
        self._variation = None
        # The movement itself is changing, so every counter for the old attempt
        # is meaningless. Session successes survive; the correction does not.
        self._clear_correction()
        self._last_spoken = ""
        self.last = Coaching(successes=self._successes)
        return variation

    def reject_variation(self) -> None:
        """User sticks with the current version: stop offering, keep coaching."""
        if self._variation is None and not self._comfort_pending:
            return
        self._variation = None
        self._variation_rejected = True
        self._comfort_pending = False
        self._outcome = CorrectionOutcome.PENDING

    # ------------------------------------------------------------------
    # message construction
    # ------------------------------------------------------------------
    def _render(self) -> Coaching:
        """Build the Coaching for the current state."""
        error = self._error
        coaching = Coaching(
            stage=self._stage,
            modality=self._modality,
            error_key=self._key,
            outcome=self._outcome,
            show_comfort_check=self._comfort_pending,
            suggested_variation=dict(self._variation) if self._variation else None,
            attempts=self._attempts,
            successes=self._successes,
            landmarks=tuple(getattr(error, "landmarks", ()) or ()),
        )
        if self._comfort_pending:
            coaching.comfort_question = self.COMFORT_QUESTION
            coaching.comfort_options = tuple(self.COMFORT_OPTIONS)

        if self._variation is not None:
            message = f"{self._variation['name']} — {self._variation['hint']}"
        elif error is not None:
            message = self._compose(self._stage, error)
        else:
            message = ""
        return self._emit(coaching, message=message)

    def _compose(self, stage: CoachStage, error: FormError) -> str:
        """The exact sentence for a stage. Text comes from the error, never a table."""
        if stage is CoachStage.NOTICE:
            return f"⚠ {self._short(error)}"
        instruction = self._instruction(error)
        if stage is CoachStage.INSTRUCT:
            return f"Try {instruction}"
        return f"🎯 Focus on ONE thing: {instruction}"

    @staticmethod
    def _short(error: FormError) -> str:
        """The glanceable form — that is exactly what `cue` already is."""
        cue = (error.cue or "").strip()
        return cue or (error.message or "").strip()

    @staticmethod
    def _instruction(error: FormError) -> str:
        """The actionable half of the error, derived from what the error already says.

        Analyzer messages are written as "<what is wrong> — <what to do>", so the
        clause after the dash is the instruction. When there is no such clause,
        or it turns out to be descriptive rather than imperative (it starts with a
        number, say), fall back to the cue, which is authored to be actionable.
        """
        message = (error.message or "").strip()
        cue = (error.cue or "").strip()

        clause = ""
        for dash in ("—", "–", " - "):
            if dash in message:
                clause = message.rsplit(dash, 1)[1].strip()
                break
        if not clause:
            # No two-part message: the sentence is already the instruction.
            clause = message

        # Descriptive tails ("8° difference between your knees") are not usable
        # as an instruction; the short cue is authored to be.
        if not clause[:1].isalpha():
            clause = cue or message

        clause = clause.rstrip(" .").strip()
        if clause.isupper():
            # Cues are written in caps for the overlay; that shouts in a sentence.
            clause = clause.lower()
        elif clause[:1].isupper() and not clause.split(" ")[0].isupper():
            # Sentence case reads wrong mid-sentence after "Try ...".
            clause = clause[0].lower() + clause[1:]
        return clause or message

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _emit(self, coaching: Coaching, message: str) -> Coaching:
        """Attach the message, flag it as new only if it is not a straight repeat."""
        coaching.message = message
        coaching.speak = bool(message) and message != self._last_spoken
        if coaching.speak:
            self._last_spoken = message
        self.last = coaching
        return coaching

    def _hold_last(self) -> Coaching:
        """Re-show the previous coaching without re-announcing it."""
        held = replace(self.last, speak=False)
        self.last = held
        return held

    @staticmethod
    def _intensity(error: Optional[FormError]) -> float:
        """How bad the fault currently is: severity weighted by how sure we are."""
        if error is None:
            return 0.0
        return float(error.severity.weight) * float(error.confidence)

    @staticmethod
    def _find(result: "ExerciseResult", key: Optional[str]) -> Optional[FormError]:
        for error in getattr(result, "errors", []) or []:
            if error.key == key:
                return error
        primary = getattr(result, "primary_error", None)
        if primary is not None and primary.key == key:
            return primary
        return None

    @staticmethod
    def _primary_of(result: "ExerciseResult") -> Optional[FormError]:
        """The error the coach should adopt: the analyzer's pick, or the top-ranked."""
        primary = getattr(result, "primary_error", None)
        if primary is not None:
            return primary
        errors: List[FormError] = list(getattr(result, "errors", []) or [])
        if not errors:
            return None
        return sorted(errors, key=lambda e: e.rank)[0]
