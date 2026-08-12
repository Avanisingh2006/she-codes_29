"""Stability & Controlled Movement Coach — motion quality inferred from landmarks.

Everything in this module is inferred from how pose landmarks move over time,
normalized by torso length so distance to the camera cancels out. It never
measures — and never claims to measure — physical force, load, or effort, and it
makes no medical judgments. It only describes what it can actually see: how
fast landmarks travel, how smoothly their speed changes, and how much the hip
centre drifts side to side.

Confidence discipline matches the rest of the stack: a landmark below the
visibility threshold contributes nothing, and a window that is not yet full
produces no score and no candidate errors. Low confidence means no strong
stability judgment.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from . import config
from .engine import EngineFrame
from .errors import FormError, Priority, Severity
from .geometry import torso_scale
from .landmarks import LM

# Joints whose paths we watch. Face landmarks are excluded on purpose — they
# jitter and say nothing about movement control.
TRACKED: Tuple[LM, ...] = (
    LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER,
    LM.LEFT_ELBOW, LM.RIGHT_ELBOW,
    LM.LEFT_WRIST, LM.RIGHT_WRIST,
    LM.LEFT_HIP, LM.RIGHT_HIP,
    LM.LEFT_KNEE, LM.RIGHT_KNEE,
    LM.LEFT_ANKLE, LM.RIGHT_ANKLE,
)


@dataclass
class MotionProfile:
    """Per-exercise expectations for landmark motion.

    All values are torso-relative, never absolute: speeds are torso-lengths per
    second, jerk is speed change (torso-lengths/s) per second, sway is the
    standard deviation of the hip centre's lateral position in torso-lengths.
    """
    max_speed: float        # typical landmark speed above which movement is "too fast"
    max_jerk: float         # speed change per second above which movement is "jerky"
    sway_tolerance: float   # lateral hip-centre drift (std dev) that counts as swaying
    window: int             # rolling window, in frames


# Dynamic exercises are allowed genuine phase movement; static poses are held,
# so their speed expectations are much lower. Balance work (tree pose) gets the
# most generous sway tolerance — a human balancing is never perfectly still,
# and treating micro-corrections as faults would misunderstand what balance is.
PROFILES: Dict[str, MotionProfile] = {
    "squat":      MotionProfile(max_speed=1.00, max_jerk=3.00, sway_tolerance=0.30,
                                window=config.MOTION_WINDOW_DYNAMIC),
    "bicep_curl": MotionProfile(max_speed=0.80, max_jerk=2.50, sway_tolerance=0.20,
                                window=config.MOTION_WINDOW_DYNAMIC),
    "warrior_2":  MotionProfile(max_speed=0.70, max_jerk=2.00, sway_tolerance=0.16,
                                window=config.MOTION_WINDOW_STATIC),
    "tree_pose":  MotionProfile(max_speed=0.65, max_jerk=2.00, sway_tolerance=0.22,
                                window=config.MOTION_WINDOW_STATIC),
}

# Middle-ground defaults for an exercise this module has never heard of.
DEFAULT_PROFILE = MotionProfile(max_speed=0.90, max_jerk=2.50, sway_tolerance=0.22,
                                window=config.MOTION_WINDOW_DYNAMIC)

# Calm, supportive wording. Movement observations, not judgments about the body.
_TEXTS: Dict[str, Tuple[str, str]] = {
    "too_fast":       ("You're moving too quickly — slow down.", "SLOW DOWN"),
    "jerky_movement": ("Try a smoother, more controlled pace.", "SMOOTH IT OUT"),
    "torso_sway":     ("Keep your torso steady.", "STEADY YOUR TORSO"),
}


def _pressure(value: float, limit: float) -> float:
    """0 inside the dead zone, ramping to 1 at twice the limit.

    The dead zone is what lets normal controlled movement score ~90–100: small
    excursions below the threshold cost nothing at all.
    """
    if limit <= 1e-9:
        return 0.0
    lo = config.MOTION_DEAD_ZONE * limit
    if value <= lo:
        return 0.0
    return float(min(1.0, (value - lo) / max(1e-9, 2.0 * limit - lo)))


class MotionAnalyzer:
    """Tracks landmark motion over a rolling window and scores its control.

    Feed it EngineFrames with `update`, read `control_score()` for a 0–100
    smoothness/consistency score, and `candidates()` for raw per-frame error
    candidates. Candidates are NOT debounced here — the caller's ErrorTracker
    owns hysteresis — but they do require the rolling window to be full and the
    pattern to hold across the window's aggregate, never one frame.
    """

    def __init__(self, profile_key: str) -> None:
        self.profile_key = profile_key
        self.profile = PROFILES.get(profile_key, DEFAULT_PROFILE)
        self.reset()

    # -- lifecycle ---------------------------------------------------------
    def reset(self) -> None:
        w = self.profile.window
        self._speeds: Deque[float] = deque(maxlen=w)   # smoothed mean landmark speed
        self._jerks: Deque[float] = deque(maxlen=w)    # |change of that speed| per second
        self._hip_x: Deque[float] = deque(maxlen=w)    # hip centre lateral position
        self._lm_speed: Dict[LM, Deque[float]] = {lm: deque(maxlen=w) for lm in TRACKED}
        self._prev_pos: Dict[LM, np.ndarray] = {}
        self._prev_t: Optional[float] = None
        self._speed_ema: Optional[float] = None
        self._history: List[float] = []                # control samples, whole session
        self._events = 0
        self._in_episode = False
        self._clean_frames = 0

    def _start_chain(self, pos: Dict[LM, np.ndarray], t: float) -> None:
        """Begin a fresh velocity chain — after a gap, a glitch, or frame one."""
        self._prev_pos = pos
        self._prev_t = t
        self._speed_ema = None

    # -- feeding -----------------------------------------------------------
    def update(self, frame: EngineFrame) -> None:
        """Track landmark positions for this frame. Safe to call every frame."""
        pose = getattr(frame, "pose", None)
        if pose is None or not frame.detected:
            return
        scale = torso_scale(pose)
        if scale is None:
            return

        t = float(frame.timestamp)
        pos: Dict[LM, np.ndarray] = {}
        for lm in TRACKED:
            got = pose.get(lm)
            if got is None or got.visibility < config.VISIBILITY_THRESHOLD:
                continue        # low confidence: this joint says nothing today
            pos[lm] = got.world.astype(np.float64) / scale

        if len(pos) < 4:
            return              # not enough confident joints for any judgment

        if (self._prev_t is None or t <= self._prev_t
                or (t - self._prev_t) > config.MOTION_MAX_GAP_SECONDS):
            self._start_chain(pos, t)
            return

        dt = t - self._prev_t
        common = [lm for lm in pos if lm in self._prev_pos]
        if not common:
            self._start_chain(pos, t)
            return

        disps = {lm: float(np.linalg.norm(pos[lm] - self._prev_pos[lm])) for lm in common}
        mean_disp = float(np.mean(list(disps.values())))
        if mean_disp > config.MOTION_TELEPORT_LIMIT:
            # A whole-body jump this large in one frame is a tracking glitch,
            # not movement. Judge nothing from it; start over from here.
            self._start_chain(pos, t)
            return

        raw_speed = mean_disp / dt
        smoothed = (raw_speed if self._speed_ema is None
                    else 0.5 * raw_speed + 0.5 * self._speed_ema)
        if self._speed_ema is not None:
            self._jerks.append(abs(smoothed - self._speed_ema) / dt)
        self._speed_ema = smoothed
        self._speeds.append(smoothed)

        for lm in common:
            self._lm_speed[lm].append(disps[lm] / dt)

        if LM.LEFT_HIP in pos and LM.RIGHT_HIP in pos:
            self._hip_x.append(float((pos[LM.LEFT_HIP][0] + pos[LM.RIGHT_HIP][0]) / 2.0))

        self._prev_pos = pos
        self._prev_t = t

        score = self.control_score()
        if score is not None:
            self._history.append(score)

    # -- internals ---------------------------------------------------------
    @property
    def window_full(self) -> bool:
        return len(self._speeds) >= self.profile.window

    def _sway(self) -> Optional[float]:
        """Lateral hip-centre drift over the window, in torso-lengths."""
        if len(self._hip_x) < max(10, self.profile.window // 2):
            return None
        return float(np.std(np.asarray(self._hip_x)))

    def _busiest_joints(self, count: int = 2) -> Tuple[LM, ...]:
        """The joints that moved most over the window — for arrows/highlights."""
        min_len = max(4, self.profile.window // 2)
        means = [(lm, float(np.mean(d))) for lm, d in self._lm_speed.items()
                 if len(d) >= min_len]
        if not means:
            return (LM.LEFT_HIP, LM.RIGHT_HIP)
        means.sort(key=lambda kv: kv[1], reverse=True)
        return tuple(lm for lm, _v in means[:count])

    # -- read-outs ---------------------------------------------------------
    def control_score(self) -> Optional[float]:
        """0–100 smoothness/consistency score. None until the window is full."""
        p = self.profile
        if not self.window_full:
            return None
        speeds = np.asarray(self._speeds)
        mean_speed = float(speeds.mean())
        spike_frac = float((speeds > p.max_speed).mean())
        mean_jerk = float(np.mean(self._jerks)) if self._jerks else 0.0
        sway = self._sway() or 0.0

        speed_term = _pressure(mean_speed, p.max_speed)
        spike_term = float(min(1.0, max(0.0, (spike_frac - 0.15) / 0.55)))
        jerk_term = _pressure(mean_jerk, p.max_jerk)
        sway_term = _pressure(sway, p.sway_tolerance)

        score = (100.0 - 30.0 * speed_term - 20.0 * spike_term
                 - 30.0 * jerk_term - 20.0 * sway_term)
        return float(max(0.0, min(100.0, score)))

    def candidates(self, phase: str, now: float) -> List[FormError]:
        """Raw per-frame candidates for the caller's ErrorTracker to debounce.

        At most one candidate per frame — the worst offender — and only when the
        rolling window is full and the pattern holds across the window aggregate
        (mean, backed by the median so one wild frame can never fire anything).
        """
        p = self.profile
        if not self.window_full:
            self._note_frame(False)
            return []

        speeds = np.asarray(self._speeds)
        mean_speed = float(speeds.mean())
        med_speed = float(np.median(speeds))
        jerks = np.asarray(self._jerks) if self._jerks else np.zeros(1)
        mean_jerk = float(jerks.mean())
        med_jerk = float(np.median(jerks))
        sway = self._sway()

        issues: List[Tuple[str, float]] = []
        if mean_speed > p.max_speed and med_speed > 0.75 * p.max_speed:
            issues.append(("too_fast", mean_speed / p.max_speed))
        if mean_jerk > p.max_jerk and med_jerk > 0.75 * p.max_jerk:
            issues.append(("jerky_movement", mean_jerk / p.max_jerk))
        if sway is not None and sway > p.sway_tolerance:
            issues.append(("torso_sway", sway / p.sway_tolerance))

        if not issues:
            self._note_frame(False)
            return []

        key, ratio = max(issues, key=lambda kv: kv[1])
        message, cue = _TEXTS[key]
        landmarks = ((LM.LEFT_HIP, LM.RIGHT_HIP) if key == "torso_sway"
                     else self._busiest_joints())
        self._note_frame(True)
        return [FormError(
            key=key,
            message=message,
            cue=cue,
            severity=Severity.MAJOR if ratio > 1.6 else Severity.MINOR,
            priority=Priority.STABILITY,
            confidence=0.85,
            landmarks=landmarks,
            phase=phase,
        )]

    def _note_frame(self, fired: bool) -> None:
        """Episode bookkeeping: an episode = the pattern began after a clean spell."""
        if fired:
            if not self._in_episode:
                self._events += 1
                self._in_episode = True
            self._clean_frames = 0
        else:
            self._clean_frames += 1
            if self._clean_frames >= config.MOTION_EPISODE_CLEAR_FRAMES:
                self._in_episode = False

    @property
    def unstable_events(self) -> int:
        """Distinct unstable episodes this session, for the summary."""
        return self._events

    def improved(self) -> Optional[bool]:
        """Did control improve from the first third to the last third of the session?

        None when there is not enough data to say anything honest.
        """
        n = len(self._history)
        if n < 30:
            return None
        third = n // 3
        first = float(np.mean(self._history[:third]))
        last = float(np.mean(self._history[-third:]))
        return bool(last >= first + 1.0)
