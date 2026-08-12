"""Tunables in one place so demo-day adjustments never mean hunting through modules."""
from __future__ import annotations

# --- detection -------------------------------------------------------------
MODEL_COMPLEXITY = 1              # 0 fastest / 1 balanced / 2 most accurate
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# --- confidence gating -----------------------------------------------------
# Below this, a landmark is treated as UNMEASURED rather than trusted.
VISIBILITY_THRESHOLD = 0.6
# A metric needs at least this confidence before we are willing to speak about it.
METRIC_SPEAK_THRESHOLD = 0.65

# --- missing-landmark tolerance -------------------------------------------
# How many frames we keep re-using a landmark's last known position before
# declaring it genuinely gone. Covers brief occlusion without freezing forever.
MAX_STALE_FRAMES = 8

# --- smoothing (One Euro filter) ------------------------------------------
# Lower min_cutoff = smoother but laggier. beta trades jitter for responsiveness.
ONE_EURO_MIN_CUTOFF = 1.2
ONE_EURO_BETA = 0.02
ONE_EURO_D_CUTOFF = 1.0

# --- body map calibration --------------------------------------------------
CALIBRATION_SECONDS = 6.0
# Fraction of calibration frames a landmark must appear in to count as trackable.
TRACKABLE_HIT_RATIO = 0.7

# --- exercise auto-detection ----------------------------------------------
AUTODETECT_MIN_SCORE = 0.55       # below this we ask the user to pick manually
AUTODETECT_MIN_MARGIN = 0.12      # winner must beat runner-up by this much
AUTODETECT_WINDOW = 20            # frames of voting before we commit

# --- error debounce --------------------------------------------------------
# An error must persist this many frames before the user hears about it, and be
# gone this many before it clears. Asymmetric on purpose: slow to complain,
# slower to forget. This is what stops pose jitter becoming a stuttering coach.
ERROR_ON_FRAMES = 5
ERROR_OFF_FRAMES = 10

# --- ghost coach -----------------------------------------------------------
SHOW_GHOST_DEFAULT = True
GHOST_ARROW_MIN_PIXELS = 22.0     # below this the joint is close enough; no arrow

# --- camera ----------------------------------------------------------------
CAMERA_INDEX = 0
FRAME_WIDTH = 960
FRAME_HEIGHT = 720

# --- movement control (stability & controlled-movement coach) ---------------
# All motion values are torso-relative landmark measurements — never physical
# force. Per-exercise thresholds live in core/motion.py PROFILES.
MOTION_WINDOW_DYNAMIC = 24        # rolling window, frames (~1 s at 24 fps)
MOTION_WINDOW_STATIC = 30         # holds get a slightly longer look
MOTION_TELEPORT_LIMIT = 0.5       # mean per-frame jump (torso-lengths) treated as a tracking glitch
MOTION_MAX_GAP_SECONDS = 0.4      # a longer gap breaks the velocity chain instead of spiking it
MOTION_DEAD_ZONE = 0.6            # fraction of each limit inside which control still scores ~100
MOTION_EPISODE_CLEAR_FRAMES = 12  # clean frames before an unstable episode is considered over
