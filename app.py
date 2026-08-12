Library
/
app_updated_comfort_reps_timer.py


"""MoveWise — adaptive AI movement coach.

    streamlit run app.py

Screens: Home -> Library / Auto-detect -> Calibration -> Live analysis ->
Session summary -> Progress.
"""
from __future__ import annotations

import sys
import re
import sqlite3
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import db_handler as db

from core import config
from core.bodygroups import group_status
from core.bodymap import BodyMode
from core.coaching import AdaptiveCoach, Modality
from core.engine import PoseEngine
from core.reference import alignment_status, correction_arrow, fit_reference
from core.session import SessionRecorder
from core.source import CameraSource, VideoSource
from core.storage import ProgressStore
from core.synthetic import CLIPS, SyntheticSource
from exercises.registry import ExerciseRecognizer, ExerciseRegistry
from ui import overlay, theme

st.set_page_config(page_title="MoveWise", page_icon="🧍", layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown(theme.CSS, unsafe_allow_html=True)

HOME, LIBRARY, DETECT, CALIBRATE, LIVE, SUMMARY, PROGRESS, ACCOUNT = (
    "home", "library", "detect", "calibrate", "live", "summary", "progress", "account")

LIBRARY_GROUPS = [
    ("Yoga", [("warrior_2", "△", "Static hold. Front knee, hips, shoulders, arm line "
                               "and torso, all checked independently."),
              ("tree_pose", "⊥", "Balance hold. Standing-leg stability and body sway, "
                               "with a tolerance that forgives natural micro-movement.")]),
    ("Fitness", [("squat", "◇", "Full rep cycle through descent, bottom and ascent. "
                                "Depth, knee tracking, back angle and symmetry."),
                 ("bicep_curl", "◈", "Per-arm reps. Range of motion, elbow stability "
                                "and the body swing that means momentum took over.")]),
]

PHASE_TRACKS = {
    "squat": ["standing", "descending", "bottom", "ascending"],
    "bicep_curl": ["start", "curl", "peak", "return"],
}

# --------------------------------------------------------------------------
# EARLY SESSION-STATE INITIALIZATION
# --------------------------------------------------------------------------
# These values must exist before source_picker()/match_clip_to_exercise()
# can access them.
for _key, _default in {
    "clip_key": None,
    "exercise": None,
    "authenticated": False,
    "user_id": None,
    "username": "",
    "app_rep_count": 0,
    "app_rep_phase": None,
    "app_rep_started": False,
    "comfort_variation": None,
    "yoga_hold_started_at": None,
    "yoga_hold_seconds": 0.0,
}.items():
    if _key not in st.session_state:
        st.session_state[_key] = _default

# --------------------------------------------------------------------------
# SESSION STATE & AUTHENTICATION INITIALIZATION
# --------------------------------------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None
if "username" not in st.session_state:
    st.session_state["username"] = ""
if "user_profile" not in st.session_state:
    st.session_state["user_profile"] = None
if "editing_profile" not in st.session_state:
    st.session_state["editing_profile"] = False

def S():
    s = st.session_state
    if "screen" not in s:
        s.screen = HOME
        s.registry = ExerciseRegistry()
        s.engine = PoseEngine()
        s.recognizer = ExerciseRecognizer(s.registry)
        s.coach = AdaptiveCoach()
        s.recorder = None
        s.store = ProgressStore()
        s.exercise = None
        s.summary = None
        s.source_mode = "Sample clip"
        s.clip_key = CLIPS[0].key
        s.upload = None
        s.ghost = True
        s.body_map = None
        s.last_coaching = None
        s.live_src = None
        s.app_rep_count = 0
        s.app_rep_phase = None
        s.app_rep_started = False
        s.comfort_variation = None
        s.yoga_hold_started_at = None
        s.yoga_hold_seconds = 0.0
    return s

s = S()

def go(screen: str) -> None:
    s.screen = screen
    st.rerun()

def header(title: str, sub: str = "", back: str = None) -> None:
    c1, c2 = st.columns([5, 1])
    with c1:
        st.markdown(theme.page_head(title, sub), unsafe_allow_html=True)
    with c2:
        st.write("")
        if back and st.button("← Back", key=f"back_{title}"):
            go(back)

def disclaimer() -> None:
    st.markdown(theme.note(theme.DISCLAIMER), unsafe_allow_html=True)

def source_picker(where: str) -> None:
    s.source_mode = st.radio("Input", ["Webcam", "Video file", "Sample clip"],
                             horizontal=True, key=f"src_{where}",
                             index=["Webcam", "Video file", "Sample clip"].index(s.source_mode))
    if s.source_mode == "Sample clip":
        match_clip_to_exercise()
        titles = {c.key: c.title for c in CLIPS}
        if s.clip_key not in titles:
            s.clip_key = next(iter(titles))
        s.clip_key = st.selectbox("Clip", list(titles), format_func=lambda k: titles[k],
                                  key=f"clip_{where}", index=list(titles).index(s.clip_key))
        picked = next((c for c in CLIPS if c.key == s.clip_key), None)
        if picked is not None and s.exercise is not None and picked.exercise_key != s.exercise:
            s.exercise = picked.exercise_key
    elif s.source_mode == "Video file":
        up = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv", "webm"],
                              key=f"up_{where}")
        if up is not None:
            p = Path(tempfile.gettempdir()) / f"movewise_{up.name}"
            p.write_bytes(up.getbuffer())
            s.upload = str(p)
        if s.upload:
            st.caption(f"Loaded {Path(s.upload).name}")

def match_clip_to_exercise() -> None:
    if s.exercise is None or s.source_mode != "Sample clip":
        return
    clip_key = st.session_state.get("clip_key")
    current = next((c for c in CLIPS if c.key == clip_key), None)
    if current is not None and current.exercise_key == s.exercise:
        return
    fallback = next((c for c in CLIPS if c.exercise_key == s.exercise), None)
    if fallback is not None:
        s.clip_key = fallback.key
        for widget_key in [k for k in st.session_state if isinstance(k, str) and k.startswith("clip_")]:
            del st.session_state[widget_key]

def _clear_live_panel_cache() -> None:
    for key in ("last_cue_html", "last_strip_html", "last_ring_html", "last_metric_html"):
        if key in st.session_state:
            del st.session_state[key]

def finish_session() -> None:
    if s.recorder is None:
        _clear_live_panel_cache()
        go(LIBRARY)
        return
    summary = s.recorder.finish()
    if getattr(s, "app_rep_count", 0) and getattr(summary, "movement", "") == "dynamic":
        summary.reps = max(int(getattr(summary, "reps", 0) or 0), int(s.app_rep_count))
    elif getattr(s, "app_rep_count", 0) and s.exercise in ("squat", "bicep_curl"):
        summary.reps = max(int(getattr(summary, "reps", 0) or 0), int(s.app_rep_count))
    summary.corrections = s.coach.total_attempts
    summary.successful_corrections = s.coach.successes
    if getattr(summary, "control_score", None) is None:
        samples = getattr(s.recorder, "_control_samples", None) or []
        if samples:
            summary.control_score = float(sum(samples) / len(samples))
    if not getattr(summary, "unstable_events", 0):
        summary.unstable_events = int(getattr(s.recorder, "unstable_events", 0) or 0)
    
    # Save local store + Database
    s.store.save(summary)
    if st.session_state["authenticated"]:
        user_id = st.session_state["user_id"]
        if hasattr(db, "save_exercise_summary"):
            db.save_exercise_summary(user_id, summary)
        elif hasattr(db, "save_exercise_log"):
            session_data = {
                "exercise_name": getattr(summary, "exercise_name", "Unknown"),
                "movement_score": getattr(summary, "movement_score", None),
                "control_score": getattr(summary, "control_score", None),
                "reps": getattr(summary, "reps", 0) or 0,
                "good_reps": getattr(summary, "good_reps", 0) or 0,
                "main_issue": getattr(summary, "main_issue", "") or "",
                "unstable_events": getattr(summary, "unstable_events", 0) or 0,
                "metric_scores": getattr(summary, "metric_scores", {}) or {},
            }
            db.save_exercise_log(user_id, session_data)

    s.summary = summary
    s.recorder = None
    _clear_live_panel_cache()
    go(SUMMARY)

# --------------------------------------------------------------------------
# MOVEMENT-SAFETY / COUNTING HELPERS
# --------------------------------------------------------------------------
def easier_variation(exercise_key: str) -> dict:
    """Safe, simple alternatives shown when a user reports discomfort."""
    variations = {
        "warrior_2": {
            "name": "Supported Warrior II",
            "hint": "Take a smaller stance, bend the front knee less, and use a wall or chair for support. Hold only 5–7 seconds, then relax."
        },
        "tree_pose": {
            "name": "Supported Tree Pose",
            "hint": "Keep the toes of the raised foot on the floor or place the foot at the ankle. Keep one hand on a wall for balance and hold 5–7 seconds."
        },
        "squat": {
            "name": "Chair-Assisted Squat",
            "hint": "Place a chair behind you, squat only as far as comfortable, lightly tap the chair, then stand. Keep the movement slow and controlled."
        },
        "bicep_curl": {
            "name": "Seated Light Curl",
            "hint": "Sit down, use a lighter load (or no load), keep your elbows close to your body, and use a smaller comfortable range of motion."
        },
    }
    return variations.get(exercise_key, {
        "name": "Reduced-Range Version",
        "hint": "Reduce the range of motion, slow down, and use support if needed. Stop if the movement remains uncomfortable."
    })


def update_app_rep_counter(result) -> int:
    """Independent phase-based fallback counter for the two dynamic exercises.

    It complements the exercise analyser instead of replacing it. A rep is
    completed only after the full movement cycle returns to its start phase.
    """
    if getattr(result, "movement", None) != "dynamic":
        return int(getattr(s, "app_rep_count", 0) or 0)

    phase = str(getattr(result, "phase", "") or "").lower()
    exercise = getattr(s, "exercise", None)

    if exercise == "squat":
        start, finish, peak = "standing", "standing", "bottom"
        active_phases = {"descending", "bottom", "ascending"}
    elif exercise == "bicep_curl":
        start, finish, peak = "start", "start", "peak"
        active_phases = {"curl", "peak", "return"}
    else:
        return int(getattr(s, "app_rep_count", 0) or 0)

    previous = getattr(s, "app_rep_phase", None)
    if phase == peak:
        s.app_rep_started = True
    if s.app_rep_started and previous in active_phases and phase == finish:
        # Prevent duplicate increments while the analyser stays in the same
        # return/start phase for several frames.
        if previous != finish:
            s.app_rep_count = int(getattr(s, "app_rep_count", 0) or 0) + 1
        s.app_rep_started = False
    s.app_rep_phase = phase
    return int(getattr(s, "app_rep_count", 0) or 0)


def update_yoga_hold_timer(result) -> float:
    """Stable-form timer for yoga holds, capped at a 5–7 second target.

    The timer runs only while the pose is ready and no primary correction is
    active, so a broken pose does not falsely accumulate hold time.
    """
    if getattr(result, "movement", None) != "static":
        return 0.0
    now = time.monotonic()
    good = bool(getattr(result, "ready", False)) and getattr(result, "primary_error", None) is None
    if good:
        if getattr(s, "yoga_hold_started_at", None) is None:
            s.yoga_hold_started_at = now
        s.yoga_hold_seconds = min(7.0, max(0.0, now - s.yoga_hold_started_at))
    else:
        s.yoga_hold_started_at = None
        s.yoga_hold_seconds = 0.0
    return float(s.yoga_hold_seconds)


def reset_movement_tracking() -> None:
    s.app_rep_count = 0
    s.app_rep_phase = None
    s.app_rep_started = False
    s.comfort_variation = None
    s.yoga_hold_started_at = None
    s.yoga_hold_seconds = 0.0


def build_source():
    if s.source_mode == "Webcam":
        return CameraSource()
    if s.source_mode == "Video file":
        return VideoSource(s.upload, loop=False, stride=1) if s.upload else None
    match_clip_to_exercise()
    clip = next(c for c in CLIPS if c.key == s.clip_key)
    return SyntheticSource(clip, loop=False)

def _release_live_src() -> None:
    src = getattr(s, "live_src", None)
    if src is not None:
        try:
            src.release()
        except Exception:
            pass
        s.live_src = None

# --------------------------------------------------------------------------
# AUTHENTICATION HELPERS
# --------------------------------------------------------------------------
def valid_email(email: str) -> bool:
    pattern = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
    return bool(re.fullmatch(pattern, email.strip()))


def password_is_strong(password: str) -> bool:
    return bool(
        len(password) >= 8
        and re.search(r"[A-Z]", password)
        and re.search(r"[a-z]", password)
        and re.search(r"\d", password)
        and re.search(r"[^A-Za-z0-9]", password)
    )


def password_strength(password: str) -> str:
    if not password:
        return ""
    if len(password) < 8:
        return "Weak — use at least 8 characters."
    if not re.search(r"[A-Z]", password):
        return "Medium — add an uppercase letter."
    if not re.search(r"[a-z]", password):
        return "Medium — add a lowercase letter."
    if not re.search(r"\d", password):
        return "Medium — add a number."
    if not re.search(r"[^A-Za-z0-9]", password):
        return "Medium — add a special character."
    return "Strong ✓"


def _db_path() -> Path:
    """Use the same database file as db_handler.py."""
    return Path(__file__).resolve().parent / "database.db"


def get_user_profile(user_id):
    if not user_id:
        return None
    try:
        with sqlite3.connect(_db_path()) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT id, username, email, age, fitness_goal, experience_level
                FROM users WHERE id = ?
                """,
                (user_id,),
            ).fetchone()
            return dict(row) if row else None
    except Exception:
        return None


def update_user_profile(user_id, email, age, fitness_goal, experience_level):
    if not valid_email(email):
        return False, "Please enter a valid email address."
    try:
        with sqlite3.connect(_db_path()) as conn:
            conn.execute(
                """
                UPDATE users
                SET email = ?, age = ?, fitness_goal = ?, experience_level = ?
                WHERE id = ?
                """,
                (
                    email.strip().lower(),
                    int(age),
                    fitness_goal,
                    experience_level,
                    user_id,
                ),
            )
            conn.commit()
        return True, "Profile updated successfully."
    except sqlite3.IntegrityError:
        return False, "That email is already associated with another account."
    except Exception as exc:
        return False, f"Could not update profile: {exc}"


def register_account(username, email, password, age, fitness_goal, experience_level):
    """Support both the old 3-argument and newer 6-argument db_handler."""
    try:
        try:
            result = db.register_user(
                username, email, password, age, fitness_goal, experience_level
            )
        except TypeError:
            result = db.register_user(username, email, password)
            if result[0]:
                with sqlite3.connect(_db_path()) as conn:
                    conn.execute(
                        """
                        UPDATE users
                        SET age = ?, fitness_goal = ?, experience_level = ?
                        WHERE username = ?
                        """,
                        (int(age), fitness_goal, experience_level, username.strip()),
                    )
                    conn.commit()
        return result
    except Exception as exc:
        return False, f"Could not create account: {exc}"


def logout_user() -> None:
    _release_live_src()
    st.session_state["authenticated"] = False
    st.session_state["user_id"] = None
    st.session_state["username"] = ""
    st.session_state["user_profile"] = None
    st.session_state["editing_profile"] = False
    s.screen = HOME
    st.rerun()


def render_auth_page():
    st.markdown(theme.wordmark(), unsafe_allow_html=True)
    st.write("")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown(
            "<div class='mw-glass' style='padding:32px;border-radius:24px;'>",
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div style="text-align:center;margin-bottom:22px;">
                <div style="font-size:34px;font-weight:800;">MoveWise</div>
                <div style="opacity:.72;margin-top:5px;">
                    Your AI-powered movement & posture coach
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        tab1, tab2 = st.tabs(["Sign In", "Create Account"])

        with tab1:
            st.subheader("Welcome back 👋")
            st.caption("Sign in to continue your movement journey.")
            login_user = st.text_input(
                "Username", placeholder="Enter your username", key="login_user"
            )
            login_pass = st.text_input(
                "Password", type="password", placeholder="Enter your password", key="login_pass"
            )
            if st.button("Sign In →", type="primary", use_container_width=True, key="signin_button"):
                if not login_user or not login_pass:
                    st.warning("Please enter both username and password.")
                else:
                    ok, res = db.authenticate_user(login_user, login_pass)
                    if ok:
                        st.session_state["authenticated"] = True
                        st.session_state["user_id"] = res["id"]
                        st.session_state["username"] = res["username"]
                        st.session_state["user_profile"] = dict(res)
                        st.success(f"Welcome back, {res['username']}! 👋")
                        st.rerun()
                    else:
                        st.error(res)

        with tab2:
            st.subheader("Create your MoveWise account")
            st.caption("A few details help personalize your coaching experience.")
            reg_user = st.text_input(
                "Username", placeholder="Choose a username", key="reg_user"
            )
            reg_email = st.text_input(
                "Email", placeholder="you@example.com", key="reg_email"
            )
            age = st.number_input(
                "Age", min_value=13, max_value=100, value=20, step=1, key="reg_age"
            )
            fitness_goal = st.selectbox(
                "Main fitness goal",
                [
                    "Improve posture",
                    "Build strength",
                    "Improve flexibility",
                    "Improve mobility",
                    "General fitness",
                    "Physiotherapy / rehabilitation",
                ],
                key="reg_goal",
            )
            experience_level = st.selectbox(
                "Experience level",
                ["Beginner", "Intermediate", "Advanced"],
                key="reg_experience",
            )
            reg_pass = st.text_input(
                "Password", type="password", placeholder="At least 8 characters", key="reg_pass"
            )
            if reg_pass:
                strength = password_strength(reg_pass)
                if strength == "Strong ✓":
                    st.success("Password strength: Strong ✓")
                elif strength.startswith("Medium"):
                    st.warning(f"Password strength: {strength}")
                else:
                    st.error(f"Password strength: {strength}")

            confirm_pass = st.text_input(
                "Confirm password", type="password", placeholder="Re-enter your password", key="confirm_pass"
            )

            if st.button(
                "Create Account →", type="primary", use_container_width=True, key="signup_button"
            ):
                if len(reg_user.strip()) < 3:
                    st.error("Username must contain at least 3 characters.")
                elif not valid_email(reg_email):
                    st.error("Please enter a valid email address.")
                elif not password_is_strong(reg_pass):
                    st.error(
                        "Password must contain 8+ characters, uppercase, lowercase, number and special character."
                    )
                elif reg_pass != confirm_pass:
                    st.error("Passwords do not match.")
                else:
                    ok, msg = register_account(
                        reg_user, reg_email, reg_pass, age, fitness_goal, experience_level
                    )
                    if ok:
                        st.success("Account created successfully! You can now sign in. 🎉")
                    else:
                        st.error(msg)

        st.markdown("</div>", unsafe_allow_html=True)
    disclaimer()


# --------------------------------------------------------------------------
# AUTHENTICATION GATE
# --------------------------------------------------------------------------
if not st.session_state["authenticated"]:
    render_auth_page()
    st.stop()


# --------------------------------------------------------------------------
# PROFILE UI
# --------------------------------------------------------------------------
def render_profile_card(show_logout: bool = False):
    user_id = st.session_state.get("user_id")
    profile = get_user_profile(user_id)
    if not profile:
        st.warning("Your profile could not be loaded. Please sign in again.")
        return

    st.session_state["user_profile"] = profile

    st.markdown(theme.eyebrow("Your MoveWise profile"), unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="mw-glass" style="padding:24px;margin:12px 0 22px;">
            <div style="font-size:28px;font-weight:800;">👤 {profile.get('username', 'User')}</div>
            <div style="opacity:.68;margin-top:4px;">Personalized movement profile</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("✏️ Edit Profile", use_container_width=True, key=f"edit_profile_{show_logout}"):
        st.session_state["editing_profile"] = True
        st.rerun()

    if st.session_state.get("editing_profile", False):
        st.markdown("### Edit profile")
        st.caption("Username cannot be changed here because it identifies your account.")

        email = st.text_input("Email", value=profile.get("email") or "", key=f"profile_email_{show_logout}")
        age = st.number_input(
            "Age", min_value=13, max_value=100,
            value=int(profile.get("age") or 20), step=1,
            key=f"profile_age_{show_logout}",
        )
        goals = [
            "Improve posture", "Build strength", "Improve flexibility",
            "Improve mobility", "General fitness", "Physiotherapy / rehabilitation",
        ]
        current_goal = profile.get("fitness_goal")
        goal_index = goals.index(current_goal) if current_goal in goals else 0
        goal = st.selectbox("Main fitness goal", goals, index=goal_index, key=f"profile_goal_{show_logout}")

        levels = ["Beginner", "Intermediate", "Advanced"]
        current_level = profile.get("experience_level")
        level_index = levels.index(current_level) if current_level in levels else 0
        level = st.selectbox("Experience level", levels, index=level_index, key=f"profile_level_{show_logout}")

        save_col, cancel_col = st.columns(2)
        with save_col:
            if st.button("💾 Save Changes", type="primary", use_container_width=True, key=f"save_profile_{show_logout}"):
                ok, msg = update_user_profile(user_id, email, age, goal, level)
                if ok:
                    st.session_state["editing_profile"] = False
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
        with cancel_col:
            if st.button("Cancel", use_container_width=True, key=f"cancel_profile_{show_logout}"):
                st.session_state["editing_profile"] = False
                st.rerun()
    else:
        p1, p2, p3 = st.columns(3)
        p1.markdown(theme.stat(profile.get("username") or "—", "Username"), unsafe_allow_html=True)
        p2.markdown(theme.stat(profile.get("email") or "—", "Email"), unsafe_allow_html=True)
        p3.markdown(theme.stat(str(profile.get("age") or "Not set"), "Age"), unsafe_allow_html=True)
        p4, p5 = st.columns(2)
        p4.markdown(theme.stat(profile.get("fitness_goal") or "Not set", "Fitness goal"), unsafe_allow_html=True)
        p5.markdown(theme.stat(profile.get("experience_level") or "Not set", "Experience"), unsafe_allow_html=True)

    if show_logout:
        st.write("")
        st.markdown(theme.eyebrow("Account actions"), unsafe_allow_html=True)
        if st.button("🚪 Log out of MoveWise", type="primary", use_container_width=True, key="account_logout"):
            logout_user()


# --------------------------------------------------------------------------
# TOP ACCOUNT ACCESS
# --------------------------------------------------------------------------
def account_access():
    if st.button("👤 Account", key="global_account", use_container_width=True):
        go(ACCOUNT)


# --------------------------------------------------------------------------
# HOME
# --------------------------------------------------------------------------
if s.screen == HOME:
    st.markdown(theme.wordmark(), unsafe_allow_html=True)
    st.write("")
    top_left, top_right = st.columns([5, 1])
    with top_right:
        account_access()

    cols = st.columns(4)
    tiles = [
        ("◎", "Auto-detect", "Start moving and MoveWise works out which of the four exercises you're performing.", "Recognise", DETECT),
        ("▤", "Library", "Two yoga holds and two gym movements, each with its own analysis profile.", "Browse", LIBRARY),
        ("◐", "Adaptive Mode", "Calibrates to the body you have — a missing landmark is never an error.", "Calibrate", CALIBRATE),
        ("◭", "My Progress", "Scores, trends and your movement profile across sessions.", "Review", PROGRESS),
    ]
    for col, (ico, title, body, foot, dest) in zip(cols, tiles):
        with col:
            st.markdown(theme.tile(ico, title, body, foot), unsafe_allow_html=True)
            st.write("")
            if st.button(foot, key=f"home_{dest}"):
                go(dest)

    st.write("")
    user_sessions = db.get_user_sessions(st.session_state["user_id"])
    n = len(user_sessions)
    if n:
        st.markdown(
            f"<p style='text-align:center;color:{theme.MUTED};font-size:.8rem;letter-spacing:.16em;text-transform:uppercase;margin-top:18px'>{n} session{'s' if n != 1 else ''} saved in your database account</p>",
            unsafe_allow_html=True,
        )
    disclaimer()


# --------------------------------------------------------------------------
# LIBRARY
# --------------------------------------------------------------------------
elif s.screen == LIBRARY:
    header("Exercise Library", "Four movements, each with its own analysis profile.", HOME)
    for group, items in LIBRARY_GROUPS:
        st.markdown(theme.eyebrow(group), unsafe_allow_html=True)
        cols = st.columns(2)
        for col, (key, ico, desc) in zip(cols, items):
            prof = s.registry.get(key)
            with col:
                st.markdown(theme.tile(ico, prof.name, desc, prof.movement.value + " · " + prof.category.value), unsafe_allow_html=True)
                st.write("")
                if st.button("Select", key=f"lib_{key}"):
                    s.exercise = key
                    go(CALIBRATE)
        st.write("")
    disclaimer()


# --------------------------------------------------------------------------
# AUTO-DETECT
# --------------------------------------------------------------------------
elif s.screen == DETECT:
    header("Auto-detect my exercise", "Get into position — recognition runs over a short window so one odd frame can't flip the answer.", HOME)
    source_picker("detect")
    st.info("Recognition covers Warrior II, Tree Pose, Squat and Bicep Curl. If confidence is low you'll be asked to choose manually.")
    c1, c2 = st.columns(2)
    if c1.button("Start detection", type="primary"):
        s.exercise = None
        go(CALIBRATE)
    if c2.button("Choose Exercise Manually"):
        go(LIBRARY)
    disclaimer()


# --------------------------------------------------------------------------
# CALIBRATION
# --------------------------------------------------------------------------
elif s.screen == CALIBRATE:
    name = s.registry.get(s.exercise).name if s.exercise else "Auto-detect"
    header("Calibration", f"{name} — building your Personal Body Map.", HOME)
    source_picker("cal")
    st.markdown("Stand so your whole body is in frame. This takes a few seconds and decides which measurements apply to **your** body.")
    if st.button("Run calibration", type="primary"):
        src = build_source()
        if src is None or not src.open():
            st.error("Could not open that input. Try **Sample clip** — it needs no hardware.")
        else:
            s.engine.reset()
            if not src.needs_detection:
                s.engine.calibrator.duration = 1.5
            bar = st.progress(0.0, text="Detecting landmarks…")
            slot = st.empty()
            try:
                for _ in range(600):
                    item = src.read()
                    if item is None or item.finished or item.image is None:
                        break
                    f = s.engine.process(item.image) if src.needs_detection else s.engine.process_pose(item.pose)
                    bar.progress(min(1.0, f.calibration_progress), text="Detecting landmarks…")
                    slot.image(item.image, channels="BGR", width=340)
                    if s.engine.body_map is not None:
                        break
            finally:
                src.release()
            s.body_map = s.engine.body_map
            bar.empty()
            slot.empty()
    if s.body_map is not None:
        bm = s.body_map
        st.markdown(theme.eyebrow("Personal Body Map"), unsafe_allow_html=True)
        chips = "".join(theme.chip(g, ok) for g, ok in group_status(bm))
        st.markdown(f"<div class='mw-glass' style='padding:18px 20px'>{chips}</div>", unsafe_allow_html=True)
        st.write("")
        if bm.mode is BodyMode.ADAPTIVE:
            st.warning("**Adaptive Mode** — some landmarks aren't trackable. Those measurements are switched off rather than counted against you. A landmark we can't see is never a posture error.")
        else:
            st.success("Standard mode — full landmark set available.")
        if st.button("Continue to live analysis", type="primary"):
            s.coach.reset()
            s.registry.reset_all()
            s.recognizer.reset()
            prof0 = s.registry.get(s.exercise) if s.exercise else None
            s.recorder = SessionRecorder(exercise=s.exercise or "", exercise_name=prof0.name if prof0 else "Auto-detect", adaptive_mode=(s.body_map is not None and s.body_map.mode is BodyMode.ADAPTIVE))
            _release_live_src()
            _clear_live_panel_cache()
            go(LIVE)
    disclaimer()


# --------------------------------------------------------------------------
# LIVE ANALYSIS
# --------------------------------------------------------------------------
elif s.screen == LIVE:
    profile = s.registry.get(s.exercise) if s.exercise else None
    head_slot = st.empty()

    def _set_header(name: str) -> None:
        head_slot.markdown(theme.page_head(name, "One correction at a time."), unsafe_allow_html=True)

    _set_header(profile.name if profile else "Identifying exercise…")
    bar = st.columns([0.9, 1, 1, 1, 1, 0.9, 1.2])
    if bar[0].button("← Back", key="live_back"):
        _release_live_src()
        go(LIBRARY)
    for i, ex_key in enumerate(("warrior_2", "tree_pose", "squat", "bicep_curl")):
        ex_prof = s.registry.get(ex_key)
        active = st.session_state.get("exercise") == ex_key
        if bar[i + 1].button(ex_prof.name, key=f"live_switch_{ex_key}", type="primary" if active else "secondary") and not active:
            st.session_state["exercise"] = ex_key
            ex_prof.reset()
            s.coach.reset()
            reset_movement_tracking()
            s.recorder = SessionRecorder(exercise=ex_key, exercise_name=ex_prof.name, adaptive_mode=(s.body_map is not None and s.body_map.mode is BodyMode.ADAPTIVE))
            match_clip_to_exercise()
            _release_live_src()
            st.rerun()
    if bar[5].button(("👻 On" if s.ghost else "👻 Off"), key="live_ghost", help="Toggle the reference guide"):
        s.ghost = not s.ghost
        st.rerun()
    if bar[6].button("⏹ End Session", key="live_end"):
        _release_live_src()
        finish_session()

    top = st.columns([4, 2])
    video_slot = top[0].empty()
    side = top[1]
    cue_slot = side.empty()
    strip_slot = side.empty()
    stat_slot = side.empty()
    metric_slot = side.empty()
    action_slot = side.container()
    status_slot = st.empty()

    _lc = getattr(s, "last_cue_html", None)
    _ls = getattr(s, "last_strip_html", None)
    _lr = getattr(s, "last_ring_html", None)
    _lm = getattr(s, "last_metric_html", None)
    if _lc:
        cue_slot.markdown(_lc, unsafe_allow_html=True)
    if _ls:
        strip_slot.markdown(_ls, unsafe_allow_html=True)
    if _lr:
        stat_slot.markdown(_lr, unsafe_allow_html=True)
    if _lm:
        metric_slot.markdown(_lm, unsafe_allow_html=True)

    src = getattr(s, "live_src", None)
    if src is None:
        src = build_source()
        if src is None or not src.open():
            st.error("Could not open that input. Go back and pick **Sample clip** — it needs no camera.")
            if st.button("← Back", key="live_err_back"):
                go(CALIBRATE)
            st.stop()
        s.live_src = src

    if s.body_map is not None:
        s.engine.force_body_map(s.body_map)
    if not src.needs_detection:
        s.engine.calibrator.duration = 0.5

    recorder: SessionRecorder = s.recorder or SessionRecorder(exercise=s.exercise or "", exercise_name=profile.name if profile else "Auto-detect")
    s.recorder = recorder
    coach: AdaptiveCoach = s.coach
    _loop_done = False
    try:
        while True:
            item = src.read()
            if item is None:
                continue
            if item.finished:
                _loop_done = True
                break
            if item.image is None:
                continue
            frame = s.engine.process(item.image) if src.needs_detection else s.engine.process_pose(item.pose)
            if profile is None:
                rec = s.recognizer.observe(frame)
                if rec.confident and rec.key:
                    s.exercise = rec.key
                    profile = s.registry.get(rec.key)
                    _set_header(profile.name)
                else:
                    canvas = item.image.copy()
                    overlay.draw_skeleton(canvas, frame.pose)
                    overlay.draw_banner(canvas, "Identifying exercise", rec.message)
                    video_slot.image(canvas, channels="BGR", use_container_width=True)
                    continue
            result = profile.analyse(frame)
            app_reps = update_app_rep_counter(result)
            yoga_hold = update_yoga_hold_timer(result)
            coaching = coach.update(result, frame.timestamp)
            recorder.update(result, frame.timestamp)
            s.last_coaching = coaching
            canvas = item.image.copy()
            primary = result.primary_error
            ghost = None
            emphasise = coaching.modality is Modality.GHOST_EMPHASIS
            if s.ghost and result.ready:
                ghost = fit_reference(profile.key, frame.pose, canvas.shape, progress=result.reference_progress, mirror=getattr(profile, "front_side", None) == "right")
                overlay.draw_ghost(canvas, ghost, emphasis=emphasise)
            overlay.draw_skeleton(canvas, frame.pose, highlight=primary.landmarks if primary else ())
            if primary and ghost and coaching.modality in (Modality.ARROW, Modality.GHOST_EMPHASIS):
                for lm in primary.landmarks[:1]:
                    overlay.draw_arrow(canvas, correction_arrow(frame.pose, ghost, lm, canvas.shape, config.GHOST_ARROW_MIN_PIXELS), primary.cue)
            if frame.calibrating:
                overlay.draw_calibration(canvas, frame.calibration_progress)
            elif result.ready:
                overlay.draw_banner(canvas, coaching.message or "Good form", f"{profile.name} · {result.phase}", fault=primary is not None)
                overlay.draw_score_badge(
                    canvas, result.score, result.phase,
                    reps=(max(int(getattr(result, "rep_count", 0) or 0), app_reps)
                          if result.movement == "dynamic" else None),
                    hold=(yoga_hold if result.movement == "static" else None),
                )
                if profile.key in PHASE_TRACKS:
                    overlay.draw_phase_track(canvas, PHASE_TRACKS[profile.key], result.phase)
            else:
                overlay.draw_banner(canvas, profile.name, result.notes[0] if result.notes else "", fault=True)
            try:
                video_slot.image(canvas, channels="BGR", use_container_width=True)
            except Exception:
                pass
            if result.ready:
                colour = theme.BAD if primary else theme.GOOD
                stage = coaching.stage.name if primary else "ON TRACK"
                mod = coaching.modality.name.replace("_", " ").lower()
                _cue_html = theme.cue(f"{stage} · {mod}", coaching.message or "Good form", colour, alert=bool(primary) and coaching.speak)
                cue_slot.markdown(_cue_html, unsafe_allow_html=True)
                s.last_cue_html = _cue_html
                chips = []
                control = getattr(result, "control", None)
                if control is not None:
                    if control >= 75:
                        chips.append(("Controlled", theme.GOOD))
                    elif control >= 55:
                        chips.append(("Getting quick", theme.WARN))
                    else:
                        chips.append(("Slow it down", theme.BAD))
                if ghost is not None:
                    guide = alignment_status(frame.pose, ghost, canvas.shape)
                    if guide == "aligned":
                        chips.append(("On the guide", theme.GOOD))
                    elif guide == "close":
                        chips.append(("Near the guide", theme.WARN))
                if chips:
                    _strip_html = "".join(f"<span class='mw-chip' style='color:{c};border-color:{c}55'>● {t}</span>" for t, c in chips)
                    strip_slot.markdown(_strip_html, unsafe_allow_html=True)
                    s.last_strip_html = _strip_html
                else:
                    strip_slot.empty()
                    s.last_strip_html = None
                counter = (
                    f"{max(int(getattr(result, "rep_count", 0) or 0), app_reps)}"
                    if result.movement == "dynamic"
                    else f"{yoga_hold:.1f}s"
                )
                label = "Reps" if result.movement == "dynamic" else "Hold (5–7s)"
                _ring_html = theme.ring(result.score, counter, label)
                stat_slot.markdown(_ring_html, unsafe_allow_html=True)
                if result.movement == "static":
                    stat_slot.caption(f"Yoga hold target: 5–7 seconds · current stable hold: {yoga_hold:.1f}s")
                s.last_ring_html = _ring_html
                rows = "".join(theme.bar_row(m.label, m.display, m.score(), theme.score_color(m.score())) for m in result.metrics[:6])
                extra = theme.bar_row("Corrections fixed", f"{coaching.successes}/{coaching.attempts}", None, theme.LAVENDER) if coaching.attempts else ""
                _metric_html = theme.metrics_panel(rows + extra)
                metric_slot.markdown(_metric_html, unsafe_allow_html=True)
                s.last_metric_html = _metric_html
            if coaching.show_comfort_check or coaching.suggested_variation:
                break
    finally:
        if _loop_done:
            _release_live_src()

    coaching = s.get("last_coaching")
    if coaching and coaching.show_comfort_check:
        st.markdown("### This movement seems difficult to maintain.")
        st.markdown("How does this movement feel?")
        c = st.columns(3)
        if c[0].button("🟢 Comfortable", key="comfort_ok"):
            coach.answer_comfort("comfortable")
            s.comfort_variation = None
            st.rerun()
        if c[1].button("🟡 Challenging", key="comfort_challenging"):
            coach.answer_comfort("challenging")
            s.comfort_variation = None
            st.rerun()
        if c[2].button("🔴 Uncomfortable", key="comfort_uncomfortable"):
            coach.answer_comfort("uncomfortable")
            # Always provide a useful fallback even if the adaptive coach
            # does not return a variation for this particular exercise.
            s.comfort_variation = getattr(coaching, "suggested_variation", None) or easier_variation(s.exercise)
            st.rerun()
    elif getattr(s, "comfort_variation", None):
        var = s.comfort_variation
        vname = var.get("name", "Easier variation") if isinstance(var, dict) else str(var)
        vhint = (var.get("hint", "Reduce the range and use support.") if isinstance(var, dict) else "Reduce the range and use support.")
        st.markdown("### ❤️ Let's make this easier and more comfortable.")
        st.info(f"**{vname}** — {vhint}")
        st.caption("Do not push through pain. If discomfort continues, stop the movement.")
        c = st.columns(2)
        if c[0].button("▶️ Try easier version", type="primary", key="try_easier"):
            try:
                coach.accept_variation()
            except Exception:
                pass
            if s.recorder is not None:
                try:
                    s.recorder.note_variation(vname)
                except Exception:
                    pass
            s.comfort_variation = None
            reset_movement_tracking()
            st.rerun()
        if c[1].button("↩️ Return to original", key="return_original"):
            try:
                coach.reject_variation()
            except Exception:
                pass
            s.comfort_variation = None
            st.rerun()
    elif coaching and coaching.suggested_variation:
        var = coaching.suggested_variation
        vname = var.get("name", str(var)) if isinstance(var, dict) else str(var)
        vhint = (var.get("hint") if isinstance(var, dict) else None) or "This variation may be easier to perform."
        st.markdown("### Let's not force this movement.")
        st.info(f"**{vname}** — {vhint}")
        c = st.columns(2)
        if c[0].button("Try this variation", type="primary"):
            coach.accept_variation()
            s.recorder.note_variation(vname)
            st.rerun()
        if c[1].button("Keep the original"):
            coach.reject_variation(); st.rerun()
    else:
        c = st.columns(2)
        if c[0].button("Finish session", type="primary", key="finish_bottom"):
            finish_session()
        if c[1].button("Restart", key="restart_bottom"):
            _release_live_src()
            _clear_live_panel_cache()
            go(CALIBRATE)
    disclaimer()


# --------------------------------------------------------------------------
# SESSION SUMMARY
# --------------------------------------------------------------------------
elif s.screen == SUMMARY:
    header("Session summary", "", HOME)
    sm = s.summary
    if sm is None:
        st.info("No session recorded yet.")
    else:
        cols = st.columns(5)
        cols[0].markdown(theme.stat(f"{sm.movement_score:.0f}%", "Movement score", theme.score_color(sm.movement_score)), unsafe_allow_html=True)
        control = getattr(sm, "control_score", None)
        cols[1].markdown(theme.stat("--" if control is None else f"{control:.0f}%", "Movement control", theme.score_color(control)), unsafe_allow_html=True)
        if sm.reps:
            cols[2].markdown(theme.stat(str(sm.reps), "Repetitions"), unsafe_allow_html=True)
            cols[3].markdown(theme.stat(str(sm.good_reps), "Good reps", theme.GOOD), unsafe_allow_html=True)
        else:
            cols[2].markdown(theme.stat(f"{sm.hold_duration:.0f}s", "Hold"), unsafe_allow_html=True)
            cols[3].markdown(theme.stat(f"{sm.duration:.0f}s", "Duration"), unsafe_allow_html=True)
        cols[4].markdown(theme.stat(f"{sm.successful_corrections}/{sm.corrections}", "Corrections fixed"), unsafe_allow_html=True)
        st.write("")
        left, right = st.columns(2)
        with left:
            st.markdown(theme.eyebrow("Applicable metrics"), unsafe_allow_html=True)
            if sm.metric_scores:
                st.markdown(theme.metrics_panel("".join(theme.bar_row(name, f"{value:.0f}%", value, theme.score_color(value)) for name, value in sm.metric_scores.items())), unsafe_allow_html=True)
            else:
                st.caption("No metric stayed measurable long enough to score.")
            st.caption("Metrics that weren't applicable to your body map are excluded, not scored zero.")
        with right:
            st.markdown(theme.eyebrow("What stood out"), unsafe_allow_html=True)
            st.markdown(f"**Exercise** — {sm.exercise_name}")
            st.markdown(f"**Main issue** — {sm.main_issue or 'nothing persistent'}")
            st.markdown(f"**Most common error** — {sm.most_common_error or 'none'}")
            imp = sm.biggest_improvement
            st.markdown(f"**Biggest improvement** — {imp.label if imp else 'not enough data'}")
            unstable = int(getattr(sm, "unstable_events", 0) or 0)
            st.markdown(f"**Unstable or jerky moments** — {unstable if unstable else 'none detected'}")
            if getattr(sm, "control_improved", None) is True:
                st.markdown("Your movement became smoother during the final repetitions.")
            if sm.variation_used:
                st.markdown(f"**Variation used** — {sm.variation_used}")
            if sm.adaptive_mode:
                st.markdown("**Adaptive Mode** — active for this session")
        c = st.columns(2)
        if c[0].button("View progress", type="primary"):
            go(PROGRESS)
        if c[1].button("New session"):
            go(HOME)
    disclaimer()


# --------------------------------------------------------------------------
# PROGRESS
# --------------------------------------------------------------------------
elif s.screen == PROGRESS:
    header("My Progress", "Your sessions, trends and personalized movement profile.", HOME)
    rows = db.get_user_sessions(st.session_state["user_id"])

    # Profile is intentionally part of Progress — no logout button here.
    render_profile_card(show_logout=False)

    st.write("")
    st.markdown(theme.eyebrow("Progress overview"), unsafe_allow_html=True)
    if not rows:
        st.info("No sessions recorded yet. Complete your first session to start tracking progress.")
    else:
        scores = [r["movement_score"] for r in rows if r.get("movement_score") is not None]
        controls = [r["control_score"] for r in rows if r.get("control_score") is not None]
        avg_score = sum(scores) / len(scores) if scores else None
        avg_control = sum(controls) / len(controls) if controls else None
        cols = st.columns(4)
        cols[0].markdown(theme.stat(str(len(rows)), "Total Sessions"), unsafe_allow_html=True)
        cols[1].markdown(theme.stat("--" if avg_score is None else f"{avg_score:.0f}%", "Avg accuracy", theme.score_color(avg_score)), unsafe_allow_html=True)
        cols[2].markdown(theme.stat("--" if avg_control is None else f"{avg_control:.0f}%", "Avg control", theme.score_color(avg_control)), unsafe_allow_html=True)
        cols[3].markdown(theme.stat(rows[0].get("exercise_name", "—"), "Latest Activity", theme.GOOD), unsafe_allow_html=True)

        st.write("")
        st.markdown(theme.eyebrow("Previous sessions"), unsafe_allow_html=True)
        labels = [
            f"{item.get('date', '')} — {item.get('exercise_name', '?')} — "
            f"{'--' if item.get('movement_score') is None else f'{item.get('movement_score'):.0f}%'}"
            for item in rows
        ]
        choice = st.selectbox("Session", range(len(rows)), format_func=lambda i: labels[i], key="prog_session")
        picked = rows[choice]
        pleft, pright = st.columns(2)
        with pleft:
            metrics = picked.get("metric_scores") or {}
            if isinstance(metrics, dict) and metrics:
                prows = "".join(
                    theme.bar_row(name.replace("_", " ").capitalize(), f"{val:.0f}%", val, theme.score_color(val))
                    for name, val in metrics.items() if isinstance(val, (int, float))
                )
                st.markdown(theme.metrics_panel(prows), unsafe_allow_html=True)
            else:
                st.caption("No metric stayed measurable long enough to score.")
        with pright:
            psc = picked.get("movement_score")
            pct = picked.get("control_score")
            st.markdown(f"**Exercise** — {picked.get('exercise_name', '?')}")
            st.markdown("**Accuracy** — " + ("--" if psc is None else f"{psc:.0f}%"))
            st.markdown("**Movement control** — " + ("--" if pct is None else f"{pct:.0f}%"))
            if picked.get("reps"):
                st.markdown(f"**Reps** — {picked.get('reps')} ({picked.get('good_reps', 0)} clean)")
            if picked.get("main_issue"):
                st.markdown(f"**Main issue** — {picked['main_issue']}")

        st.write("")
        if st.button("Clear exercise history", key="clear_history"):
            db.clear_user_history(st.session_state["user_id"])
            st.rerun()
    disclaimer()


# --------------------------------------------------------------------------
# ACCOUNT
# --------------------------------------------------------------------------
elif s.screen == ACCOUNT:
    header("Account", "Manage your MoveWise profile and account settings.", HOME)
    render_profile_card(show_logout=True)
    st.markdown(
        """
        <div class="mw-glass" style="padding:18px 22px;margin-top:20px;">
            <div style="font-weight:700;">Your data stays connected to your account</div>
            <div style="opacity:.68;margin-top:5px;">
                Profile information and completed exercise sessions are stored in the local SQLite database.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    disclaimer()
