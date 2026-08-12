"""MoveWise — adaptive AI movement coach.

    streamlit run app.py

Screens: Home -> Library / Auto-detect -> Calibration -> Live analysis ->
Session summary -> Progress. Correction mode, comfort check and easier
variations appear inside Live analysis rather than as separate pages, so the
user's attention never leaves their movement.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

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

HOME, LIBRARY, DETECT, CALIBRATE, LIVE, SUMMARY, PROGRESS = (
    "home", "library", "detect", "calibrate", "live", "summary", "progress")

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
        s.live_src = None          # persisted FrameSource for the live screen
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
    """Shared input chooser. Kept identical everywhere it appears."""
    s.source_mode = st.radio("Input", ["Webcam", "Video file", "Sample clip"],
                             horizontal=True, key=f"src_{where}",
                             index=["Webcam", "Video file", "Sample clip"].index(s.source_mode))
    if s.source_mode == "Sample clip":
        match_clip_to_exercise()
        titles = {c.key: c.title for c in CLIPS}
        s.clip_key = st.selectbox("Clip", list(titles), format_func=lambda k: titles[k],
                                  key=f"clip_{where}", index=list(titles).index(s.clip_key))
        # Picking a clip for a different exercise switches the exercise too —
        # the two must never disagree about what's being analysed.
        picked = next((c for c in CLIPS if c.key == s.clip_key), None)
        if picked is not None and s.exercise is not None \
                and picked.exercise_key != s.exercise:
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
    """Keep the demo clip in step with the chosen exercise.

    Without this, selecting Warrior II still played whatever clip was last
    chosen (the default squat), so every exercise appeared to 'only go up and
    down'. Exercise choice drives the clip; picking a clip in the dropdown
    drives the exercise the other way (see source_picker).
    """
    if s.exercise is None or s.source_mode != "Sample clip":
        return
    current = next((c for c in CLIPS if c.key == s.clip_key), None)
    if current is not None and current.exercise_key == s.exercise:
        return
    fallback = next((c for c in CLIPS if c.exercise_key == s.exercise), None)
    if fallback is not None:
        s.clip_key = fallback.key
        # Drop the dropdowns' stored widget state, or Streamlit would restore
        # the stale selection on the next render and flip the exercise back.
        for widget_key in [k for k in st.session_state
                           if isinstance(k, str) and k.startswith("clip_")]:
            del st.session_state[widget_key]


def _clear_live_panel_cache() -> None:
    """Wipe the cached side-panel HTML so a new session starts clean."""
    for key in ("last_cue_html", "last_strip_html", "last_ring_html", "last_metric_html"):
        if key in st.session_state:
            del st.session_state[key]


def finish_session() -> None:
    """Stop analysing, save the session, and open the summary.

    Callable from the always-visible End Session button on the live screen —
    previously this logic only existed below the capture loop, which a webcam
    session never exits, so on camera it was unreachable.
    """
    if s.recorder is None:
        _clear_live_panel_cache()
        go(LIBRARY)
        return
    summary = s.recorder.finish()
    summary.corrections = s.coach.total_attempts
    summary.successful_corrections = s.coach.successes
    if getattr(summary, "control_score", None) is None:
        samples = getattr(s.recorder, "_control_samples", None) or []
        if samples:
            summary.control_score = float(sum(samples) / len(samples))
    if not getattr(summary, "unstable_events", 0):
        summary.unstable_events = int(getattr(s.recorder, "unstable_events", 0) or 0)
    s.store.save(summary)
    s.summary = summary
    s.recorder = None
    _clear_live_panel_cache()
    go(SUMMARY)


def build_source():
    if s.source_mode == "Webcam":
        return CameraSource()
    if s.source_mode == "Video file":
        return VideoSource(s.upload, loop=False, stride=1) if s.upload else None
    match_clip_to_exercise()
    clip = next(c for c in CLIPS if c.key == s.clip_key)
    return SyntheticSource(clip, loop=False)


def _release_live_src() -> None:
    """Release any FrameSource stored in session state."""
    src = getattr(s, "live_src", None)
    if src is not None:
        try:
            src.release()
        except Exception:
            pass
        s.live_src = None


# ==========================================================================
# HOME
# ==========================================================================
if s.screen == HOME:
    st.markdown(theme.wordmark(), unsafe_allow_html=True)
    st.write("")

    cols = st.columns(4)
    tiles = [
        ("◎", "Auto-detect", "Start moving and MoveWise works out which of the four "
         "exercises you're performing.", "Recognise", DETECT),
        ("▤", "Library", "Two yoga holds and two gym movements, each with its "
         "own analysis profile.", "Browse", LIBRARY),
        ("◐", "Adaptive Mode", "Calibrates to the body you have — a missing "
         "landmark is never an error.", "Calibrate", CALIBRATE),
        ("◭", "My Progress", "Scores, trends and your movement profile across "
         "sessions.", "Review", PROGRESS),
    ]
    # All four buttons share one style: a lone glowing CTA here read as an
    # inconsistency, not an invitation. Cream stays reserved for real
    # commit-actions inside a flow (Run calibration, Continue, Finish).
    for col, (ico, title, body, foot, dest) in zip(cols, tiles):
        with col:
            st.markdown(theme.tile(ico, title, body, foot), unsafe_allow_html=True)
            st.write("")
            if st.button(foot, key=f"home_{dest}"):
                go(dest)

    st.write("")
    n = len(s.store.all())
    if n:
        st.markdown(f"<p style='text-align:center;color:{theme.MUTED};font-size:.8rem;"
                    f"letter-spacing:.16em;text-transform:uppercase;margin-top:18px'>"
                    f"{n} session{'s' if n != 1 else ''} recorded on this machine</p>",
                    unsafe_allow_html=True)
    disclaimer()


# ==========================================================================
# LIBRARY
# ==========================================================================
elif s.screen == LIBRARY:
    header("Exercise Library", "Four movements, each with its own analysis profile.", HOME)
    for group, items in LIBRARY_GROUPS:
        st.markdown(theme.eyebrow(group), unsafe_allow_html=True)
        cols = st.columns(2)
        for col, (key, ico, desc) in zip(cols, items):
            prof = s.registry.get(key)
            with col:
                st.markdown(theme.tile(ico, prof.name, desc,
                                       prof.movement.value + " · " + prof.category.value),
                            unsafe_allow_html=True)
                st.write("")
                if st.button("Select", key=f"lib_{key}"):
                    s.exercise = key
                    go(CALIBRATE)
        st.write("")
    disclaimer()


# ==========================================================================
# AUTO-DETECT
# ==========================================================================
elif s.screen == DETECT:
    header("Auto-detect my exercise", "Get into position — recognition runs over a "
           "short window so one odd frame can't flip the answer.", HOME)
    source_picker("detect")
    st.info("Recognition covers Warrior II, Tree Pose, Squat and Bicep Curl. "
            "If confidence is low you'll be asked to choose manually.")

    c1, c2 = st.columns(2)
    if c1.button("Start detection", type="primary"):
        s.exercise = None
        go(CALIBRATE)
    if c2.button("Choose Exercise Manually"):
        go(LIBRARY)
    disclaimer()


# ==========================================================================
# CALIBRATION
# ==========================================================================
elif s.screen == CALIBRATE:
    name = s.registry.get(s.exercise).name if s.exercise else "Auto-detect"
    header("Calibration", f"{name} — building your Personal Body Map.", HOME)
    source_picker("cal")

    st.markdown("Stand so your whole body is in frame. This takes a few seconds and "
                "decides which measurements apply to **your** body.")

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
                    f = (s.engine.process(item.image) if src.needs_detection
                         else s.engine.process_pose(item.pose))
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
        st.markdown(f"<div class='mw-glass' style='padding:18px 20px'>{chips}</div>",
                    unsafe_allow_html=True)
        st.write("")

        if bm.mode is BodyMode.ADAPTIVE:
            st.warning("**Adaptive Mode** — some landmarks aren't trackable. "
                       "Those measurements are switched off rather than counted "
                       "against you. A landmark we can't see is never a posture error.")
        else:
            st.success("Standard mode — full landmark set available.")

        if st.button("Continue to live analysis", type="primary"):
            s.coach.reset()
            s.registry.reset_all()
            s.recognizer.reset()
            prof0 = s.registry.get(s.exercise) if s.exercise else None
            s.recorder = SessionRecorder(
                exercise=s.exercise or "",
                exercise_name=prof0.name if prof0 else "Auto-detect",
                adaptive_mode=(s.body_map is not None
                               and s.body_map.mode is BodyMode.ADAPTIVE))
            # Always open a fresh source for a new session, and wipe stale
            # dashboard cache from any previous session.
            _release_live_src()
            _clear_live_panel_cache()
            go(LIVE)
    disclaimer()


# ==========================================================================
# LIVE ANALYSIS
# ==========================================================================
elif s.screen == LIVE:
    # ------------------------------------------------------------------
    # On every Streamlit rerun we re-enter this branch.  The camera must
    # stay open across reruns (button clicks, ghost toggle, etc.) so we
    # store the FrameSource in session state and only open it once.
    # ------------------------------------------------------------------

    profile = s.registry.get(s.exercise) if s.exercise else None

    # Header lives in a placeholder so it can update the moment auto-detect
    # locks on — otherwise it reads "Identifying exercise…" over a screen that
    # is plainly already analysing a known exercise.
    head_slot = st.empty()

    def _set_header(name: str) -> None:
        head_slot.markdown(theme.page_head(name, "One correction at a time."),
                           unsafe_allow_html=True)

    _set_header(profile.name if profile else "Identifying exercise…")

    # ---- session controls -------------------------------------------------
    # Rendered BEFORE the capture loop so they stay clickable while it runs.
    bar = st.columns([0.9, 1, 1, 1, 1, 0.9, 1.2])
    if bar[0].button("← Back", key="live_back"):
        _release_live_src()
        go(LIBRARY)
    for i, ex_key in enumerate(("warrior_2", "tree_pose", "squat", "bicep_curl")):
        ex_prof = s.registry.get(ex_key)
        active = s.exercise == ex_key
        if bar[i + 1].button(ex_prof.name, key=f"live_switch_{ex_key}",
                             type="primary" if active else "secondary") and not active:
            s.exercise = ex_key
            ex_prof.reset()
            s.coach.reset()
            s.recorder = SessionRecorder(
                exercise=ex_key, exercise_name=ex_prof.name,
                adaptive_mode=(s.body_map is not None
                               and s.body_map.mode is BodyMode.ADAPTIVE))
            match_clip_to_exercise()
            # Release camera so it reopens cleanly with the new source config
            _release_live_src()
            st.rerun()
    if bar[5].button(("👻 On" if s.ghost else "👻 Off"), key="live_ghost",
                     help="Toggle the reference guide"):
        s.ghost = not s.ghost
        st.rerun()
    if bar[6].button("⏹ End Session", key="live_end"):
        _release_live_src()
        finish_session()

    top = st.columns([4, 2])
    video_slot = top[0].empty()

    # The side-panel slots are written inside the loop.  To prevent them
    # from going blank when a button triggers a rerun before the loop has
    # run even once, we seed them with their last known values stored in
    # session state, then overwrite from inside the loop as usual.
    side = top[1]
    cue_slot   = side.empty()
    strip_slot = side.empty()
    stat_slot  = side.empty()
    metric_slot = side.empty()
    action_slot = side.container()
    status_slot = st.empty()

    # Restore last-known side panel content so the dashboard doesn't vanish
    # on button-click reruns.
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

    # ---- open / reuse the persistent source ------------------------------
    src = getattr(s, "live_src", None)
    _src_ok = src is not None
    if not _src_ok:
        src = build_source()
        if src is None or not src.open():
            st.error("Could not open that input. Go back and pick **Sample clip** — "
                     "it needs no camera.")
            if st.button("← Back", key="live_err_back"):
                go(CALIBRATE)
            st.stop()
        s.live_src = src

    if s.body_map is not None:
        s.engine.force_body_map(s.body_map)
    if not src.needs_detection:
        s.engine.calibrator.duration = 0.5

    recorder: SessionRecorder = s.recorder or SessionRecorder(
        exercise=s.exercise or "",
        exercise_name=profile.name if profile else "Auto-detect")
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

            frame = (s.engine.process(item.image) if src.needs_detection
                     else s.engine.process_pose(item.pose))

            if profile is None:
                rec = s.recognizer.observe(frame)
                if rec.confident and rec.key:
                    s.exercise = rec.key
                    profile = s.registry.get(rec.key)
                    _set_header(profile.name)   # detection locked on
                else:
                    canvas = item.image.copy()
                    overlay.draw_skeleton(canvas, frame.pose)
                    overlay.draw_banner(canvas, "Identifying exercise", rec.message)
                    video_slot.image(canvas, channels="BGR", use_container_width=True)
                    continue

            result = profile.analyse(frame)
            coaching = coach.update(result, frame.timestamp)
            recorder.update(result, frame.timestamp)
            s.last_coaching = coaching

            # ---- draw ----------------------------------------------------
            canvas = item.image.copy()
            primary = result.primary_error
            ghost = None
            emphasise = coaching.modality is Modality.GHOST_EMPHASIS

            if s.ghost and result.ready:
                ghost = fit_reference(profile.key, frame.pose, canvas.shape,
                                      progress=result.reference_progress,
                                      mirror=getattr(profile, "front_side", None) == "right")
                overlay.draw_ghost(canvas, ghost, emphasis=emphasise)

            overlay.draw_skeleton(canvas, frame.pose,
                                  highlight=primary.landmarks if primary else ())

            if primary and ghost and coaching.modality in (Modality.ARROW,
                                                           Modality.GHOST_EMPHASIS):
                for lm in primary.landmarks[:1]:
                    overlay.draw_arrow(canvas,
                                       correction_arrow(frame.pose, ghost, lm, canvas.shape,
                                                        config.GHOST_ARROW_MIN_PIXELS),
                                       primary.cue)

            if frame.calibrating:
                overlay.draw_calibration(canvas, frame.calibration_progress)
            elif result.ready:
                overlay.draw_banner(canvas, coaching.message or "Good form",
                                    f"{profile.name} · {result.phase}",
                                    fault=primary is not None)
                overlay.draw_score_badge(
                    canvas, result.score, result.phase,
                    reps=result.rep_count if result.movement == "dynamic" else None,
                    hold=result.hold_duration if result.movement == "static" else None)
                if profile.key in PHASE_TRACKS:
                    overlay.draw_phase_track(canvas, PHASE_TRACKS[profile.key], result.phase)
            else:
                overlay.draw_banner(canvas, profile.name,
                                    result.notes[0] if result.notes else "", fault=True)

            try:
                video_slot.image(canvas, channels="BGR", use_container_width=True)
            except Exception:
                # One undisplayable frame is invisible if skipped and fatal if
                # raised — the stream continues either way.
                pass

            # ---- side panel ----------------------------------------------
            if result.ready:
                colour = theme.BAD if primary else theme.GOOD
                stage = (coaching.stage.name if primary else "ON TRACK")
                mod = coaching.modality.name.replace("_", " ").lower()
                _cue_html = theme.cue(f"{stage} · {mod}", coaching.message or "Good form",
                                      colour, alert=bool(primary) and coaching.speak)
                cue_slot.markdown(_cue_html, unsafe_allow_html=True)
                s.last_cue_html = _cue_html

                # Subtle status strip: movement-control state + ghost alignment.
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
                    _strip_html = "".join(
                        f"<span class='mw-chip' style='color:{c};"
                        f"border-color:{c}55'>● {t}</span>"
                        for t, c in chips)
                    strip_slot.markdown(_strip_html, unsafe_allow_html=True)
                    s.last_strip_html = _strip_html
                else:
                    strip_slot.empty()
                    s.last_strip_html = None

                counter = (f"{result.rep_count}" if result.movement == "dynamic"
                           else f"{result.hold_duration:.0f}s")
                label = "Reps" if result.movement == "dynamic" else "Hold"
                _ring_html = theme.ring(result.score, counter, label)
                stat_slot.markdown(_ring_html, unsafe_allow_html=True)
                s.last_ring_html = _ring_html

                rows = "".join(
                    theme.bar_row(m.label, m.display, m.score(),
                                  theme.score_color(m.score()))
                    for m in result.metrics[:6])
                extra = theme.bar_row("Corrections fixed",
                                      f"{coaching.successes}/{coaching.attempts}",
                                      None, theme.LAVENDER) if coaching.attempts else ""
                _metric_html = theme.metrics_panel(rows + extra)
                metric_slot.markdown(_metric_html, unsafe_allow_html=True)
                s.last_metric_html = _metric_html

            # ---- comfort check / variation -------------------------------
            if coaching.show_comfort_check or coaching.suggested_variation:
                break

        # loop exited: clip ended or user interaction needed
    finally:
        # Only release the source when the clip/video has genuinely finished.
        # For a live webcam or a still-running stream, keep it open so the
        # next rerun (button click) can resume without re-opening the camera.
        if _loop_done:
            _release_live_src()

    coaching = s.get("last_coaching")
    if coaching and coaching.show_comfort_check:
        st.markdown("### This movement seems difficult to maintain.")
        st.markdown("How does this movement feel?")
        c = st.columns(3)
        if c[0].button("🟢 Comfortable"):
            coach.answer_comfort("comfortable"); st.rerun()
        if c[1].button("🟡 Challenging"):
            coach.answer_comfort("challenging"); st.rerun()
        if c[2].button("🔴 Uncomfortable"):
            coach.answer_comfort("uncomfortable"); st.rerun()

    elif coaching and coaching.suggested_variation:
        var = coaching.suggested_variation
        vname = var.get("name", str(var)) if isinstance(var, dict) else str(var)
        vhint = (var.get("hint") if isinstance(var, dict) else None) or             "This variation may be easier to perform."
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


# ==========================================================================
# SESSION SUMMARY
# ==========================================================================
elif s.screen == SUMMARY:
    header("Session summary", "", HOME)
    sm = s.summary
    if sm is None:
        st.info("No session recorded yet.")
    else:
        cols = st.columns(5)
        cols[0].markdown(theme.stat(f"{sm.movement_score:.0f}%", "Movement score",
                                    theme.score_color(sm.movement_score)),
                         unsafe_allow_html=True)
        control = getattr(sm, "control_score", None)
        cols[1].markdown(theme.stat("--" if control is None else f"{control:.0f}%",
                                    "Movement control", theme.score_color(control)),
                         unsafe_allow_html=True)
        if sm.reps:
            cols[2].markdown(theme.stat(str(sm.reps), "Repetitions"), unsafe_allow_html=True)
            cols[3].markdown(theme.stat(str(sm.good_reps), "Good reps",
                                        theme.GOOD), unsafe_allow_html=True)
        else:
            cols[2].markdown(theme.stat(f"{sm.hold_duration:.0f}s", "Hold"),
                             unsafe_allow_html=True)
            cols[3].markdown(theme.stat(f"{sm.duration:.0f}s", "Duration"),
                             unsafe_allow_html=True)
        cols[4].markdown(theme.stat(f"{sm.successful_corrections}/{sm.corrections}",
                                    "Corrections fixed"), unsafe_allow_html=True)

        st.write("")
        left, right = st.columns(2)
        with left:
            st.markdown(theme.eyebrow("Applicable metrics"), unsafe_allow_html=True)
            if sm.metric_scores:
                st.markdown(theme.metrics_panel("".join(
                    theme.bar_row(name, f"{value:.0f}%", value,
                                  theme.score_color(value))
                    for name, value in sm.metric_scores.items())),
                    unsafe_allow_html=True)
            else:
                st.caption("No metric stayed measurable long enough to score.")
            st.caption("Metrics that weren't applicable to your body map are excluded, "
                       "not scored zero.")
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


# ==========================================================================
# PROGRESS
# ==========================================================================
elif s.screen == PROGRESS:
    header("My Progress", "Stored locally on this machine. No account, no cloud.", HOME)
    rows = s.store.all()
    if not rows:
        st.info("No sessions recorded yet. Complete your first session to start "
                "tracking progress.")
    else:
        # ---- 1. weekly stat row -------------------------------------------
        wk = s.store.weekly_summary()
        cols = st.columns(4)
        cols[0].markdown(theme.stat(str(wk["sessions"]), "Sessions this week"),
                         unsafe_allow_html=True)
        avg_score = wk.get("avg_score")
        cols[1].markdown(theme.stat("--" if avg_score is None else f"{avg_score:.0f}%",
                                    "Avg accuracy", theme.score_color(avg_score)),
                         unsafe_allow_html=True)
        avg_control = wk.get("avg_control")
        cols[2].markdown(theme.stat("--" if avg_control is None else f"{avg_control:.0f}%",
                                    "Avg control", theme.score_color(avg_control)),
                         unsafe_allow_html=True)
        cols[3].markdown(theme.stat(wk.get("most_improved") or "--", "Most improved",
                                    theme.GOOD if wk.get("most_improved") else theme.MUTED),
                         unsafe_allow_html=True)

        # ---- 2. seven day strip -------------------------------------------
        st.write("")
        st.markdown(theme.eyebrow("7-day progress"), unsafe_allow_html=True)
        if len(rows) < 2:
            st.caption("Complete more sessions to see your 7-day progress.")
        day_cols = st.columns(7)
        for col, day in zip(day_cols, s.store.last_7_days()):
            score = day["score"]
            colour = theme.score_color(score)
            value = "—" if score is None else f"{score:.0f}"
            tip = (f"{day['date']} — no session" if not day["sessions"] else
                   f"{day['date']} — {day['sessions']} session"
                   + ("s" if day["sessions"] != 1 else ""))
            dots = "●" * min(int(day["sessions"]), 4)
            col.markdown(
                f"<div class='mw-glass mw-stat' title='{tip}'>"
                f"<div class='k'>{day['label']}</div>"
                f"<div class='v' style='color:{colour};font-size:1.7rem'>{value}</div>"
                f"<div class='k' style='color:{theme.LAVENDER}'>{dots}&#8203;</div></div>",
                unsafe_allow_html=True)

        # ---- 3. previous sessions -----------------------------------------
        st.write("")
        st.markdown(theme.eyebrow("Previous sessions"), unsafe_allow_html=True)
        listed = s.store.sessions_list()
        labels = ["{d} — {n} — {s}".format(
            d=item["date"], n=item["exercise_name"],
            s="--" if item["score"] is None else f"{item['score']:.0f}%")
            for item in listed]
        choice = st.selectbox("Session", range(len(listed)),
                              format_func=lambda i: labels[i], key="prog_session")
        picked = s.store.get_session(listed[choice]["index"]) if listed else None
        if picked:
            pleft, pright = st.columns(2)
            with pleft:
                metrics = picked.get("metric_scores") or {}
                if isinstance(metrics, dict) and metrics:
                    prows = "".join(
                        theme.bar_row(name.replace("_", " ").capitalize(), f"{val:.0f}%",
                                      val, theme.score_color(val))
                        for name, val in metrics.items()
                        if isinstance(val, (int, float)))
                    st.markdown(theme.metrics_panel(prows), unsafe_allow_html=True)
                else:
                    st.caption("No metric stayed measurable long enough to score.")
            with pright:
                psc = picked.get("movement_score")
                pct = picked.get("control_score")
                st.markdown(f"**Exercise** — {picked.get('exercise_name', '?')}")
                st.markdown("**Accuracy** — "
                            + ("--" if psc is None else f"{psc:.0f}%"))
                st.markdown("**Movement control** — "
                            + ("--" if pct is None else f"{pct:.0f}%"))
                if picked.get("reps"):
                    st.markdown(f"**Reps** — {picked.get('reps')} "
                                f"({picked.get('good_reps', 0)} clean)")
                if picked.get("main_issue"):
                    st.markdown(f"**Main issue** — {picked['main_issue']}")
                unstable = picked.get("unstable_events")
                if isinstance(unstable, (int, float)) and unstable > 0:
                    st.markdown(f"**Unstable or jerky moments** — {int(unstable)}")

        # ---- 4. per-exercise progress --------------------------------------
        st.write("")
        st.markdown(theme.eyebrow("Exercise progress"), unsafe_allow_html=True)
        for key, info in s.store.exercise_progress().items():
            avg, ctl, best = info["avg_score"], info["avg_control"], info["best_score"]
            erows = theme.bar_row("Avg accuracy",
                                  "--" if avg is None else f"{avg:.0f}%",
                                  avg, theme.score_color(avg))
            erows += theme.bar_row("Avg control",
                                   "--" if ctl is None else f"{ctl:.0f}%",
                                   ctl, theme.score_color(ctl))
            erows += theme.bar_row("Best score",
                                   "--" if best is None else f"{best:.0f}%",
                                   best, theme.score_color(best))
            n = info["sessions"]
            st.markdown(f"**{info['name']}** · {n} session{'s' if n != 1 else ''}")
            st.markdown(theme.metrics_panel(erows), unsafe_allow_html=True)

        # ---- 5. improvement -------------------------------------------------
        st.write("")
        st.markdown(theme.eyebrow("Improvement"), unsafe_allow_html=True)
        imp = s.store.improvement()
        if imp is None:
            st.caption("Complete at least two sessions to see your performance "
                       "improvement.")
        else:
            def _imp_line(label, pair):
                if not pair:
                    return
                a, b = pair
                delta = b - a
                sign = "+" if delta >= 0 else ""
                st.markdown(f"**{label}**: {a:.0f}% → {b:.0f}% ({sign}{delta:.0f}%)")
            _imp_line("Accuracy", imp.get("accuracy"))
            _imp_line("Control", imp.get("control"))
            if not imp.get("accuracy") and not imp.get("control"):
                st.caption("Not enough measured data yet for a performance "
                           "improvement comparison.")
            else:
                st.caption("Performance improvement, earliest session to latest.")

        # ---- 6. weekly note --------------------------------------------------
        st.write("")
        st.markdown(theme.eyebrow("Weekly summary"), unsafe_allow_html=True)
        st.markdown(wk["note"])
        if wk.get("focus"):
            st.markdown(f"**Recommended focus** — {wk['focus']}")

        st.write("")
        if st.button("Clear history"):
            s.store.clear()
            st.rerun()
    disclaimer()
