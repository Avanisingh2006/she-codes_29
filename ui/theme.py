"""MoveWise design system — "Aurora Glass".

The brief for this app is unusual: the camera feed is the hero, and the chrome
around it has to be striking without ever competing with the user's own body on
screen. That points at deep, saturated ground with light living *inside* panels
rather than on them — colour that glows out of glass instead of blocks of flat
fill sitting next to video.

  Ground    deep indigo-violet (#12102A), never pure black — pure black kills the
            sense of depth the glass panels rely on, and smears on OLED.
  Panels    translucent, blurred, hairline-lit. They read as lenses over the
            aurora rather than cards stuck on top of it.
  Accent    cream (#FFF4BF) is reserved for the one thing you should touch or
            read next. Nothing else in the system is allowed to use it, which is
            what makes it work as a call to action.
  Semantics mint / amber / coral for form quality — deliberately a *separate*
            scale from the brand ramp, so "this is a button" and "your knee is
            wrong" can never be confused.

Type is condensed-athletic for display (Big Shoulders Display) against a humanist
sans for body (Plus Jakarta Sans). The condensed face does the shouting; the body
face stays quiet and legible at the sizes real information lives at.
"""
from __future__ import annotations

# ---------------------------------------------------------------- palette --
GROUND      = "#12102A"
GROUND_2    = "#191541"
INK         = "#F5F2FF"
INK_2       = "#D6CFF2"
MUTED       = "#A79CD3"

INDIGO      = "#403D88"
PURPLE      = "#8C56D4"
LAVENDER    = "#DC95FF"
PINK        = "#FFBEFB"
CREAM       = "#FFF4BF"

MAUVE       = "#AF719D"
PLUM        = "#8B639B"
BLUSH       = "#F8B2B2"

GOOD        = "#7DF0C0"
WARN        = "#FFD98C"
BAD         = "#FF8FA3"

ACCENT      = CREAM          # the single call-to-action colour
GHOST       = LAVENDER

GLASS       = "rgba(255,255,255,.045)"
GLASS_HI    = "rgba(255,255,255,.10)"
GLASS_LINE  = "rgba(255,255,255,.09)"


def score_color(score) -> str:
    """Form quality → semantic colour. Never returns a brand colour."""
    if score is None:
        return MUTED
    return GOOD if score >= 80 else (WARN if score >= 55 else BAD)


DISCLAIMER = ("For fitness and movement guidance only. MoveWise does not diagnose "
              "medical conditions or replace a qualified physiotherapist or doctor.")


# ------------------------------------------------------------------- CSS --
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Big+Shoulders+Display:wght@500;700;800;900&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {{
  --ground:{GROUND}; --ground2:{GROUND_2};
  --ink:{INK}; --ink2:{INK_2}; --muted:{MUTED};
  --indigo:{INDIGO}; --purple:{PURPLE}; --lav:{LAVENDER}; --pink:{PINK}; --cream:{CREAM};
  --good:{GOOD}; --warn:{WARN}; --bad:{BAD};
  --glass:{GLASS}; --glassline:{GLASS_LINE};
  --display:'Big Shoulders Display','Oswald','Arial Narrow',sans-serif;
  --body:'Plus Jakarta Sans',-apple-system,'Segoe UI',sans-serif;
  --r:18px;
}}

/* Streamlit injects its own font-family after this block, so the type rules
   here have to be marked important or they silently lose the cascade. */
html, body, .stApp, .stApp p, .stApp span, .stApp div,
.stApp label, .stApp li {{ font-family: var(--body) !important; }}

/* ---- ground + aurora ------------------------------------------------- */
.stApp {{
  background: {GROUND};
  font-family:var(--body) !important;
}}
/* Three slow-drifting light sources. Fixed, behind everything, never
   interactive — this is the only motion in the resting UI. */
.stApp::before {{
  content:''; position:fixed; inset:-25%; z-index:0; pointer-events:none;
  background:
    radial-gradient(38% 42% at 18% 22%, {PURPLE}55 0%, transparent 62%),
    radial-gradient(34% 38% at 84% 16%, {LAVENDER}3d 0%, transparent 64%),
    radial-gradient(44% 46% at 62% 88%, {INDIGO}66 0%, transparent 66%);
  filter: blur(38px);
  animation: drift 26s ease-in-out infinite alternate;
}}
/* Fine grain stops the big gradients from banding on cheap panels. */
.stApp::after {{
  content:''; position:fixed; inset:0; z-index:0; pointer-events:none; opacity:.16;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='120'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='3'/%3E%3C/filter%3E%3Crect width='120' height='120' filter='url(%23n)' opacity='.5'/%3E%3C/svg%3E");
}}
@keyframes drift {{
  0%   {{ transform: translate3d(0,0,0) scale(1); }}
  50%  {{ transform: translate3d(2.5%,-2%,0) scale(1.06); }}
  100% {{ transform: translate3d(-2%,2.5%,0) scale(1.02); }}
}}
@media (prefers-reduced-motion: reduce) {{
  .stApp::before {{ animation:none; }}
  * {{ animation-duration:.001s !important; transition-duration:.001s !important; }}
}}

html {{ font-size:17px; }}
.block-container {{ position:relative; z-index:1; max-width:1280px; padding-top:1.6rem; }}
section[data-testid="stSidebar"] {{ display:none; }}
h1 a, h2 a, h3 a, h4 a,
span[data-testid="stHeaderActionElements"] {{ display:none !important; }}
#MainMenu, footer, header[data-testid="stHeader"] {{ visibility:hidden; height:0; }}

h1,h2,h3,h4,p,span,div,label,li {{ color:{INK}; }}

/* ---- display type ---------------------------------------------------- */
.mw-wordmark {{
  font-family:var(--display) !important; font-weight:900 !important; font-size:clamp(4.2rem,13vw,9.5rem) !important;
  line-height:.84 !important; letter-spacing:-.015em; text-transform:uppercase; margin:0;
  background:linear-gradient(103deg,{CREAM} 4%,{PINK} 34%,{LAVENDER} 62%,{PURPLE} 96%);
  -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
  filter:drop-shadow(0 6px 40px {PURPLE}55);
}}
.mw-kicker {{
  font-family:var(--display) !important; font-weight:700; font-size:.82rem; letter-spacing:.34em;
  text-transform:uppercase; color:{LAVENDER}; margin-bottom:.7rem;
}}
.mw-tagline {{ font-size:1.22rem; color:{INK}; margin:.9rem 0 .2rem; font-weight:600; }}
.mw-philosophy {{ font-size:.98rem; color:{INK_2}; }}
.mw-philosophy b {{ color:{PINK}; font-weight:600; }}

.mw-h {{
  font-family:var(--display) !important; font-weight:800 !important; font-size:clamp(2.1rem,4.6vw,3.2rem) !important;
  line-height:.94; letter-spacing:-.01em; text-transform:uppercase; margin:0;
}}
.mw-sub {{ color:{INK_2}; font-size:1rem; margin:.4rem 0 0; }}

/* ---- glass ----------------------------------------------------------- */
.mw-glass {{
  background:linear-gradient(158deg, rgba(255,255,255,.075), rgba(255,255,255,.022));
  border:1px solid var(--glassline); border-radius:var(--r);
  backdrop-filter:blur(22px) saturate(150%); -webkit-backdrop-filter:blur(22px) saturate(150%);
  box-shadow:0 18px 44px rgba(0,0,0,.42), inset 0 1px 0 rgba(255,255,255,.12);
}}

/* Nav tile — a body, then a divided footer. The border itself is a gradient
   (padding-box/border-box composite) and a light bar sweeps the top edge on
   hover: the card lights up from within instead of merely floating. */
.mw-tile {{
  position:relative; overflow:hidden; padding:24px 24px 0;
  height:292px; display:flex; flex-direction:column;
  border:1px solid transparent;
  background:
    linear-gradient(158deg, rgba(30,25,64,.92), rgba(21,17,48,.88)) padding-box,
    linear-gradient(150deg, {LAVENDER}52, rgba(255,255,255,.07) 38%, {PURPLE}3d 90%) border-box;
  transition:box-shadow .35s ease, background .35s ease, transform .35s cubic-bezier(.2,.8,.3,1);
}}
.mw-tile::before {{
  content:''; position:absolute; top:0; left:14%; right:14%; height:2px;
  border-radius:99px; opacity:.35;
  background:linear-gradient(90deg, transparent, {LAVENDER}, transparent);
  transition:opacity .35s ease, left .35s ease, right .35s ease;
}}
.mw-tile::after {{
  content:''; position:absolute; inset:0; pointer-events:none; opacity:0;
  background:radial-gradient(70% 55% at 50% -10%, {LAVENDER}2e, transparent 70%);
  transition:opacity .35s ease;
}}
.mw-tile:hover {{
  transform:translateY(-3px);
  background:
    linear-gradient(158deg, rgba(36,30,76,.94), rgba(24,20,56,.9)) padding-box,
    linear-gradient(150deg, {CREAM}66, {PINK}44 38%, {LAVENDER}66 90%) border-box;
  box-shadow:0 30px 70px rgba(0,0,0,.55), 0 0 44px {PURPLE}2e;
}}
.mw-tile:hover::before {{ opacity:1; left:4%; right:4%; }}
.mw-tile:hover::after {{ opacity:1; }}
.mw-tile .glyph {{
  width:48px;height:48px;border-radius:14px;display:grid;place-items:center;
  font-size:1.4rem;margin-bottom:14px; color:{PINK};
  background:linear-gradient(150deg,{PURPLE}70,{LAVENDER}26);
  border:1px solid rgba(255,255,255,.16);
  box-shadow:inset 0 1px 0 rgba(255,255,255,.2), 0 0 18px {PURPLE}33;
  transition:box-shadow .35s ease, color .35s ease;
}}
.mw-tile:hover .glyph {{ color:{CREAM}; box-shadow:inset 0 1px 0 rgba(255,255,255,.25), 0 0 26px {LAVENDER}66; }}
.mw-tile h4 {{
  font-family:var(--display) !important; font-weight:800; font-size:1.42rem;
  letter-spacing:.005em; text-transform:uppercase; margin:0 0 6px;
  /* Reserve two lines whether or not the title needs them, so a longer name
     never pushes its tile — and the button under it — out of alignment. */
  line-height:1.06; min-height:2.2em;
}}
.mw-tile p {{
  margin:0 0 18px; color:{INK_2}; font-size:.92rem; line-height:1.6;
  /* Fixed box, not a clamp: Streamlit's markdown pipeline drops the -webkit
     box properties, so the clamp silently never applied and long copy grew
     the card. A hard height cannot be ignored. */
  height:4.85em; overflow:hidden;
}}
.mw-tile .foot {{
  margin-top:auto; border-top:1px solid var(--glassline);
  margin-left:-24px; margin-right:-24px; padding:12px 24px;
  font-family:var(--display) !important;
  font-size:.8rem; letter-spacing:.22em; text-transform:uppercase; color:{LAVENDER};
}}

/* ---- stat ------------------------------------------------------------ */
.mw-stat {{ padding:16px 12px; text-align:center; }}
.mw-stat .v {{
  font-family:var(--display) !important; font-weight:800; font-size:2.5rem; line-height:1;
  letter-spacing:-.01em; font-variant-numeric:tabular-nums;
}}
.mw-stat .k {{
  font-size:.63rem; text-transform:uppercase; letter-spacing:.19em;
  color:{MUTED}; margin-top:7px; font-weight:600;
}}

/* ---- score ring ------------------------------------------------------ */
.mw-ring {{ display:flex; align-items:center; gap:18px; padding:18px 20px; }}
.mw-ring .dial {{
  --p:0; --c:{GOOD};
  width:96px;height:96px;border-radius:50%;flex:none;display:grid;place-items:center;
  background:conic-gradient(var(--c) calc(var(--p)*1%), rgba(255,255,255,.07) 0);
  filter:drop-shadow(0 0 16px color-mix(in srgb, var(--c) 55%, transparent));
}}
.mw-ring .dial > span {{
  width:76px;height:76px;border-radius:50%;background:{GROUND};
  display:grid;place-items:center;font-family:var(--display) !important;font-weight:800;
  font-size:2rem;font-variant-numeric:tabular-nums;
}}
.mw-ring .meta .lbl {{
  font-size:.63rem;letter-spacing:.2em;text-transform:uppercase;color:{MUTED};font-weight:600;
}}
.mw-ring .meta .big {{
  font-family:var(--display) !important;font-weight:800;font-size:1.8rem;line-height:1.05;
}}

/* ---- cue ------------------------------------------------------------- */
.mw-cue {{ padding:17px 20px 17px 22px; position:relative; overflow:hidden; }}
.mw-cue::before {{ content:''; position:absolute; left:0; top:0; bottom:0; width:4px; background:var(--cue,{GOOD}); }}
.mw-cue .stage {{
  font-size:.63rem;letter-spacing:.19em;text-transform:uppercase;color:{MUTED};font-weight:700;
}}
.mw-cue .msg {{
  font-family:var(--display) !important;font-weight:700;font-size:1.5rem;line-height:1.12;
  margin-top:5px;letter-spacing:.004em;
}}
/* A live cue pulses once so a *changed* instruction is noticed without motion
   running constantly in the user's peripheral vision. */
.mw-cue.alert {{ animation:cuepulse 1.5s ease-out 1; }}
@keyframes cuepulse {{
  0%   {{ box-shadow:0 0 0 0 {BAD}55, 0 18px 44px rgba(0,0,0,.42); }}
  70%  {{ box-shadow:0 0 0 16px transparent, 0 18px 44px rgba(0,0,0,.42); }}
  100% {{ box-shadow:0 0 0 0 transparent, 0 18px 44px rgba(0,0,0,.42); }}
}}

/* ---- metric rows ----------------------------------------------------- */
.mw-metrics {{ padding:14px 18px; }}
.mw-row {{ display:flex; align-items:center; gap:12px; padding:7px 0; font-size:.83rem; }}
.mw-row + .mw-row {{ border-top:1px solid rgba(255,255,255,.05); }}
.mw-row .n {{ min-width:112px; color:{INK_2}; font-weight:500; }}
.mw-row .bar {{ flex:1; height:7px; border-radius:99px; background:rgba(255,255,255,.07); overflow:hidden; }}
.mw-row .bar > i {{ display:block; height:100%; border-radius:99px; transition:width .35s cubic-bezier(.2,.8,.3,1); }}
.mw-row .v {{ min-width:78px; text-align:right; font-variant-numeric:tabular-nums; color:{MUTED}; }}

/* ---- chips ----------------------------------------------------------- */
.mw-chip {{
  display:inline-flex; align-items:center; gap:8px; padding:8px 15px; border-radius:99px;
  font-size:.78rem; font-weight:600; margin:0 7px 8px 0;
  background:var(--glass); border:1px solid var(--glassline);
}}
.mw-chip.on {{ border-color:{GOOD}55; color:{GOOD}; background:{GOOD}12; }}
.mw-chip.off {{ color:{MUTED}; }}

.mw-eyebrow {{
  font-family:var(--display) !important; font-weight:700; font-size:.78rem; letter-spacing:.3em;
  text-transform:uppercase; color:{LAVENDER}; margin:6px 0 12px;
}}

.mw-note {{
  color:{MUTED}; font-size:.78rem; line-height:1.6; border-top:1px solid rgba(255,255,255,.07);
  padding-top:14px; margin-top:30px;
}}

/* ---- buttons: floating pills, one cream CTA -------------------------- */
div.stButton > button {{
  font-family:var(--body) !important; font-weight:700; font-size:.92rem; letter-spacing:.01em;
  background:linear-gradient(158deg, rgba(255,255,255,.09), rgba(255,255,255,.03));
  color:{INK}; border:1px solid rgba(255,255,255,.16);
  border-radius:99px; padding:.72rem 1.35rem; width:100%; min-height:2.9rem;
  backdrop-filter:blur(16px); -webkit-backdrop-filter:blur(16px);
  box-shadow:0 6px 18px rgba(0,0,0,.3), inset 0 1px 0 rgba(255,255,255,.1);
  transition:transform .18s cubic-bezier(.2,.8,.3,1), box-shadow .22s ease,
             border-color .22s ease, background .22s ease, color .22s ease;
}}
div.stButton > button:hover {{
  transform:translateY(-2px); border-color:{LAVENDER}99; color:{CREAM};
  background:linear-gradient(158deg, rgba(220,149,255,.16), rgba(140,86,212,.08));
  box-shadow:0 12px 30px rgba(0,0,0,.45), 0 0 24px {LAVENDER}40,
             inset 0 1px 0 rgba(255,255,255,.16);
}}
div.stButton > button:active {{ transform:translateY(0) scale(.985); }}
div.stButton > button:focus-visible {{ outline:2px solid {CREAM}; outline-offset:3px; }}

/* The one cream call-to-action: solid, dark-inked, glowing. No sheen sweep —
   an animation running over the label is why it was unreadable before. */
div.stButton > button[kind="primary"] {{
  background:linear-gradient(100deg,{CREAM} 8%,{PINK} 92%);
  color:#241640 !important; border:none; font-weight:800; letter-spacing:.02em;
  box-shadow:0 12px 34px {PINK}55, 0 0 24px {CREAM}33,
             inset 0 1px 0 rgba(255,255,255,.5), inset 0 -2px 6px {PURPLE}22;
}}
div.stButton > button[kind="primary"] p {{ color:#241640 !important; font-weight:800; }}
div.stButton > button[kind="primary"]:hover {{
  transform:translateY(-2px) scale(1.015);
  box-shadow:0 18px 46px {PINK}77, 0 0 36px {CREAM}55,
             inset 0 1px 0 rgba(255,255,255,.6);
}}

/* ---- streamlit control skins ----------------------------------------- */
div[data-testid="stRadio"] label, div[data-testid="stSelectbox"] label,
div[data-testid="stFileUploader"] label {{
  color:{MUTED} !important; font-size:.68rem !important; letter-spacing:.18em;
  text-transform:uppercase; font-weight:700;
}}
div[data-baseweb="select"] > div {{
  background:var(--glass) !important; border:1px solid var(--glassline) !important;
  border-radius:14px !important;
}}
div[data-testid="stRadio"] > div {{ gap:.5rem; }}
div[data-testid="stFileUploaderDropzone"] {{
  background:var(--glass); border:1px dashed rgba(255,255,255,.16); border-radius:16px;
}}
div[data-testid="stProgress"] > div > div > div {{
  background:linear-gradient(90deg,{PURPLE},{LAVENDER},{PINK});
}}
div[data-testid="stAlert"] {{
  background:var(--glass); border:1px solid var(--glassline);
  border-radius:14px; backdrop-filter:blur(18px);
}}
div[data-testid="stImage"] img {{
  border-radius:20px; box-shadow:0 26px 66px rgba(0,0,0,.55), 0 0 0 1px rgba(255,255,255,.08);
}}
div.stButton > button p {{ color:inherit !important; }}
div[data-testid="stCaptionContainer"], div[data-testid="stCaptionContainer"] p
  {{ color:{MUTED} !important; font-size:.82rem !important; }}
hr {{ border-color:rgba(255,255,255,.07); }}
</style>
"""


# ------------------------------------------------------------ components --
def wordmark() -> str:
    return (f"<div style='text-align:center;padding:18px 0 4px'>"
            f"<div class='mw-kicker'>Adaptive AI movement coach</div>"
            f"<h1 class='mw-wordmark'>MoveWise</h1>"
            f"<p class='mw-tagline'>Smarter guidance for every movement.</p>"
            f"<p class='mw-philosophy'>Don't adapt your body to the AI. "
            f"<b>Let the AI adapt to you.</b></p></div>")


def page_head(title: str, sub: str = "") -> str:
    s = f"<p class='mw-sub'>{sub}</p>" if sub else ""
    return f"<div style='margin-bottom:18px'><h2 class='mw-h'>{title}</h2>{s}</div>"


def tile(glyph: str, title: str, body: str, foot: str) -> str:
    return (f"<div class='mw-glass mw-tile'><div class='glyph'>{glyph}</div>"
            f"<h4>{title}</h4><p>{body}</p><div class='foot'>{foot}</div></div>")


def stat(value: str, label: str, color: str = INK) -> str:
    return (f"<div class='mw-glass mw-stat'><div class='v' style='color:{color}'>{value}</div>"
            f"<div class='k'>{label}</div></div>")


def ring(score, big: str, label: str) -> str:
    pct = 0 if score is None else max(0, min(100, score))
    col = score_color(score)
    txt = "--" if score is None else f"{score:.0f}"
    return (f"<div class='mw-glass mw-ring'>"
            f"<div class='dial' style='--p:{pct};--c:{col}'>"
            f"<span style='color:{col}'>{txt}</span></div>"
            f"<div class='meta'><div class='lbl'>{label}</div>"
            f"<div class='big'>{big}</div></div></div>")


def cue(stage: str, message: str, color: str, alert: bool = False) -> str:
    cls = "mw-glass mw-cue alert" if alert else "mw-glass mw-cue"
    return (f"<div class='{cls}' style='--cue:{color}'>"
            f"<div class='stage'>{stage}</div>"
            f"<div class='msg' style='color:{color}'>{message}</div></div>")


def metrics_panel(rows_html: str) -> str:
    return f"<div class='mw-glass mw-metrics'>{rows_html}</div>"


def bar_row(name: str, value: str, pct, color: str) -> str:
    w = 0 if pct is None else max(0, min(100, pct))
    return (f"<div class='mw-row'><span class='n'>{name}</span>"
            f"<span class='bar'><i style='width:{w:.0f}%;background:"
            f"linear-gradient(90deg,{color}99,{color})'></i></span>"
            f"<span class='v'>{value}</span></div>")


def chip(label: str, ok: bool) -> str:
    return (f"<span class='mw-chip {'on' if ok else 'off'}'>"
            f"{'●' if ok else '○'} {label}</span>")


def eyebrow(text: str) -> str:
    return f"<div class='mw-eyebrow'>{text}</div>"


def note(text: str) -> str:
    return f"<div class='mw-note'>{text}</div>"
