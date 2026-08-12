"""On-video drawing: user skeleton, ghost reference, correction arrows, HUD.

Everything here is presentation only — nothing in this module feeds back into
analysis. The constraint that shapes it is real-time legibility: one cue at a
time, one highlighted segment, and a ghost faint enough to read as a guide
rather than compete with the user's own skeleton.

Visually it speaks the same "colourful glass / aurora" language as the page
around it: frosted translucent panels with rounded corners, a hairline light
border, a coloured accent edge, and glow behind the few elements that must be
read at a glance (the score, the active cue, the correction arrow). Every
effect is plain OpenCV and every blur is confined to the small ROI it affects,
because this runs on every frame.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

from core import config
from core.landmarks import LM, PoseFrame

# --------------------------------------------------------------------------
# Palette — BGR, because OpenCV. Hex in comments is the RGB the page uses.
# --------------------------------------------------------------------------
COLOR_GOOD = (192, 240, 125)    # mint     #7DF0C0
COLOR_WARN = (140, 217, 255)    # amber    #FFD98C
COLOR_BAD = (163, 143, 255)     # coral    #FF8FA3
COLOR_GHOST = (255, 149, 220)   # lavender #DC95FF
COLOR_CREAM = (191, 244, 255)   # cream    #FFF4BF — primary attention accent
COLOR_PANEL = (42, 16, 18)      # indigo   #12102A
COLOR_TEXT = (255, 242, 245)    # near-white #F5F2FF
COLOR_DIM = (150, 130, 140)

# Older names kept as aliases so nothing outside this module has to change.
COLOR_OK = COLOR_GOOD
COLOR_FAULT = COLOR_BAD

FONT = cv2.FONT_HERSHEY_SIMPLEX
RADIUS = 14

SKELETON = [
    (LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER),
    (LM.LEFT_SHOULDER, LM.LEFT_ELBOW), (LM.LEFT_ELBOW, LM.LEFT_WRIST),
    (LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW), (LM.RIGHT_ELBOW, LM.RIGHT_WRIST),
    (LM.LEFT_SHOULDER, LM.LEFT_HIP), (LM.RIGHT_SHOULDER, LM.RIGHT_HIP),
    (LM.LEFT_HIP, LM.RIGHT_HIP),
    (LM.LEFT_HIP, LM.LEFT_KNEE), (LM.LEFT_KNEE, LM.LEFT_ANKLE),
    (LM.RIGHT_HIP, LM.RIGHT_KNEE), (LM.RIGHT_KNEE, LM.RIGHT_ANKLE),
    (LM.LEFT_ANKLE, LM.LEFT_FOOT_INDEX), (LM.RIGHT_ANKLE, LM.RIGHT_FOOT_INDEX),
]


def _point(shape, lm) -> Tuple[int, int]:
    h, w = shape[:2]
    return int(lm.x * w), int(lm.y * h)


# OpenCV's Hershey fonts are ASCII-only: anything else renders as "???".
_ASCII_MAP = {
    "—": "-", "–": "-", "·": "-", "°": "deg", "’": "'", "‘": "'",
    "“": '"', "”": '"', "≤": "<=", "≥": ">=", "→": "->", "×": "x", "…": "...",
}


def _ascii(text: str) -> str:
    for bad, good in _ASCII_MAP.items():
        text = text.replace(bad, good)
    return text.encode("ascii", "ignore").decode("ascii")


def _fit(text: str, max_width: int, scale: float, thickness: int) -> str:
    """Trim text to the available width, so a long cue never runs off frame."""
    text = _ascii(text)
    if max_width <= 0:
        return text
    width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
    if width <= max_width:
        return text
    while text and cv2.getTextSize(text + "...", cv2.FONT_HERSHEY_SIMPLEX,
                                   scale, thickness)[0][0] > max_width:
        text = text[:-1]
    return text.rstrip() + "..."


# ---------------------------------------------------------------------------
# Glass, glow and rounded geometry
# ---------------------------------------------------------------------------
def _shade(color, factor: float):
    """A dimmer (factor < 1) or brighter (factor > 1) version of a colour."""
    return tuple(int(max(0, min(255, round(c * factor)))) for c in color)


def _mix(a, b, t: float):
    """Blend towards `b`: t=0 is `a`, t=1 is `b`."""
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def _clip_box(shape, x1, y1, x2, y2):
    """Clamp a box to the frame; None when nothing of it is visible."""
    h, w = shape[:2]
    x1 = int(max(0, min(round(x1), w)))
    x2 = int(max(0, min(round(x2), w)))
    y1 = int(max(0, min(round(y1), h)))
    y2 = int(max(0, min(round(y2), h)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return x1, y1, x2, y2


def _soft_blur(img, k: int = 21):
    """GaussianBlur with the kernel shrunk to whatever the ROI can take."""
    h, w = img.shape[:2]
    k = min(k, w, h)
    if k % 2 == 0:
        k -= 1
    if k < 3:
        return img
    return cv2.GaussianBlur(img, (k, k), 0)


def _frost(roi, alpha: float):
    """Frosted-glass version of a ROI: blurred and tinted with the panel colour.

    The blur happens at 1/6 scale — a sixth of a linear dimension is far cheaper
    than a 21x21 kernel at full size, and the downscale/upscale pair is what
    makes it read as glass rather than as a smudge.
    """
    h, w = roi.shape[:2]
    sw, sh = max(2, w // 6), max(2, h // 6)
    small = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_AREA)
    small = _soft_blur(small, 5)
    tint = np.empty_like(small)
    tint[:] = COLOR_PANEL
    small = cv2.addWeighted(small, 1.0 - alpha, tint, alpha, 0)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _rounded_rect(img, p1, p2, color, thickness: int = 1, radius: int = RADIUS) -> None:
    """Rounded rectangle from four arcs plus four straight edges.

    `thickness < 0` fills it. Corner radius is clamped so small pills degrade
    into stadium shapes instead of drawing garbage.
    """
    x1, y1 = int(p1[0]), int(p1[1])
    x2, y2 = int(p2[0]), int(p2[1])
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    r = int(max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2)))
    corners = ((x1 + r, y1 + r, 180), (x2 - r, y1 + r, 270),
               (x2 - r, y2 - r, 0), (x1 + r, y2 - r, 90))
    if thickness < 0:
        if x2 - x1 > 2 * r:
            cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        if y2 - y1 > 2 * r:
            cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        for cx, cy, start in corners:
            if r > 0:
                cv2.ellipse(img, (cx, cy), (r, r), start, 0, 90, color, -1, cv2.LINE_AA)
        return
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)
    for cx, cy, start in corners:
        if r > 0:
            cv2.ellipse(img, (cx, cy), (r, r), start, 0, 90, color, thickness, cv2.LINE_AA)


def _glow(image, box, paint: Callable, k: int = 21, strength: float = 1.0) -> None:
    """Additive halo: `paint` draws into a transparent ROI-sized layer, which is
    blurred and added back. Cost is bounded by the box, never the frame."""
    clipped = _clip_box(image.shape, *box)
    if clipped is None:
        return
    x1, y1, x2, y2 = clipped
    roi = image[y1:y2, x1:x2]
    layer = np.zeros_like(roi)
    paint(layer, x1, y1)
    layer = _soft_blur(layer, k)
    if strength != 1.0:
        layer = cv2.convertScaleAbs(layer, alpha=strength)
    roi[:] = cv2.add(roi, layer)


def _soft_fill(image, p1, p2, color, alpha: float = 0.5, radius: int = 8) -> None:
    """A translucent rounded fill — used for track bars and inactive pills."""
    clipped = _clip_box(image.shape, p1[0], p1[1], p2[0], p2[1])
    if clipped is None:
        return
    x1, y1, x2, y2 = clipped
    roi = image[y1:y2, x1:x2]
    layer = roi.copy()
    _rounded_rect(layer, (0, 0), (x2 - x1 - 1, y2 - y1 - 1), color, -1, radius)
    roi[:] = cv2.addWeighted(layer, alpha, roi, 1.0 - alpha, 0)


def _glass_panel(image, x1, y1, x2, y2, alpha: float = 0.55,
                 radius: int = RADIUS, accent=None) -> None:
    """Frosted glass card: blur what is behind it, tint it with the panel
    colour, round the corners, then a hairline light border and an optional
    accent edge down the left."""
    clipped = _clip_box(image.shape, x1, y1, x2, y2)
    if clipped is None:
        return
    x1, y1, x2, y2 = clipped
    roi = image[y1:y2, x1:x2]
    h, w = roi.shape[:2]

    try:
        # Slice views off live webcam frames occasionally trip OpenCV's C++
        # layer inside resize/blur. A cosmetic effect must never take the app
        # down, so frost on a contiguous copy and fall back to a plain tint.
        glass = _frost(np.ascontiguousarray(roi), alpha)
    except cv2.error:
        tint = np.empty_like(roi)
        tint[:] = COLOR_PANEL
        glass = cv2.addWeighted(roi, 1.0 - alpha, tint, alpha, 0)
    mask = np.zeros((h, w), np.uint8)
    _rounded_rect(mask, (0, 0), (w - 1, h - 1), (255, 255, 255), -1, radius)
    roi[:] = cv2.copyTo(glass, mask, roi)

    # Hairline border. Pre-mixed rather than alpha-blended per pixel: over a
    # panel this dark the result is the same light rgba stroke, for one cheap
    # pass instead of two full-ROI ones.
    _rounded_rect(roi, (0, 0), (w - 1, h - 1), _mix(COLOR_PANEL, COLOR_TEXT, 0.42),
                  1, radius)

    if accent is not None and h > 14:
        inset = min(8, h // 4)
        _rounded_rect(roi, (1, inset - 2), (7, h - inset + 1),
                      _shade(accent, 0.32), -1, 3)
        _rounded_rect(roi, (2, inset), (5, h - inset), accent, -1, 2)


def _glow_text(image, text: str, org, scale: float, color, thickness: int = 2,
               glow=None, strength: float = 1.0) -> None:
    """Text with a soft halo under it, so it survives a busy video behind it."""
    text = _ascii(text)
    if not text:
        return
    halo = glow if glow is not None else color
    x, y = int(org[0]), int(org[1])
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thickness)
    pad = int(10 + scale * 10)
    box = (x - pad, y - th - pad, x + tw + pad, y + base + pad)

    def _wide(layer, ox, oy):
        cv2.putText(layer, text, (x - ox, y - oy), FONT, scale,
                    _shade(halo, 0.42), thickness + 5, cv2.LINE_AA)

    def _tight(layer, ox, oy):
        cv2.putText(layer, text, (x - ox, y - oy), FONT, scale,
                    _shade(halo, 0.55), thickness + 2, cv2.LINE_AA)

    _glow(image, box, _wide, k=21, strength=0.85 * strength)
    _glow(image, box, _tight, k=9, strength=0.7 * strength)
    cv2.putText(image, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Ghost reference
# ---------------------------------------------------------------------------
def draw_ghost(image, ghost: Optional[Dict[LM, Tuple[int, int]]],
               emphasis: bool = False) -> None:
    """Draw the reference skeleton as a faint dashed guide behind the user.

    `emphasis` is the last rung of the coaching modality ladder: the same guide,
    drawn thicker, brighter and with a lavender bloom behind it, for a user who
    has not responded to words or an arrow. It never becomes a different
    reference — only a louder one.
    """
    if not ghost:
        return
    from core.reference import GHOST_BONES

    thickness = 6 if emphasis else 4
    alpha = 0.9 if emphasis else 0.7

    skip = (LM.LEFT_EYE, LM.RIGHT_EYE, LM.LEFT_EAR, LM.RIGHT_EAR, LM.NOSE)
    points = list(ghost.values())
    pad = 28
    box = _clip_box(image.shape,
                    min(p[0] for p in points) - pad, min(p[1] for p in points) - pad,
                    max(p[0] for p in points) + pad, max(p[1] for p in points) + pad)
    if box is None:
        return
    x1, y1, x2, y2 = box
    roi = image[y1:y2, x1:x2]
    layer = roi.copy()

    if emphasis:
        def _bloom(dst, ox, oy):
            for a, b in GHOST_BONES:
                if a in ghost and b in ghost:
                    cv2.line(dst, (ghost[a][0] - ox, ghost[a][1] - oy),
                             (ghost[b][0] - ox, ghost[b][1] - oy),
                             _shade(COLOR_GHOST, 0.5), thickness + 6, cv2.LINE_AA)
        _glow(layer, (0, 0, x2 - x1, y2 - y1), _bloom, k=21, strength=0.9)

    for a, b in GHOST_BONES:
        if a not in ghost or b not in ghost:
            continue
        _dashed_line(layer, (ghost[a][0] - x1, ghost[a][1] - y1),
                     (ghost[b][0] - x1, ghost[b][1] - y1),
                     COLOR_GHOST, thickness, dash=12)
    for lm, pt in ghost.items():
        if lm in skip:
            continue
        centre = (pt[0] - x1, pt[1] - y1)
        cv2.circle(layer, centre, 7 if emphasis else 5, COLOR_GHOST, -1, cv2.LINE_AA)
        cv2.circle(layer, centre, 3 if emphasis else 2,
                   _mix(COLOR_GHOST, COLOR_TEXT, 0.6), -1, cv2.LINE_AA)

    # Blended rather than drawn solid, so the guide never outshouts the user.
    roi[:] = cv2.addWeighted(layer, alpha, roi, 1.0 - alpha, 0)


def _dashed_line(img, p1, p2, color, thickness: int, dash: int = 12) -> None:
    p1 = np.array(p1, dtype=np.float32)
    p2 = np.array(p2, dtype=np.float32)
    length = float(np.linalg.norm(p2 - p1))
    if length < 1.0:
        return
    direction = (p2 - p1) / length
    step = dash * 2
    for start in range(0, int(length), step):
        a = p1 + direction * start
        b = p1 + direction * min(start + dash, length)
        cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                 color, thickness, cv2.LINE_AA)


def draw_skeleton(image, pose: PoseFrame, highlight: Sequence[LM] = ()) -> None:
    """Draw the tracked skeleton.

    Low-confidence joints are drawn dim rather than hidden, so the user can see
    what the system is and is not tracking — the same honesty rule the metrics
    follow, expressed visually. The faulted segment gets a coral bloom so the
    eye lands on it first.
    """
    if not pose.detected:
        return

    highlight_set = {int(l) for l in highlight}

    faulted = []
    for a, b in SKELETON:
        la, lb = pose.get(a), pose.get(b)
        if la is None or lb is None:
            continue
        if int(a) in highlight_set and int(b) in highlight_set:
            faulted.append((_point(image.shape, la), _point(image.shape, lb)))

    # One blur pass for the whole highlighted segment, not one per bone.
    if faulted:
        xs = [p[0] for seg in faulted for p in seg]
        ys = [p[1] for seg in faulted for p in seg]
        pad = 22

        def _bloom(layer, ox, oy):
            for pa, pb in faulted:
                cv2.line(layer, (pa[0] - ox, pa[1] - oy), (pb[0] - ox, pb[1] - oy),
                         _shade(COLOR_BAD, 0.55), 11, cv2.LINE_AA)
        _glow(image, (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad),
              _bloom, k=21, strength=0.95)

    for a, b in SKELETON:
        la, lb = pose.get(a), pose.get(b)
        if la is None or lb is None:
            continue
        pa, pb = _point(image.shape, la), _point(image.shape, lb)
        confident = min(la.visibility, lb.visibility) >= config.VISIBILITY_THRESHOLD
        if int(a) in highlight_set and int(b) in highlight_set:
            cv2.line(image, pa, pb, _shade(COLOR_BAD, 0.6), 8, cv2.LINE_AA)
            cv2.line(image, pa, pb, COLOR_BAD, 5, cv2.LINE_AA)
            continue
        if confident:
            cv2.line(image, pa, pb, _shade(COLOR_GOOD, 0.35), 5, cv2.LINE_AA)
            cv2.line(image, pa, pb, COLOR_GOOD, 2, cv2.LINE_AA)
        else:
            # Dim, never hidden: the user sees what tracking is unsure about.
            cv2.line(image, pa, pb, COLOR_DIM, 1, cv2.LINE_AA)

    for idx, lm in pose.landmarks.items():
        # Body only. Indices 0-10 are the face (eyes, nose, ears, mouth) —
        # dots on someone's face read as surveillance, not coaching, and no
        # exercise rule ever uses them.
        if idx > 32 or idx < 11:
            continue
        confident = lm.visibility >= config.VISIBILITY_THRESHOLD
        pt = _point(image.shape, lm)
        if idx in highlight_set:
            cv2.circle(image, pt, 13, _shade(COLOR_BAD, 0.55), 3, cv2.LINE_AA)
            cv2.circle(image, pt, 11, COLOR_BAD, 2, cv2.LINE_AA)
            cv2.circle(image, pt, 5, COLOR_BAD, -1, cv2.LINE_AA)
            cv2.circle(image, pt, 2, COLOR_CREAM, -1, cv2.LINE_AA)
            continue
        if confident:
            cv2.circle(image, pt, 6, _shade(COLOR_GOOD, 0.35), -1, cv2.LINE_AA)
            cv2.circle(image, pt, 4, COLOR_GOOD, -1, cv2.LINE_AA)
        else:
            cv2.circle(image, pt, 3, COLOR_DIM, -1, cv2.LINE_AA)


def draw_arrow(image, arrow: Optional[Tuple[Tuple[int, int], Tuple[int, int]]],
               label: str = "") -> None:
    """Directional cue: from where the joint is to where the guide puts it.

    Coral shaft (it marks a fault), cream head (it marks the target), glow under
    both so it reads over any background.
    """
    if arrow is None:
        return
    start, end = arrow
    start = (int(start[0]), int(start[1]))
    end = (int(end[0]), int(end[1]))
    pad = 26

    def _bloom(layer, ox, oy):
        cv2.arrowedLine(layer, (start[0] - ox, start[1] - oy),
                        (end[0] - ox, end[1] - oy), _shade(COLOR_BAD, 0.6), 11,
                        cv2.LINE_AA, tipLength=0.34)
    _glow(image, (min(start[0], end[0]) - pad, min(start[1], end[1]) - pad,
                  max(start[0], end[0]) + pad, max(start[1], end[1]) + pad),
          _bloom, k=21, strength=0.95)

    cv2.arrowedLine(image, start, end, COLOR_BAD, 5, cv2.LINE_AA, tipLength=0.32)

    # Where the joint is now, and — in cream — where it should go.
    cv2.circle(image, start, 5, _shade(COLOR_BAD, 0.75), -1, cv2.LINE_AA)
    vec = np.array(end, dtype=np.float32) - np.array(start, dtype=np.float32)
    length = float(np.linalg.norm(vec))
    if length > 6.0:
        head = np.array(end, dtype=np.float32) - vec / length * min(30.0, length * 0.55)
        cv2.arrowedLine(image, (int(head[0]), int(head[1])), end, COLOR_CREAM, 3,
                        cv2.LINE_AA, tipLength=0.85)
    if label:
        # Above the whole arrow, not across its middle: the chip must never
        # sit on top of the head, which is the part carrying the meaning.
        anchor = ((start[0] + end[0]) // 2, min(start[1], end[1]) - 18)
        _label(image, label, anchor, COLOR_CREAM)


def _label(image, text: str, anchor: Tuple[int, int], color) -> None:
    """Small glass chip used for arrow cues and keys."""
    scale = 0.5
    text = _fit(text, image.shape[1] - 64, scale, 1)
    if not text:
        return
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 1)
    x, y = anchor
    x = max(16, min(x - tw // 2, image.shape[1] - tw - 18))
    y = max(th + 20, min(y, image.shape[0] - 12))
    _glass_panel(image, x - 18, y - th - 12, x + tw + 12, y + 10,
                 alpha=0.62, radius=11, accent=color)
    cv2.putText(image, text, (x, y), FONT, scale, COLOR_TEXT, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# HUD
# ---------------------------------------------------------------------------
def draw_banner(image, text: str, sub: Optional[str] = None, fault: bool = False) -> None:
    """Top banner carrying the single active message. Never more than one."""
    if not text:
        return
    h, w = image.shape[:2]
    pad = max(10, int(w * 0.018))
    card_h = 84 if sub else 60
    x1, y1 = pad, pad
    x2, y2 = w - pad, pad + card_h

    accent = COLOR_BAD if fault else COLOR_GOOD
    _glass_panel(image, x1, y1, x2, y2, alpha=0.62, radius=16, accent=accent)

    tx = x1 + 26
    avail = (x2 - tx) - 24
    # Shrink a long cue a step or two before resorting to an ellipsis.
    scale = 0.72 if w >= 620 else 0.56
    for candidate in (scale, scale - 0.08, scale - 0.14):
        if cv2.getTextSize(_ascii(text), FONT, candidate, 2)[0][0] <= avail:
            scale = candidate
            break
        scale = candidate
    _glow_text(image, _fit(text, avail, scale, 2), (tx, y1 + (40 if sub else 39)),
               scale, COLOR_TEXT, 2, glow=accent, strength=0.75)
    if sub:
        cv2.putText(image, _fit(sub, avail, 0.46, 1), (tx, y1 + 66),
                    FONT, 0.46, COLOR_DIM, 1, cv2.LINE_AA)


def draw_score_badge(image, score: Optional[float], phase: str,
                     reps: Optional[int] = None,
                     hold: Optional[float] = None) -> None:
    """Top-right badge: score ring, phase, and rep or hold readout."""
    h, w = image.shape[:2]
    pad = max(10, int(w * 0.018))
    box_w, box_h = min(186, w - 2 * pad), 136
    x0 = w - box_w - pad
    y0 = min(pad + 96, max(pad, h - box_h - pad))

    if score is not None:
        ring = COLOR_GOOD if score >= 80 else (COLOR_WARN if score >= 55 else COLOR_BAD)
    else:
        ring = COLOR_DIM
    _glass_panel(image, x0, y0, x0 + box_w, y0 + box_h, alpha=0.6, radius=16, accent=ring)

    cx, cy, radius = x0 + 52, y0 + 50, 30
    cv2.circle(image, (cx, cy), radius, _mix(COLOR_PANEL, COLOR_TEXT, 0.22), 7, cv2.LINE_AA)

    if score is not None:
        sweep = max(0.0, min(360.0, 360.0 * (float(score) / 100.0)))

        def _bloom(layer, ox, oy):
            cv2.ellipse(layer, (cx - ox, cy - oy), (radius, radius), -90, 0, sweep,
                        _shade(ring, 0.7), 11, cv2.LINE_AA)
        _glow(image, (cx - radius - 14, cy - radius - 14,
                      cx + radius + 14, cy + radius + 14), _bloom, k=21, strength=0.9)

        cv2.ellipse(image, (cx, cy), (radius, radius), -90, 0, sweep, ring, 7, cv2.LINE_AA)
        # Rounded caps, plus a cream head so the arc reads as a progress dial.
        cv2.circle(image, (cx, cy - radius), 3, ring, -1, cv2.LINE_AA)
        ang = math.radians(-90.0 + sweep)
        hx, hy = int(cx + radius * math.cos(ang)), int(cy + radius * math.sin(ang))
        cv2.circle(image, (hx, hy), 4, ring, -1, cv2.LINE_AA)
        cv2.circle(image, (hx, hy), 2, COLOR_CREAM, -1, cv2.LINE_AA)
        text = f"{score:.0f}"
    else:
        text = "--"

    scale = 0.7
    while scale > 0.4 and cv2.getTextSize(text, FONT, scale, 2)[0][0] > 2 * radius - 16:
        scale -= 0.04
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 2)
    _glow_text(image, text, (cx - tw // 2, cy + th // 2), scale, COLOR_TEXT, 2,
               glow=ring, strength=1.0)
    _centred(image, "SCORE", cx, y0 + 99, 0.34, COLOR_DIM)

    right_x = x0 + 98
    right_w = (x0 + box_w - 12) - right_x
    if reps is not None:
        cv2.putText(image, _fit(f"{reps}", right_w, 0.86, 2), (right_x, y0 + 52),
                    FONT, 0.86, COLOR_TEXT, 2, cv2.LINE_AA)
        cv2.putText(image, "REPS", (right_x, y0 + 72), FONT, 0.34,
                    COLOR_DIM, 1, cv2.LINE_AA)
    elif hold is not None:
        cv2.putText(image, _fit(f"{hold:.1f}s", right_w, 0.64, 2), (right_x, y0 + 52),
                    FONT, 0.64, COLOR_TEXT, 2, cv2.LINE_AA)
        cv2.putText(image, "HOLD", (right_x, y0 + 72), FONT, 0.34,
                    COLOR_DIM, 1, cv2.LINE_AA)

    if phase:
        pill_x1, pill_x2 = x0 + 12, x0 + box_w - 12
        pill_y1, pill_y2 = y0 + 104, y0 + 126
        _soft_fill(image, (pill_x1, pill_y1), (pill_x2, pill_y2),
                   _mix(COLOR_PANEL, COLOR_WARN, 0.55), 0.28, 11)
        label = _fit(phase.upper(), pill_x2 - pill_x1 - 16, 0.38, 1)
        _centred(image, label, (pill_x1 + pill_x2) // 2, pill_y2 - 7, 0.38, COLOR_WARN)


def _centred(image, text: str, cx: int, baseline: int, scale: float, color,
             thickness: int = 1) -> None:
    text = _ascii(text)
    if not text:
        return
    tw = cv2.getTextSize(text, FONT, scale, thickness)[0][0]
    cv2.putText(image, text, (int(cx - tw // 2), int(baseline)), FONT, scale,
                color, thickness, cv2.LINE_AA)


def draw_phase_track(image, phases: Sequence[str], current: str) -> None:
    """A small strip showing where in the movement cycle the user is."""
    if not phases:
        return
    h, w = image.shape[:2]
    pad = max(10, int(w * 0.018))
    gap = 12
    seg_w = max(56, min(116, (w - 2 * pad) // len(phases)))
    total = seg_w * len(phases)
    x0 = max(pad, (w - total) // 2)
    y0 = h - 104

    for i, name in enumerate(phases):
        x = x0 + i * seg_w
        x2 = min(x + seg_w - gap, w - pad)
        if x2 - x < 8:
            continue
        active = name == current
        if active:
            def _bloom(layer, ox, oy, _x=x, _x2=x2):
                _rounded_rect(layer, (_x - ox, y0 - oy), (_x2 - ox, y0 + 8 - oy),
                              _shade(COLOR_CREAM, 0.65), -1, 4)
            _glow(image, (x - 12, y0 - 12, x2 + 12, y0 + 20), _bloom, k=15, strength=0.85)
            _rounded_rect(image, (x, y0), (x2, y0 + 8), COLOR_CREAM, -1, 4)
        else:
            _soft_fill(image, (x, y0 + 2), (x2, y0 + 8),
                       _mix(COLOR_PANEL, COLOR_TEXT, 0.45), 0.45, 3)
        label = _fit(name.upper(), x2 - x, 0.38, 1)
        cv2.putText(image, label, (x, y0 + 26), FONT, 0.38,
                    COLOR_CREAM if active else COLOR_DIM, 1, cv2.LINE_AA)


def draw_calibration(image, progress: float) -> None:
    """Body-map calibration progress bar."""
    h, w = image.shape[:2]
    progress = max(0.0, min(1.0, float(progress)))
    scale = 0.52
    label = _fit("Building your body map - stand in full view", w - 96, scale, 1)
    tw = cv2.getTextSize(label, FONT, scale, 1)[0][0]

    bar_w = max(180, int(w * 0.5))
    inner = min(max(bar_w, tw), w - 64)
    bar_w = min(bar_w, inner)
    x = (w - inner) // 2
    bar_h, bar_y = 12, h - 60

    _glass_panel(image, x - 24, bar_y - 56, x + inner + 24, bar_y + bar_h + 18,
                 alpha=0.62, radius=16, accent=COLOR_CREAM)
    _centred(image, label, w // 2, bar_y - 22, scale, COLOR_TEXT)

    bar_x = x + (inner - bar_w) // 2
    _soft_fill(image, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
               _mix(COLOR_PANEL, COLOR_TEXT, 0.35), 0.5, 6)
    fill = int(bar_w * progress)
    if fill > 4:
        def _bloom(layer, ox, oy):
            _rounded_rect(layer, (bar_x - ox, bar_y - oy),
                          (bar_x + fill - ox, bar_y + bar_h - oy),
                          _shade(COLOR_CREAM, 0.7), -1, 6)
        _glow(image, (bar_x - 14, bar_y - 14, bar_x + fill + 14, bar_y + bar_h + 14),
              _bloom, k=21, strength=0.85)
        _rounded_rect(image, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h),
                      COLOR_CREAM, -1, 6)


def draw_legend(image, ghost_on: bool) -> None:
    """Tiny key so the ghost is never mistaken for a second person."""
    if not ghost_on:
        return
    h, w = image.shape[:2]
    pad = max(10, int(w * 0.018))
    text = _ascii("reference guide")
    scale = 0.42
    tw = cv2.getTextSize(text, FONT, scale, 1)[0][0]
    y = h - 46
    x1 = pad
    x2 = min(w - pad, x1 + 58 + tw + 16)

    _glass_panel(image, x1, y - 17, x2, y + 17, alpha=0.55, radius=14)
    _dashed_line(image, (x1 + 14, y), (x1 + 44, y), COLOR_GHOST, 3, dash=7)
    cv2.putText(image, text, (x1 + 54, y + 5), FONT, scale,
                _mix(COLOR_DIM, COLOR_TEXT, 0.35), 1, cv2.LINE_AA)
