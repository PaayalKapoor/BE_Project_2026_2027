"""
PowerPulse - Lateral Raise Analyzer (video-file version + logging for trainer comparison)
Camera must face the person (front view).

Usage:
  python powerpulse_lateral_raise_fixed.py --input path/to/video.mp4
  python powerpulse_lateral_raise_fixed.py --input path/to/folder --no-display      # batch, headless

Outputs (in --out folder, default ./outputs_raise):
  <video>_frames.csv        per-frame angles / states (for debugging)
  <video>_reps.csv          one row per rep PER ARM with predicted label + error codes
  all_reps_predicted.csv    all videos combined  -> feed this to compare_labels.py
"""
import argparse
import csv
import glob
import os
import urllib.request
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ----------------------------------------------------------------- config
MODEL_PATH = "pose_landmarker_heavy.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_heavy/float16/latest/"
    "pose_landmarker_heavy.task"
)
VIDEO_PATH = r"C:\Users\ADMIN\Documents\GitHub\BE_Project_2026_2027\videos\Phase_1\Lateral_Raises\Sayalee_LTR_R.mp4"

# MediaPipe pose landmark indices (person's own left/right, video NOT mirrored)
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
ARM_IDX = {"LEFT": (L_HIP, L_SHOULDER, L_ELBOW, L_WRIST),
           "RIGHT": (R_HIP, R_SHOULDER, R_ELBOW, R_WRIST)}

REST_ENTER = 30        # shoulder angle above this = arm leaves rest (rep starts)
REST_EXIT = 25         # ...below this = back at rest (rep ends); gap avoids chatter around one value
TOP_MIN = 92           # must reach this to be a full rep
TOP_MAX = 105          # above this = raised too high
PARTIAL_MIN_PEAK = 55  # a rise that never exceeds this is ignored (just a wobble)
ELBOW_MIN = 150      # elbow angle below this = elbow bent
ELBOW_MID = 175
ELBOW_CHECK_MIN = 40   # only judge the elbow once the arm is clearly raised
SYM_MAX_DIFF = 15      # left/right shoulder-angle difference allowed
ERR_MIN_FRAMES = 5     # an error must persist this many CONSECUTIVE frames to flag the rep (1 = flag instantly)
TORSO_TILT_MAX = 5.65     # deg: tilt away from its value at the start of the rep, above this -> TORSO_LEAN (PLACEHOLDER - tune with m_torso_tilt_shift)

# ---- momentum / swing detection.  These numbers are PLACEHOLDERS: tune them on your own videos
# ---- (the m_* columns in the reps CSV are there for exactly that - see compare_labels.py)
VEL_ERR_MIN_FRAMES = 3  # velocity-based errors need fewer consecutive frames: the velocity is already averaged over VEL_WINDOW_S
VEL_WINDOW_S = 0.2     # velocities are measured over this many seconds (less noise, independent of fps)
FAST_LIFT_VEL = 142    # deg/s: shoulder angle rising faster than this   -> FAST_LIFT  (jerked up with momentum)
FAST_LOWER_VEL = 150   # deg/s: shoulder angle falling faster than this  -> FAST_LOWER (dropped, not controlled)
LOWER_MIN_S = 0.985     # s: lowering (peak -> back at rest) shorter than this = not controlled -> LOWERED_TOO_QUICKLY (PLACEHOLDER - tune with m_lower_s)
TORSO_SWING_VEL = 8.95   # deg/s: side-to-side torso tilt changing faster than this -> TORSO_SWING (body English)
VEL_MSG = {"FAST_LIFT": "TOO FAST - control the lift", "FAST_LOWER": "TOO FAST - control the lowering",
           "TORSO_SWING": "BODY SWING - keep torso still"}
ALPHA = 0.3            # smoothing weight of previous value
VIS_MIN = 0.5

window_name = "PowerPulse - Lateral Raise Detector"
summary_window_name = "Lateral Raise Summary"
stop_flag = False
stop_button_rect = [0, 0, 0, 0]
summary_close_flag = False
summary_close_rect = (220, 200, 330, 245)

REP_FIELDS = ["video", "rep_id", "arm", "start_s", "end_s", "label", "errors",
              "peak_angle", "min_elbow_angle", "reached_top",
              "m_peak_lift_vel", "m_peak_lower_vel", "m_peak_torso_vel", "m_lift_s", "m_lower_s", "m_torso_tilt_max", "m_torso_tilt_shift"]   # m_* = tuning metrics
FRAME_FIELDS = ["frame", "time_s", "detected", "l_shoulder", "l_elbow", "l_state", "l_errors", "l_vel",
                "r_shoulder", "r_elbow", "r_state", "r_errors", "r_vel", "torso_tilt", "torso_vel"]

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ----------------------------------------------------------------- model
def download_model():
    if os.path.exists(MODEL_PATH):
        print(f"Model already present: {MODEL_PATH}")
        return
    print(f"Downloading model -> {MODEL_PATH}")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.")


# ----------------------------------------------------------------- math
def calculate_angle(a, b, c):
    """Angle at b (degrees). Pass PIXEL coordinates: normalised x/y distort angles on non-square frames."""
    a, b, c = np.array(a, float), np.array(b, float), np.array(c, float)
    ba, bc = a - b, c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def arm_angles(lm, side, w, h):
    """(shoulder_angle, elbow_angle, landmarks) or None if any joint is not visible enough."""
    pts = [lm[i] for i in ARM_IDX[side]]               # hip, shoulder, elbow, wrist
    if min(p.visibility for p in pts) < VIS_MIN:
        return None
    px = [(p.x * w, p.y * h) for p in pts]
    return calculate_angle(px[0], px[1], px[2]), calculate_angle(px[1], px[2], px[3]), pts


class Velocity:
    """Rate of change (units per second) over a sliding TIME window, so it does not depend on fps."""
    def __init__(self, window_s):
        self.window_s = window_s
        self.hist = deque()

    def reset(self):
        self.hist.clear()

    def update(self, t, value):
        self.hist.append((t, value))
        while len(self.hist) > 2 and t - self.hist[1][0] >= self.window_s:
            self.hist.popleft()
        (t0, v0), (t1, v1) = self.hist[0], self.hist[-1]
        dt = t1 - t0
        return (v1 - v0) / dt if dt >= 0.5 * self.window_s else 0.0


def torso_tilt(lm, w, h):
    """Side-to-side tilt of the shoulder-midpoint vs hip-midpoint, degrees from vertical (front view)."""
    pts = [lm[i] for i in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)]
    if min(p.visibility for p in pts) < VIS_MIN:
        return None
    sx, sy = (pts[0].x + pts[1].x) / 2 * w, (pts[0].y + pts[1].y) / 2 * h
    hx, hy = (pts[2].x + pts[3].x) / 2 * w, (pts[2].y + pts[3].y) / 2 * h
    return float(np.degrees(np.arctan2(sx - hx, hy - sy)))


# ----------------------------------------------------------------- rep counter
class RepCounter:
    """
    One per arm.  DOWN -> RISING -> UP -> LOWERING -> DOWN.
    A rep ATTEMPT runs from leaving rest to returning to rest.  Reps that never reach TOP_MIN are
    logged as NOT_HIGH_ENOUGH instead of being silently dropped.  An error flags the rep only if it
    fires on ERR_MIN_FRAMES CONSECUTIVE frames.
    """
    def __init__(self, side):
        self.side = side
        self.state = "DOWN"
        self.reps = 0               # reps with no flagged error
        self.attempts = 0
        self.records = []
        self.last_flash = -99.0
        self.last_result = "-"
        self._clear()

    def _clear(self):
        self.tilt0, self.tilt_max, self.tilt_shift = None, 0.0, 0.0
        self.start_t = 0.0
        self.peak, self.peak_t = 0.0, 0.0
        self.min_elbow = 999.0
        self.peak_up = self.peak_down = self.peak_torso = 0.0
        self.err_streak, self.err_max = {}, {}

    def flashing(self, t):
        return t - self.last_flash < 0.8

    def break_streaks(self):
        """Arm not visible this frame: nothing can be 'consecutive' across the gap."""
        self.err_streak.clear()
        return []

    def update(self, angle, elbow, t, video_name, vel=0.0, torso_swing=False, torso_vel=0.0, tilt=None):
        """Feed one frame. Returns the error codes firing on this frame."""
        if angle is None:
            return self.break_streaks()
        if self.state == "DOWN":
            if angle <= REST_ENTER:
                return []
            self._clear()
            self.state, self.start_t = "RISING", t

        if angle > self.peak:
            self.peak, self.peak_t = angle, t
        self.peak_up = max(self.peak_up, vel)
        self.peak_down = max(self.peak_down, -vel)
        self.peak_torso = max(self.peak_torso, abs(torso_vel))
        lean_now = False
        if tilt is not None:
            if self.tilt0 is None:
                self.tilt0 = tilt                      # lean at the start of the rep = this person's own baseline
            shift = abs(tilt - self.tilt0)
            self.tilt_max = max(self.tilt_max, abs(tilt))
            self.tilt_shift = max(self.tilt_shift, shift)
            lean_now = shift > TORSO_TILT_MAX
        frame_errs = []
        if elbow is not None and angle >= ELBOW_CHECK_MIN:
            self.min_elbow = min(self.min_elbow, elbow)
            #if elbow < ELBOW_MIN:
                #frame_errs.append("ELBOW_BENT")
            #if elbow > ELBOW_MID:
                #frame_errs.append("ELBOW_STRAIGHT")
        if lean_now:
            frame_errs.append("TORSO_LEAN")
        if angle > TOP_MAX:
            frame_errs.append("TOO_HIGH")
        #if sym_bad:
            #frame_errs.append("UNEVEN_RAISE")
        if vel > FAST_LIFT_VEL:
            frame_errs.append("FAST_LIFT")
        if vel < -FAST_LOWER_VEL:
            frame_errs.append("FAST_LOWER")
        if torso_swing:
            frame_errs.append("TORSO_SWING")

        for code in list(self.err_streak):               # a streak ends the moment the error stops firing
            if code not in frame_errs:
                self.err_streak[code] = 0
        for code in frame_errs:
            self.err_streak[code] = self.err_streak.get(code, 0) + 1
            self.err_max[code] = max(self.err_max.get(code, 0), self.err_streak[code])

        if self.state == "RISING" and angle >= TOP_MIN:
            self.state = "UP"
        elif self.state == "UP" and angle < TOP_MIN - 10:
            self.state = "LOWERING"
        elif self.state == "LOWERING" and angle >= TOP_MIN:
            self.state = "UP"

        if angle < REST_EXIT:
            self._finish(t, video_name)
            self.state = "DOWN"
        return frame_errs

    def _finish(self, t, video_name):
        reached = self.peak >= TOP_MIN
        if not reached and self.peak < PARTIAL_MIN_PEAK:
            return                                          # wobble, not a rep attempt
        errs = sorted(c for c, n in self.err_max.items() if n >= (VEL_ERR_MIN_FRAMES if c in VEL_MSG else ERR_MIN_FRAMES))
        if reached and t - self.peak_t < LOWER_MIN_S:
            errs = sorted(set(errs) | {"LOWERED_TOO_QUICKLY"})
        if not reached:
            errs = ["NOT_HIGH_ENOUGH"]
        self.attempts += 1
        if not errs:
            self.reps += 1
            self.last_flash = t
        self.last_result = "OK" if not errs else ";".join(errs)
        self.records.append({
            "video": video_name, "rep_id": self.attempts, "arm": self.side.lower(),
            "start_s": round(self.start_t, 3), "end_s": round(t, 3),
            "label": "incorrect" if errs else "correct", "errors": ";".join(errs),
            "peak_angle": round(self.peak, 1),
            "min_elbow_angle": round(self.min_elbow, 1) if self.min_elbow < 999 else "",
            "reached_top": int(reached),
            "m_peak_lift_vel": round(self.peak_up, 1), "m_peak_lower_vel": round(self.peak_down, 1),
            "m_peak_torso_vel": round(self.peak_torso, 1), "m_torso_tilt_max": round(self.tilt_max, 1), "m_torso_tilt_shift": round(self.tilt_shift, 1),
            "m_lift_s": round(self.peak_t - self.start_t, 2), "m_lower_s": round(t - self.peak_t, 2)})


# ----------------------------------------------------------------- UI helpers
def get_stop_button_rect(frame):
    h, w = frame.shape[:2]
    x2, y2 = w - 15, h - 15
    return [x2 - 150, y2 - 55, x2, y2]


def on_mouse(event, x, y, flags, param):
    global stop_flag
    x1, y1, x2, y2 = param
    if event == cv2.EVENT_LBUTTONDOWN and x1 <= x <= x2 and y1 <= y <= y2:
        stop_flag = True


def on_summary_mouse(event, x, y, flags, param):
    global summary_close_flag
    x1, y1, x2, y2 = param
    if event == cv2.EVENT_LBUTTONDOWN and x1 <= x <= x2 and y1 <= y <= y2:
        summary_close_flag = True


def rounded_rect(image, rect, color, radius=15):
    x1, y1, x2, y2 = rect
    radius = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(image, (x1, y1 + radius), (x2, y2 - radius), color, -1, cv2.LINE_AA)
    for cx, cy in ((x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
                   (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)):
        cv2.circle(image, (cx, cy), radius, color, -1, cv2.LINE_AA)


def draw_stop_button(frame):
    x1, y1, x2, y2 = stop_button_rect
    rounded_rect(frame, (x1, y1, x2, y2), (0, 0, 255), radius=15)
    (tw, th), _ = cv2.getTextSize("STOP", FONT, 0.85, 2)
    cv2.putText(frame, "STOP", (x1 + (x2 - x1 - tw) // 2, y1 + (y2 - y1 + th) // 2),
                FONT, 0.85, (255, 255, 255), 2)


def fit(frame, max_w=900, max_h=650):
    h, w = frame.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        return cv2.resize(frame, (int(w * scale), int(h * scale)))
    return frame.copy()


def present(img, show):
    """Show a frame, handle keys. Returns True if the user wants to quit."""
    if not show:
        return False
    stop_button_rect[:] = get_stop_button_rect(img)    # in-place so the mouse callback sees it
    draw_stop_button(img)
    cv2.imshow(window_name, img)
    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), 27):
        return True
    if key == ord(' '):                                  # space = pause / resume
        while not stop_flag:
            k = cv2.waitKey(50) & 0xFF
            if k == ord(' '):
                break
            if k in (ord('q'), 27):
                return True
    return stop_flag


def make_summary_image(left, right):
    img = np.zeros((260, 350, 3), np.uint8)
    img[:] = (40, 40, 40)
    cv2.putText(img, "SUMMARY", (20, 40), FONT, 0.9, (255, 255, 255), 2)
    cv2.putText(img, f"Left : {left.reps} correct / {left.attempts}", (20, 90), FONT, 0.7, (255, 255, 255), 2)
    cv2.putText(img, f"Right: {right.reps} correct / {right.attempts}", (20, 130), FONT, 0.7, (255, 255, 255), 2)
    rounded_rect(img, summary_close_rect, (0, 0, 255), radius=12)
    cv2.putText(img, "Close", (summary_close_rect[0] + 25, summary_close_rect[1] + 30), FONT, 0.7, (255, 255, 255), 2)
    return img


def show_summary(left, right):
    global summary_close_flag
    summary_close_flag = False
    cv2.namedWindow(summary_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(summary_window_name, 350, 260)
    cv2.setMouseCallback(summary_window_name, on_summary_mouse, summary_close_rect)
    while True:
        cv2.imshow(summary_window_name, make_summary_image(left, right))
        if (cv2.waitKey(20) & 0xFF) == 27 or summary_close_flag:
            break
    cv2.destroyWindow(summary_window_name)


# ----------------------------------------------------------------- side panel + on-video labels
# Same look as the squat analyzer: ONE panel beside the video (never on top of the person), long text is
# word-wrapped so nothing is clipped on narrow portrait videos, and the STOP button sits in the panel.
PANEL_W = 330
PANEL_MIN_H = 660
PANEL_BG = (30, 30, 45)
C_GREEN, C_RED, C_ORANGE = (80, 220, 80), (70, 70, 255), (0, 165, 255)
C_WHITE, C_GREY, C_CYAN = (255, 255, 255), (170, 170, 170), (255, 255, 0)
LEVEL_COLOR = {"ok": C_GREEN, "warn": C_ORANGE, "error": C_RED}
STATE_NAME = {"DOWN": "down", "RISING": "rising", "UP": "top", "LOWERING": "lowering"}
STATE_HEX = {"DOWN": "#9ecae1", "RISING": "#fdd835", "UP": "#66bb6a", "LOWERING": "#ffa726"}   # same palette as the squat


def hex_bgr(h):
    return tuple(int(h[i:i + 2], 16) for i in (5, 3, 1))


STATE_BGR = {k: hex_bgr(v) for k, v in STATE_HEX.items()}

# what the live message says for each error (priority order = which one is shown first)
LIVE_PRIORITY = ["ELBOW_BENT", "TOO_HIGH", "FAST_LIFT", "FAST_LOWER", "TORSO_LEAN", "TORSO_SWING", "ELBOW_STRAIGHT"]
LIVE_MSG = {"ELBOW_BENT": "Keep Your Elbows Straight", "TOO_HIGH": "Do Not Raise Above Shoulder Height",
            "FAST_LIFT": "Too Fast - Control The Lift",
            "FAST_LOWER": "Too Fast - Control The Lowering", "TORSO_SWING": "Keep Your Body Still", "TORSO_LEAN": "Do Not Lean Sideways - Stay Upright",
            "ELBOW_STRAIGHT": "Elbow Straight: Bend the Elbow Slightly"}


def put_text_shadow(img, text, pos, scale, color, thickness=1, shadow=True):
    x, y = pos
    if shadow:                                   # (not on the dark panel, where it just smudges the text)
        cv2.putText(img, text, (x + 1, y + 1), FONT, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def put_plain(img, text, pos, scale, color, thickness=1):
    cv2.putText(img, text, pos, FONT, scale, color, thickness, cv2.LINE_AA)


def wrap_lines(text, max_w, scale, thickness=1):
    lines, cur = [], ""
    for word in text.split():
        trial = (cur + " " + word).strip()
        if not cur or cv2.getTextSize(trial, FONT, scale, thickness)[0][0] <= max_w:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def put_wrapped(img, text, x, y, max_w, scale, color, line_h, thickness=1, max_lines=3, shadow=True):
    """Word-wrapped text; returns the y just below the last line."""
    lines = wrap_lines(text, max_w, scale, thickness)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] += " ..."
    for i, ln in enumerate(lines):
        put_text_shadow(img, ln, (x, y + i * line_h), scale, color, thickness, shadow)
    return y + len(lines) * line_h


def video_label(img, text, anchor, side, scale=0.55):
    """Small angle label next to a joint (side=+1 to its right, -1 to its left) on a darkened box."""
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 1)
    x = anchor[0] + 12 if side > 0 else anchor[0] - 12 - tw
    x = int(np.clip(x, 4, max(4, img.shape[1] - tw - 4)))
    y = int(np.clip(anchor[1] + th // 2, th + 4, img.shape[0] - 4))
    roi = img[max(y - th - 3, 0):y + 5, max(x - 3, 0):x + tw + 3]
    roi[:] = (roi * 0.35).astype(np.uint8)
    cv2.putText(img, text, (x, y), FONT, scale, C_WHITE, 1, cv2.LINE_AA)


def arm_panel_info(a, counter, vel):
    """What the panel shows for one arm. a = (shoulder_ang, elbow_ang, landmarks) or None."""
    return {"state": counter.state, "reps": counter.reps, "attempts": counter.attempts, "last": counter.last_result,
            "sh": None if not a else a[0], "el": None if not a else a[1], "vel": None if not a else vel}


def draw_panel(canvas, x0, info):
    x, avail = x0 + 16, PANEL_W - 32
    put_wrapped(canvas, info["feedback"] or " ", x, 40, avail, 0.75, LEVEL_COLOR[info["level"]], 30, 2, 3, shadow=False)
    cv2.line(canvas, (x0 + 10, 118), (x0 + PANEL_W - 10, 118), (80, 80, 100), 1)

    for i, side in enumerate(("LEFT", "RIGHT")):                       # one column per arm
        d, cx = info["arms"][side], x + i * 160
        put_plain(canvas, f"{side} ARM", (cx, 146), 0.6, C_CYAN)
        put_plain(canvas, STATE_NAME[d["state"]].upper(), (cx, 174), 0.6, STATE_BGR[d["state"]])
        put_plain(canvas, f"Reps {d['reps']} (of {d['attempts']})", (cx, 200), 0.6, C_WHITE)

    put_plain(canvas, "LAST REP RESULT", (x, 236), 0.5, C_GREY)
    y = 262
    for side in ("LEFT", "RIGHT"):
        res, name = info["arms"][side]["last"], side.capitalize()
        if res == "-":
            put_plain(canvas, f"{name}: -", (x, y), 0.6, C_GREY)
            y += 26
        elif res == "OK":
            put_plain(canvas, f"{name}: OK - no errors", (x, y), 0.6, C_GREEN, 2)
            y += 26
        else:                                                           # errors are shown in RED
            y = put_wrapped(canvas, f"{name}: " + res.replace(";", ", "), x, y, avail, 0.52, C_RED, 21, 1, 3,
                            shadow=False) + 6
    cv2.line(canvas, (x0 + 10, 384), (x0 + PANEL_W - 10, 384), (80, 80, 100), 1)

    fmt = lambda v: "--" if v is None else f"{v:.0f}"
    put_plain(canvas, "ANGLES (deg)", (x, 408), 0.5, C_GREY)
    for i, (label, key, side) in enumerate((("L shoulder", "sh", "LEFT"), ("R shoulder", "sh", "RIGHT"),
                                            ("L elbow", "el", "LEFT"), ("R elbow", "el", "RIGHT"))):
        put_plain(canvas, f"{label} {fmt(info['arms'][side][key])}", (x + (i % 2) * 160, 436 + (i // 2) * 28), 0.62, C_WHITE)

    sgn = lambda v: "--" if v is None else f"{int(round(v)):+d}"
    put_plain(canvas, "SPEED (deg/s)", (x, 494), 0.5, C_GREY)
    put_plain(canvas, f"L arm {sgn(info['arms']['LEFT']['vel'])}", (x, 520), 0.6, C_WHITE)
    put_plain(canvas, f"R arm {sgn(info['arms']['RIGHT']['vel'])}", (x + 160, 520), 0.6, C_WHITE)

    tl = "--" if info["torso_tilt"] is None else f"{info['torso_tilt']:+.0f}"
    put_plain(canvas, f"Torso tilt {tl}", (x, 546), 0.6, C_RED if info["torso_bad"] else C_WHITE)
    put_plain(canvas, f"sway {sgn(info['torso_vel'])}", (x + 160, 546), 0.6, C_WHITE)

    # sd = info["sym_diff"]
    # if sd is None:
    #     put_plain(canvas, "Arm diff: --", (x, 576), 0.6, C_WHITE)
    # else:
    #     put_plain(canvas, f"Arm diff: {sd:.0f} deg  (limit {SYM_MAX_DIFF})", (x, 576), 0.6,
    #               C_RED if sd > SYM_MAX_DIFF else C_WHITE)


def make_canvas(disp, info):
    dh, dw = disp.shape[:2]
    canvas = np.full((max(dh, PANEL_MIN_H), dw + PANEL_W, 3), PANEL_BG, np.uint8)
    canvas[:dh, :dw] = disp
    draw_panel(canvas, dw, info)
    return canvas

# ----------------------------------------------------------------- session graph (saved after each video ends)
OK_BG, BAD_BG = "#d9f2d9", "#fcd9d9"                      # correct / flagged rep shading


def _col(rows, key):
    """One column of the per-frame rows as floats; '' / missing (arm not visible) -> NaN = gap in the line."""
    return np.array([float(r[key]) if r.get(key, "") != "" else np.nan for r in rows])


def _hline(ax, y, label, color, ls="--"):
    ax.axhline(y, color=color, ls=ls, lw=1.0, zorder=3)
    ax.text(1.005, y, label, transform=ax.get_yaxis_transform(), va="center", fontsize=7.5, color=color)


def make_session_plot(frames, rep_rows, counters, stem, out_dir, show_plots=False):
    """Shoulder angle (the primary tracking angle) over the whole video, with the rep boundaries.
    One panel per arm that did reps (normally just the arm doing the exercise).  Saves <stem>_shoulder.png."""
    import matplotlib
    if not show_plots:
        matplotlib.use("Agg")                              # headless: no window, just save the PNG
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    if len(frames) < 2:
        return
    t = _col(frames, "time_s")
    ang = {s: _col(frames, f"{s[0].lower()}_shoulder") for s in ("LEFT", "RIGHT")}
    arms = [s for s in ("LEFT", "RIGHT") if counters[s].attempts > 0]
    if not arms:                                           # no rep logged: plot the arm that moved the most
        arms = [max(ang, key=lambda s: float(np.nanmax(ang[s])) if np.isfinite(ang[s]).any() else -1.0)]

    n = len(arms)
    fig, axes = plt.subplots(n, 1, figsize=(13, 4.8 * n), sharex=True, squeeze=False)
    fig.subplots_adjust(left=0.07, right=0.86, top=0.88 if n == 1 else 0.92, bottom=0.2 if n == 1 else 0.12, hspace=0.55)

    for ax, side in zip(axes[:, 0], arms):
        k = side[0].lower()
        y = ang[side]
        states = [r[f"{k}_state"] for r in frames]
        reps = [r for r in rep_rows if r["arm"] == side.lower()]

        for r in reps:                                     # rep shading, boundaries, labels, error codes
            t0, t1 = r["start_s"], r["end_s"]
            mid = (t0 + t1) / 2
            ax.axvspan(t0, t1, color=BAD_BG if r["label"] == "incorrect" else OK_BG, zorder=0)
            for xb in (t0, t1):
                ax.axvline(xb, color="k", ls=":", lw=0.9, zorder=1)
            ax.text(mid, 1.01, f"Rep {r['rep_id']}", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=8, fontweight="bold")
            if r["errors"]:
                ax.text(mid, 0.97, str(r["errors"]).replace(";", "\n"), transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=6.5, color="#b00020", zorder=6)
        start = 0                                          # thin strip along the bottom: down / rising / top / lowering
        for i in range(1, len(states) + 1):
            if i == len(states) or states[i] != states[start]:
                ax.axvspan(t[start], t[min(i, len(t) - 1)], ymin=0, ymax=0.035,
                           color=STATE_HEX.get(states[start], "#cccccc"), lw=0, zorder=2)
                start = i

        ax.plot(t, y, color="#1f77b4", lw=1.8, zorder=4)
        ax.plot(t, np.where(y > TOP_MAX, y, np.nan), color="#d62728", lw=3.2, zorder=5)    # over the height limit
        _hline(ax, TOP_MIN, f"target ({TOP_MIN}°)", "#2e7d32")
        _hline(ax, TOP_MAX, f"max height ({TOP_MAX}°)", "#d62728")
        _hline(ax, REST_ENTER, f"rest ({REST_ENTER}°)", "grey", ls=":")
        for r in reps:                                     # dot + value at each rep's peak
            idx = np.where((t >= r["start_s"]) & (t <= r["end_s"]) & ~np.isnan(y))[0]
            if len(idx):
                j = idx[np.argmax(y[idx])]
                ax.plot(t[j], y[j], "o", color="#1f77b4", ms=5, zorder=5)
                ax.annotate(f"{y[j]:.0f}°", (t[j], y[j]), textcoords="offset points", xytext=(0, 6),
                            ha="center", fontsize=7.5, zorder=6)
        top = float(np.nanmax(y)) if np.isfinite(y).any() else TOP_MAX
        ax.set_ylim(0, max(top, TOP_MAX) + 30)
        ax.set_xlim(t[0], t[-1])
        ax.set_ylabel("Shoulder angle (deg)")
        ax.set_title(f"{stem} - {side.title()} arm: shoulder angle (primary tracking angle)", fontsize=12, pad=26)
        ax.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("Time (s)")

    handles = [Line2D([0], [0], color="#1f77b4", lw=1.8, label="Shoulder angle"),
               Line2D([0], [0], color="#d62728", lw=3.2, label="Over max height"),
               Patch(color=OK_BG, label="Rep OK"), Patch(color=BAD_BG, label="Rep flagged"),
               Line2D([0], [0], color="k", ls=":", label="Rep boundary")]
    handles += [Patch(color=STATE_HEX[s], label=STATE_NAME[s]) for s in STATE_HEX]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=7.5, frameon=False)
    path = os.path.join(out_dir, f"{stem}_shoulder.png")
    fig.savefig(path, dpi=160)
    print("saved graph:", path)

    if show_plots:
        plt.show()                                         # blocks until the graph window is closed
    else:
        plt.close("all")

# ----------------------------------------------------------------- main processing
def process_video(video_path, show=True, out_dir="outputs_raise", show_plots=False):
    global stop_flag
    stop_flag = False

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    video_name = os.path.basename(video_path)
    stem = os.path.splitext(video_name)[0]
    os.makedirs(out_dir, exist_ok=True)

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.6,
        min_pose_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    if show:
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, on_mouse, stop_button_rect)

    counters = {"LEFT": RepCounter("LEFT"), "RIGHT": RepCounter("RIGHT")}
    smooth = {"LEFT": None, "RIGHT": None}
    vtr = {"LEFT": Velocity(VEL_WINDOW_S), "RIGHT": Velocity(VEL_WINDOW_S)}   # shoulder-angle velocity per arm
    tv, tilt_prev = Velocity(VEL_WINDOW_S), None                               # torso-tilt velocity
    frame_rows = []
    frame_idx = 0

    # NO cv2.flip here: flipping before pose estimation swaps MediaPipe's left/right labels,
    # which would make "left arm" in your CSV mean the person's right arm.
    with vision.PoseLandmarker.create_from_options(options) as detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            this_frame = frame_idx
            t_s = this_frame / fps
            ts_ms = int(this_frame * 1000 / fps)
            frame_idx += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = detector.detect_for_video(mp_image, ts_ms)

            disp = fit(frame)
            dh, dw = disp.shape[:2]
            arm = {"LEFT": None, "RIGHT": None}       # smoothed (shoulder_ang, elbow_ang, landmarks)
            arm_vel = {"LEFT": 0.0, "RIGHT": 0.0}
            tilt, tilt_vel = None, 0.0

            if results.pose_landmarks:
                lm = results.pose_landmarks[0]
                raw_tilt = torso_tilt(lm, w, h)
                if raw_tilt is None:
                    tv.reset()
                    tilt_prev = None
                else:
                    tilt = raw_tilt if tilt_prev is None else ALPHA * tilt_prev + (1 - ALPHA) * raw_tilt
                    tilt_prev = tilt
                    tilt_vel = tv.update(t_s, tilt)
                for side in ("LEFT", "RIGHT"):
                    got = arm_angles(lm, side, w, h)
                    if got is None:
                        vtr[side].reset()
                        continue
                    raw = got[:2]
                    if smooth[side] is None:
                        smooth[side] = raw
                    smooth[side] = tuple(ALPHA * p + (1 - ALPHA) * r for p, r in zip(smooth[side], raw))
                    arm[side] = (smooth[side][0], smooth[side][1], got[2])
                    arm_vel[side] = vtr[side].update(t_s, smooth[side][0])

            else:
                tv.reset()
                tilt_prev = None
                for v in vtr.values():
                    v.reset()

            # ---- symmetry (only when both arms are off rest, so a one-arm exercise isn't flagged)
            # a_l, a_r = arm["LEFT"], arm["RIGHT"]
            # sym_diff, sym_ok, sym_bad = None, True, False
            # if a_l and a_r:
            #     sym_diff = abs(a_l[0] - a_r[0])
            #     sym_ok = sym_diff < SYM_MAX_DIFF
            #     sym_bad = (not sym_ok) and min(a_l[0], a_r[0]) > REST_ENTER

            torso_swing = abs(tilt_vel) > TORSO_SWING_VEL
            fired = set()                                 # error codes firing on THIS frame
            arm_errs = {"LEFT": [], "RIGHT": []}

            # ---- rep counters + logging
            row = {"frame": this_frame, "time_s": round(t_s, 3), "detected": int(bool(results.pose_landmarks)),
                   "torso_tilt": "" if tilt is None else round(tilt, 1), "torso_vel": round(tilt_vel, 1)}
            for side, key in (("LEFT", "l"), ("RIGHT", "r")):
                a = arm[side]
                c = counters[side]
                errs = (c.update(a[0], a[1], t_s, video_name, vel=arm_vel[side],
                                 torso_swing=torso_swing, torso_vel=tilt_vel, tilt=tilt) if a else c.break_streaks())
                fired.update(errs)
                arm_errs[side] = errs
                row[f"{key}_vel"] = "" if not a else round(arm_vel[side], 1)
                row[f"{key}_shoulder"] = "" if not a else round(a[0], 1)
                row[f"{key}_elbow"] = "" if not a else round(a[1], 1)
                row[f"{key}_state"] = c.state
                row[f"{key}_errors"] = ";".join(errs)
            frame_rows.append(row)

            # ---- live feedback.  Same idea as the squat: an error confirmed during a rep stays on screen, in red,
            # ---- until that arm's rep ends; an error that is only firing right now is shown in orange.
            confirmed = {}
            for side, c in counters.items():
                if c.state == "DOWN":
                    continue
                for code, n in c.err_max.items():
                    if n >= (VEL_ERR_MIN_FRAMES if code in VEL_MSG else ERR_MIN_FRAMES):
                        confirmed.setdefault(code, []).append(side)
            states = {c.state for c in counters.values()}
            if not results.pose_landmarks:
                feedback, level = "No Person Detected", "error"
            elif not (arm["LEFT"] or arm["RIGHT"]):
                feedback, level = "Arms Not Visible - Adjust Position", "error"
            elif confirmed:
                code = next(c for c in LIVE_PRIORITY if c in confirmed)
                who = confirmed[code]
                feedback = ("" if len(who) == 2 else who[0].capitalize() + " arm: ") + LIVE_MSG[code]
                if len(confirmed) > 1:
                    feedback += f"  (+{len(confirmed) - 1} more)"
                level = "error"
            else:
                firing = [c for c in LIVE_PRIORITY if c in fired]
                too_low = [s_ for s_, c in counters.items() if c.state == "RISING" and arm_vel[s_] < -20
                           and PARTIAL_MIN_PEAK <= c.peak < TOP_MIN]          # coming back down without reaching the top
                if firing:
                    feedback, level = LIVE_MSG[firing[0]], "warn"
                elif too_low:
                    feedback, level = "Raise Your Arms Higher", "error"
                elif "UP" in states:
                    feedback, level = "Good - Hold At Shoulder Height", "ok"
                elif "LOWERING" in states:
                    feedback, level = "Good - Lower With Control", "ok"
                elif "RISING" in states:
                    feedback, level = "Good - Keep Lifting", "ok"
                else:
                    feedback, level = "Arms Down - Start The Raise", "ok"

            # ---- drawing: skeleton + angle labels on the video, everything else in the side panel
            for side in ("LEFT", "RIGHT"):
                a = arm[side]
                if not a:
                    continue
                pts = [(int(p.x * dw), int(p.y * dh)) for p in a[2]]      # hip, shoulder, elbow, wrist
                cv2.polylines(disp, [np.array(pts, np.int32)], False, C_RED if arm_errs[side] else (255, 200, 0), 3)
                for p in pts:
                    cv2.circle(disp, p, 6, (0, 255, 255), -1)
                other_sh = lm[ARM_IDX["RIGHT" if side == "LEFT" else "LEFT"][1]]
                out = 1 if a[2][1].x >= other_sh.x else -1                 # +1 = this arm is on the image's right
                video_label(disp, f"S {a[0]:.0f}", (pts[1][0], pts[1][1] - 28), out)   # shoulder angle, above the shoulder
                video_label(disp, f"E {a[1]:.0f}", (pts[2][0], pts[2][1] + 28), out)   # elbow angle, below the elbow
            if tilt is not None:
                sm = (int((lm[L_SHOULDER].x + lm[R_SHOULDER].x) / 2 * dw), int((lm[L_SHOULDER].y + lm[R_SHOULDER].y) / 2 * dh))
                hm = (int((lm[L_HIP].x + lm[R_HIP].x) / 2 * dw), int((lm[L_HIP].y + lm[R_HIP].y) / 2 * dh))
                cv2.line(disp, hm, (hm[0], max(sm[1] - 20, 0)), C_GREY, 1, cv2.LINE_AA)    # vertical reference
                cv2.line(disp, hm, sm, (0, 0, 255) if "TORSO_LEAN" in fired else (255, 0, 255), 2, cv2.LINE_AA)   # magenta, red when leaning too far
                video_label(disp, f"T {tilt:+.0f}", (hm[0], hm[1] + 26), 1)
            # state of each arm, top-left corner of the video (same colours as the squat's state label)
            rows = [(f"{s_.capitalize()} arm: {STATE_NAME[counters[s_].state]}", STATE_BGR[counters[s_].state])
                    for s_ in ("LEFT", "RIGHT")]
            sizes = [cv2.getTextSize(t, FONT, 0.7, 2)[0] for t, _ in rows]
            bw, th = max(sz[0] for sz in sizes), sizes[0][1]
            roi = disp[8:8 + 2 * (th + 14) + 12, 8:bw + 28]
            roi[:] = (roi * 0.35).astype(np.uint8)                          # darken a box so it reads on any background
            for i, (txt, colr) in enumerate(rows):
                cv2.putText(disp, txt, (16, 16 + th + i * (th + 14)), FONT, 0.7, colr, 2, cv2.LINE_AA)

            info = {"feedback": feedback, "level": level,
                    "arms": {s_: arm_panel_info(arm[s_], counters[s_], arm_vel[s_]) for s_ in ("LEFT", "RIGHT")},
                    "torso_vel": tilt_vel if tilt is not None else None, "torso_tilt": tilt, "torso_bad": "TORSO_LEAN" in fired}
            if present(make_canvas(disp, info), show):
                break

    cap.release()
    rep_rows = sorted(counters["LEFT"].records + counters["RIGHT"].records,
                      key=lambda r: (r["start_s"], r["arm"]))
    write_csv(os.path.join(out_dir, f"{stem}_frames.csv"), frame_rows, FRAME_FIELDS)
    write_csv(os.path.join(out_dir, f"{stem}_reps.csv"), rep_rows, REP_FIELDS)
    L, R = counters["LEFT"], counters["RIGHT"]
    print(f"{video_name}: left {L.reps}/{L.attempts} correct, right {R.reps}/{R.attempts} correct")
    try:
        make_session_plot(frame_rows, rep_rows, counters, stem, out_dir, show_plots)
    except Exception as e:                                   # a plotting problem must never lose the CSVs
        print(f"Could not make graph: {e}")
    return {"left": L, "right": R, "rep_rows": rep_rows}


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=VIDEO_PATH, help="video file or folder of videos")
    ap.add_argument("--out", default="outputs_raise")
    ap.add_argument("--no-display", action="store_true", help="headless batch run")
    args = ap.parse_args()
    show = not args.no_display

    download_model()

    if os.path.isdir(args.input):
        videos = sorted({p for ext in ("mp4", "mov", "avi", "mkv")
                         for p in glob.glob(os.path.join(args.input, "**", f"*.{ext}"), recursive=True)})
    else:
        videos = [args.input]
    if not videos:
        raise SystemExit("No videos found.")

    all_rows, last = [], None
    for v in videos:
        last = process_video(v, show=show, out_dir=args.out, show_plots=show and len(videos) == 1)
        all_rows += last["rep_rows"]
        if stop_flag:
            break

    os.makedirs(args.out, exist_ok=True)
    write_csv(os.path.join(args.out, "all_reps_predicted.csv"), all_rows, REP_FIELDS)
    cv2.destroyAllWindows()
    if show and len(videos) == 1 and last:
        show_summary(last["left"], last["right"])


if __name__ == "__main__":
    main()