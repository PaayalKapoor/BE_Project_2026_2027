"""
PowerPulse - Squat Analyzer (fixed version + logging for trainer comparison)

Usage:
  python powerpulse_squat_fixed.py --input path/to/video.mp4
  python powerpulse_squat_fixed.py --input path/to/folder --no-display      # batch, headless

Outputs (in --out folder, default ./outputs):
  <video>_frames.csv        per-frame angles / state / feedback (for debugging)
  <video>_reps.csv          one row per rep with predicted label + error codes
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
VIDEO_PATH = r"C:\Users\ADMIN\Documents\GitHub\BE_Project_2026_2027\videos\Phase_1\Squats\Participant_107_SQ_L.mp4"

# Tasks API has no PoseLandmark enum (mp_pose / options.PoseLandmark don't exist) -> use indices
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_HEEL, RIGHT_HEEL = 29, 30
LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX = 31, 32
BOTTOM_KNEE = 100        # knee angle below this = bottom phase (S3); a rep that never gets below it is SHALLOW
DEPTH_TARGET_KNEE = 80   # lowest knee angle must reach this; above it (up to 100) -> SQUAT_DEEPER
STAND_KNEE = 165        # knee angle above this = standing
PARTIAL_MIN_KNEE = 130  # a rep that never goes below this is ignored (just a wobble)
ALPHA = 0.3             # smoothing weight of previous value
VIS_MIN = 0.6
ERR_MIN_FRAMES = 5      # an error must persist this many CONSECUTIVE frames to flag the rep (1 = old behaviour)

# ---- momentum / swing detection.  The four numbers below are PLACEHOLDERS: tune them on your own
# ---- videos (the m_* columns in the reps CSV are there for exactly that - see compare_labels.py)
VEL_ERR_MIN_FRAMES = 3  # velocity-based errors need fewer consecutive frames: the velocity is already averaged over VEL_WINDOW_S
VEL_WINDOW_S = 0.2      # velocities are measured over this many seconds (less noise, independent of fps)
FAST_DESCENT_VEL = 200  # deg/s: knee angle closing faster than this            -> FAST_DESCENT
TORSO_SWING_VEL = 90    # deg/s: back-lean angle changing faster than this      -> TORSO_SWING
BOUNCE_VEL = 60         # deg/s: still this fast going INTO the bottom and back OUT -> BOUNCE
BOUNCE_WIN_S = 0.2      # window either side of the bottom used for the bounce check
VEL_MSG = {"FAST_DESCENT": "Too Fast - Control The Descent",
           "TORSO_SWING": "Avoid Swinging - Stay Controlled"}

# ---- heel lift: heel rising above its flat-foot position for ERR_MIN_FRAMES consecutive frames
HEEL_LIFT_RATIO = 0.10   # rise as a fraction of shank (knee-ankle) length -> HEEL_LIFT   (PLACEHOLDER - tune it)
HEEL_BASE_FRAMES = 30    # standing frames used to find where the flat heel sits (~1 s at 30 fps)
HEEL_BASE_MIN_FRAMES = 5 # need at least this many standing frames, otherwise fall back (see heel block)
ALERT_MSG = {**VEL_MSG, "HEEL_LIFT": "Keep Your Heels Down"}
LIVE_PRIORITY = ["HEEL_LIFT", "LEAN_FORWARD", "FAST_DESCENT", "TORSO_SWING"]
LIVE_MSG = {"HEEL_LIFT": "Keep Your Heels Down", "LEAN_FORWARD": "Lean Less Forward",
            "FAST_DESCENT": "Too Fast - Control The Descent", "TORSO_SWING": "Avoid Swinging - Stay Controlled"}

# ---- pose-rule thresholds (S3 / bottom phase).  Angles are camera-independent; the knee-over-toe distance is
# ---- divided by shank length (knee-ankle, px), so it no longer depends on how far the camera is.
LEAN_FORWARD_BACK = 45     # deg: |back lean| above this           -> LEAN_FORWARD
TOO_DEEP_KNEE = 70         # deg: knee angle below this            -> TOO_DEEP
KNEE_TOE_MAX = 0.25        # (knee x - toe x) / shank length, toward the facing direction -> KNEES_FORWARD (PLACEHOLDER - tune with m_knee_toe_max)
BUTT_WIN_S = 0.3           # s: window ending at the bottom of the squat, used to measure the "butt wink" proxy
BUTT_WINK_SHIFT = 8.0      # deg: |change in trunk lean over that window| above this -> BUTT_WINK (PLACEHOLDER - tune with m_bottom_lean_shift)
HIP_FLEXION_MIN, ANKLE_MOBILITY_MIN = 50, 70       # legacy rules, OFF by default (see ENABLED_ERRORS)
HIPS_BACK_MAX, WEIGHT_BACK_MAX = 90, 100           # legacy rules, OFF by default

# KNEES_FORWARD is OFF for now (trainer never flagged it); add "KNEES_FORWARD" here to turn it back on.
# Only these codes can be reported.  HIP_FLEXION / ANKLE_MOBILITY duplicate TOO_DEEP / KNEES_FORWARD, and
# HIPS_BACK / WEIGHT_BACK contradict KNEES_FORWARD.  Add a code back here to re-enable it.
# TOO_DEEP is OFF (depth only matters when the spine rounds).  BUTT_WINK is implemented but OFF until you have
# checked m_bottom_lean_shift against the trainer notes and tuned BUTT_WINK_SHIFT - then add "BUTT_WINK" here.
ENABLED_ERRORS = {"LEAN_FORWARD", "FAST_DESCENT", "TORSO_SWING",
                  "BOUNCE", "HEEL_LIFT", "SHALLOW", "SQUAT_DEEPER"}
# Flagged and written to the CSV, but does NOT turn a rep "incorrect" (e.g. {"KNEES_FORWARD"} while tuning).
ADVISORY_ERRORS = set()

window_name = "PowerPulse - Squat Analyzer"
summary_window_name = "Squat Summary"
stop_flag = False
stop_button_rect = [0, 0, 0, 0]
summary_close_flag = False
summary_close_rect = (200, 130, 310, 175)

REP_FIELDS = ["video", "rep_id", "start_s", "end_s", "label", "errors", "accuracy_pct", "reached_depth",
              "m_peak_descent_vel", "m_peak_ascent_vel", "m_peak_back_vel", "m_bounce_in", "m_bounce_out",
              "m_descent_s", "m_ascent_s", "m_heel_lift_max", "m_knee_toe_max",
              "m_bottom_lean_shift", "m_bottom_hip_shift"]          # m_* = metrics for tuning thresholds
FRAME_FIELDS = ["frame", "time_s", "detected", "state", "feedback", "errors", "knee", "hip", "back", "ankle",
                "knee_vel", "back_vel", "heel_rise"]


# ----------------------------------------------------------------- model
def download_model():
    if os.path.exists(MODEL_PATH):
        print(f"Model already present: {MODEL_PATH}")
        return
    print(f"Downloading model -> {MODEL_PATH}")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.")


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
    text = "STOP"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
    cv2.putText(frame, text, (x1 + (x2 - x1 - tw) // 2, y1 + (y2 - y1 + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)


def fit(frame, max_w=900, max_h=650):
    """Aspect-ratio-preserving resize (the old fixed 1200x600 resize distorted the video)."""
    h, w = frame.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        return cv2.resize(frame, (int(w * scale), int(h * scale)))
    return frame.copy()


def present(disp, show):
    """Show a frame, handle keys. Returns True if the user wants to quit."""
    if not show:
        return False
    stop_button_rect[:] = get_stop_button_rect(disp)   # update IN PLACE so the mouse callback sees it
    draw_stop_button(disp)
    cv2.imshow(window_name, disp)
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


def make_summary_image(reps, avg_accuracy):
    img = np.zeros((220, 340, 3), np.uint8)
    img[:] = (40, 40, 40)
    cv2.putText(img, "SUMMARY", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(img, f"Correct reps: {reps}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, f"Accuracy: {avg_accuracy:.1f}%", (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    rounded_rect(img, summary_close_rect, (0, 0, 255), radius=12)
    cv2.putText(img, "Close", (summary_close_rect[0] + 25, summary_close_rect[1] + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


# def show_summary(reps, avg_accuracy):
#     global summary_close_flag
#     summary_close_flag = False
#     cv2.namedWindow(summary_window_name, cv2.WINDOW_NORMAL)
#     cv2.resizeWindow(summary_window_name, 340, 220)
#     cv2.setMouseCallback(summary_window_name, on_summary_mouse, summary_close_rect)
#     while True:
#         cv2.imshow(summary_window_name, make_summary_image(reps, avg_accuracy))
#         if (cv2.waitKey(20) & 0xFF) == 27 or summary_close_flag:
#             break
#     cv2.destroyWindow(summary_window_name)


# ----------------------------------------------------------------- side panel + on-video labels
# All text lives in a panel BESIDE the video (never on top of the person), and long text is word-wrapped,
# so nothing gets clipped on narrow portrait videos.  The STOP button is drawn in the panel too.
PANEL_W = 330
PANEL_MIN_H = 640
PANEL_BG = (30, 30, 45)
FONT = cv2.FONT_HERSHEY_SIMPLEX
C_GREEN, C_RED, C_ORANGE = (80, 220, 80), (70, 70, 255), (0, 165, 255)
C_WHITE, C_GREY, C_CYAN = (255, 255, 255), (170, 170, 170), (255, 255, 0)
LEVEL_COLOR = {"ok": C_GREEN, "warn": C_ORANGE, "error": C_RED}


def put_text_shadow(img, text, pos, scale, color, thickness=1, shadow=True):
    x, y = pos
    if shadow:                                   # (not on the dark panel, where it just smudges the text)
        cv2.putText(img, text, (x + 1, y + 1), FONT, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


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


def put_plain(img, text, pos, scale, color, thickness=1):
    cv2.putText(img, text, pos, FONT, scale, color, thickness, cv2.LINE_AA)


def draw_panel(canvas, x0, info):
    x, avail = x0 + 16, PANEL_W - 32
    put_wrapped(canvas, info["feedback"] or " ", x, 40, avail, 0.75, LEVEL_COLOR[info["level"]], 30, 2, 3, shadow=False)
    cv2.line(canvas, (x0 + 10, 118), (x0 + PANEL_W - 10, 118), (80, 80, 100), 1)

    put_plain(canvas, f"State: {info['state']}", (x, 146), 0.6, C_CYAN)
    put_plain(canvas, f"Correct reps: {info['reps']}  (of {info['attempts']})", (x, 174), 0.65, C_WHITE)

    

    put_plain(canvas, "LAST REP RESULT", (x, 258), 0.5, C_GREY)
    res = info["last_result"]
    if res == "-":
        put_plain(canvas, "-", (x, 284), 0.65, C_GREY)
    elif res == "OK":
        put_plain(canvas, "OK - no errors", (x, 284), 0.65, C_GREEN, 2)
    else:                                                    # errors are shown in RED
        put_wrapped(canvas, res.replace(";", ", "), x, 284, avail, 0.62, C_RED, 26, 2, 3, shadow=False)
    cv2.line(canvas, (x0 + 10, 352), (x0 + PANEL_W - 10, 352), (80, 80, 100), 1)

    put_plain(canvas, "ANGLES (deg)", (x, 376), 0.5, C_GREY)
    a = info["angles"]
    vals = ["--"] * 4 if a is None else [f"{v:.0f}" for v in a]
    for i, (name, v) in enumerate(zip(("Knee", "Hip", "Back", "Ankle"), vals)):
        put_plain(canvas, f"{name} {v}", (x + (i % 2) * 160, 404 + (i // 2) * 28), 0.65, C_WHITE)

    put_plain(canvas, "SPEED (deg/s)", (x, 462), 0.5, C_GREY)
    kv, bv = ("--", "--") if info["vels"] is None else (f"{int(round(info['vels'][0])):+d}", f"{int(round(info['vels'][1])):+d}")
    put_plain(canvas, f"Knee {kv}", (x, 490), 0.6, C_WHITE)
    put_plain(canvas, f"Back {bv}", (x + 160, 490), 0.6, C_WHITE)

    hr = info["heel_rise"]
    if hr is None:
        put_plain(canvas, "Heel rise: --", (x, 522), 0.6, C_WHITE)
    else:
        put_plain(canvas, f"Heel rise: {hr:.2f}  (limit {HEEL_LIFT_RATIO:.2f})", (x, 522), 0.6,
                        C_RED if hr > HEEL_LIFT_RATIO else C_WHITE)


def make_canvas(disp, info):
    dh, dw = disp.shape[:2]
    canvas = np.full((max(dh, PANEL_MIN_H), dw + PANEL_W, 3), PANEL_BG, np.uint8)
    canvas[:dh, :dw] = disp
    draw_panel(canvas, dw, info)
    return canvas


# ----------------------------------------------------------------- geometry
def calculate_jointangles(x1, y1, x2, y2, x3, y3):
    """Angle at point 2 (degrees) between 2->1 and 2->3. Returns NaN if points coincide."""
    v1 = np.array([x1 - x2, y1 - y2], dtype=float)
    v2 = np.array([x3 - x2, y3 - y2], dtype=float)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom < 1e-6:
        return float("nan")
    cosine = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def calculate_backlean(shoulder_x, shoulder_y, hip_x, hip_y):
    """Angle of hip->shoulder vector from vertical; sign depends on lean direction."""
    dx, dy = shoulder_x - hip_x, shoulder_y - hip_y
    lean = float(np.degrees(np.arctan2(abs(dx), -dy)))   # image y points down, so up = -dy
    return lean if shoulder_x < hip_x else -lean


def is_side_view(lm):
    width = abs(lm[LEFT_SHOULDER].x - lm[RIGHT_SHOULDER].x)
    if width > 0.25:
        return False, "Please Turn Sideways to the Camera"
    return True, "Good Position"


def s3_errors(back, knee, hip, ankle, knee_toe):
    """All bottom-phase errors firing this frame, in priority order: (code, message).
    knee_toe = (knee x - toe x) / shank length, positive = knee is PAST the toe in the facing direction."""
    errs = []
    if abs(back) > LEAN_FORWARD_BACK:        errs.append(("LEAN_FORWARD", "Lean Less Forward"))
    if knee < TOO_DEEP_KNEE:                 errs.append(("TOO_DEEP", "Too Deep - Control The Squat"))
    if hip < HIP_FLEXION_MIN:                errs.append(("HIP_FLEXION", "Too Much Hip Flexion"))
    if ankle < ANKLE_MOBILITY_MIN:           errs.append(("ANKLE_MOBILITY", "Limited Mobility - Fix Posture"))
    if knee_toe > KNEE_TOE_MAX:              errs.append(("KNEES_FORWARD", "Knees Too Far Forward"))
    if hip > HIPS_BACK_MAX:                  errs.append(("HIPS_BACK", "Push Hips Back More"))
    if ankle > WEIGHT_BACK_MAX:              errs.append(("WEIGHT_BACK", "Weight Shifting Backward"))
    return [(c, m) for c, m in errs if c in ENABLED_ERRORS]


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


def rep_metrics(series, t_end):
    """series = [(t, knee_angle, knee_vel, back_vel), ...] for one rep -> (metrics dict, bounced?)."""
    ts = [x[0] for x in series]
    knee = [x[1] for x in series]
    kv = [x[2] for x in series]
    bv = [x[3] for x in series]
    ib = int(np.argmin(knee))
    tb = ts[ib]                                                     # time of the bottom of the squat

    def nearest(t_target):
        return min(range(len(ts)), key=lambda i: abs(ts[i] - t_target))

    # One-sided slopes measured straight from the angles either side of the bottom. (The running
    # velocity above is a trailing window, which blurs a sharp reversal exactly at the bottom.)
    i0, i1 = nearest(tb - BOUNCE_WIN_S), nearest(tb + BOUNCE_WIN_S)
    dt_in, dt_out = tb - ts[i0], ts[i1] - tb
    v_in = (knee[ib] - knee[i0]) / dt_in if dt_in > 0.05 else 0.0      # negative = still descending into the bottom
    v_out = (knee[i1] - knee[ib]) / dt_out if dt_out > 0.05 else 0.0   # positive = already driving back up
    # butt-wink proxy: how much the trunk lean (toward the facing direction) and the hip angle change over the last
    # BUTT_WIN_S before the bottom.  MediaPipe has no spine/pelvis points, so this is only an indirect signal.
    fl = [x[4] for x in series]
    ha = [x[5] for x in series]
    ib0 = nearest(tb - BUTT_WIN_S)
    lean_shift = fl[ib] - fl[ib0] if tb - ts[ib0] > 0.1 else 0.0
    hip_shift = ha[ib] - ha[ib0] if tb - ts[ib0] > 0.1 else 0.0
    metrics = {"m_peak_descent_vel": round(max(0.0, -min(kv)), 1),
               "m_peak_ascent_vel": round(max(0.0, max(kv)), 1),
               "m_peak_back_vel": round(max(abs(v) for v in bv), 1),
               "m_bounce_in": round(v_in, 1), "m_bounce_out": round(v_out, 1),
               "m_descent_s": round(tb - ts[0], 2), "m_ascent_s": round(t_end - tb, 2),
               "m_bottom_lean_shift": round(lean_shift, 1), "m_bottom_hip_shift": round(hip_shift, 1)}
    return metrics, (v_in < -BOUNCE_VEL and v_out > BOUNCE_VEL)


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)

# ----------------------------------------------------------------- session graphs (saved after each video ends)
OK_BG, BAD_BG = "#d9f2d9", "#fcd9d9"                      # correct / flagged rep shading
STATE_COLORS = {"S1": "#9ecae1", "S2": "#fdd835", "S3": "#66bb6a", "S4": "#ffa726"}
STATE_NAMES = {"S1": "S1 standing", "S2": "S2 descending", "S3": "S3 bottom", "S4": "S4 rising"}


def _col(rows, key):
    """One column of the per-frame rows as floats; '' / missing (no detection, not in a rep) -> NaN = gap in the line."""
    return np.array([float(r[key]) if r.get(key, "") != "" else np.nan for r in rows])


def _graph_base(plt, t, states, reps, title, ylabel):
    """Figure with rep shading (green OK / red flagged), dotted rep boundaries, 'Rep N' labels, error codes, state strip."""
    fig, ax = plt.subplots(figsize=(13, 4.8))
    fig.subplots_adjust(right=0.86, top=0.86, bottom=0.2)
    for r in reps:
        t0, t1 = r["start_s"], r["end_s"]
        mid = (t0 + t1) / 2
        ax.axvspan(t0, t1, color=BAD_BG if r["label"] == "incorrect" else OK_BG, zorder=0)
        for x in (t0, t1):
            ax.axvline(x, color="k", ls=":", lw=0.9, zorder=1)
        ax.text(mid, 1.01, f"Rep {r['rep_id']}", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8, fontweight="bold")
        if r["errors"]:
            ax.text(mid, 0.97, str(r["errors"]).replace(";", "\n"), transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=6.5, color="#b00020", zorder=6)
    start = 0                                              # thin strip along the bottom: S1..S4
    for i in range(1, len(states) + 1):
        if i == len(states) or states[i] != states[start]:
            ax.axvspan(t[start], t[min(i, len(t) - 1)], ymin=0, ymax=0.035,
                       color=STATE_COLORS.get(states[start], "#cccccc"), lw=0, zorder=2)
            start = i
    ax.set_title(title, fontsize=12, pad=26)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(t[0], t[-1])
    ax.grid(alpha=0.25)
    return fig, ax


def _hline(ax, y, label, color, ls="--"):
    ax.axhline(y, color=color, ls=ls, lw=1.0, zorder=3)
    ax.text(1.005, y, label, transform=ax.get_yaxis_transform(), va="center", fontsize=7.5, color=color)


def _mark(ax, t, y, reps, kind, fmt, color):
    """Dot + value at each rep's lowest (kind='min') or highest (kind='max') point."""
    for r in reps:
        idx = np.where((t >= r["start_s"]) & (t <= r["end_s"]) & ~np.isnan(y))[0]
        if len(idx) == 0:
            continue
        j = idx[np.argmin(y[idx])] if kind == "min" else idx[np.argmax(y[idx])]
        ax.plot(t[j], y[j], "o", color=color, ms=5, zorder=5)
        ax.annotate(fmt.format(y[j]), (t[j], y[j]), textcoords="offset points",
                    xytext=(0, -13 if kind == "min" else 6), ha="center", fontsize=7.5, zorder=6)


def _legend(ax, plt_lines, plt_patch, extra):
    handles = extra + [plt_patch(color=OK_BG, label="Rep OK"), plt_patch(color=BAD_BG, label="Rep flagged"),
                       plt_lines([0], [0], color="k", ls=":", label="Rep boundary")]
    handles += [plt_patch(color=c, label=STATE_NAMES[s]) for s, c in STATE_COLORS.items()]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=len(handles),
              fontsize=7.5, frameon=False)


def make_session_plots(frames, reps, stem, out_dir, show_plots=False):
    """Knee angle (primary tracking angle), heel rise and back lean over the whole session, with rep boundaries.
    Saves <stem>_knee.png / _heel.png / _back.png in out_dir (and opens them if show_plots)."""
    import matplotlib
    if not show_plots:
        matplotlib.use("Agg")                              # headless: no window, just save the PNGs
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    if len(frames) < 2:
        return
    t = _col(frames, "time_s")
    states = [r["state"] for r in frames]
    knee, back, heel = _col(frames, "knee"), np.abs(_col(frames, "back")), _col(frames, "heel_rise")
    leg = lambda ax, extra: _legend(ax, Line2D, Patch, extra)
    top = lambda a, floor: max(float(np.nanmax(a)) if np.isfinite(a).any() else floor, floor)

    # 1) knee angle
    fig, ax = _graph_base(plt, t, states, reps, f"{stem} - Knee angle (primary tracking angle)", "Knee angle (deg)")
    ax.plot(t, knee, color="#1f77b4", lw=1.8, zorder=4)
    _hline(ax, STAND_KNEE, f"standing ({STAND_KNEE}°)", "grey")
    _hline(ax, BOTTOM_KNEE, f"bottom phase ({BOTTOM_KNEE}°)", "#e08a00")
    _hline(ax, DEPTH_TARGET_KNEE, f"depth target ({DEPTH_TARGET_KNEE}°)", "#2e7d32")
    _mark(ax, t, knee, reps, "min", "{:.0f}°", "#1f77b4")
    lo = min(float(np.nanmin(knee)) if np.isfinite(knee).any() else DEPTH_TARGET_KNEE, DEPTH_TARGET_KNEE)
    ax.set_ylim(lo - 18, top(knee, STAND_KNEE) + 28)
    leg(ax, [Line2D([0], [0], color="#1f77b4", lw=1.8, label="Knee angle")])
    fig.savefig(os.path.join(out_dir, f"{stem}_knee.png"), dpi=160)

    # 2) heel lift (red where over the limit)
    fig2, ax = _graph_base(plt, t, states, reps, f"{stem} - Heel lift (heel height above its flat-foot position)",
                           "Heel rise (fraction of shank length)")
    ax.plot(t, heel, color="#2ca02c", lw=1.8, zorder=4)
    ax.plot(t, np.where(heel > HEEL_LIFT_RATIO, heel, np.nan), color="#d62728", lw=3.2, zorder=5)
    _hline(ax, HEEL_LIFT_RATIO, f"HEEL_LIFT limit ({HEEL_LIFT_RATIO:.2f})", "#d62728")
    _hline(ax, 0, "flat foot", "grey", ls=":")
    _mark(ax, t, heel, reps, "max", "{:.2f}", "#2ca02c")
    lo = min(float(np.nanmin(heel)) if np.isfinite(heel).any() else 0.0, 0.0)
    ax.set_ylim(lo - 0.02, top(heel, HEEL_LIFT_RATIO) * 1.45)
    leg(ax, [Line2D([0], [0], color="#2ca02c", lw=1.8, label="Heel rise"),
             Line2D([0], [0], color="#d62728", lw=3.2, label="Over limit")])
    fig2.savefig(os.path.join(out_dir, f"{stem}_heel.png"), dpi=160)

    # 3) back lean (absolute: the sign only says which way the person faces)
    fig3, ax = _graph_base(plt, t, states, reps, f"{stem} - Back lean from vertical", "Back lean (deg, absolute)")
    ax.plot(t, back, color="#9467bd", lw=1.8, zorder=4)
    ax.plot(t, np.where(back > LEAN_FORWARD_BACK, back, np.nan), color="#d62728", lw=3.2, zorder=5)
    _hline(ax, LEAN_FORWARD_BACK, f"LEAN_FORWARD limit ({LEAN_FORWARD_BACK}°)", "#d62728")
    _mark(ax, t, back, reps, "max", "{:.0f}°", "#9467bd")
    ax.set_ylim(0, top(back, LEAN_FORWARD_BACK) * 1.35)
    leg(ax, [Line2D([0], [0], color="#9467bd", lw=1.8, label="Back lean"),
             Line2D([0], [0], color="#d62728", lw=3.2, label="Over limit")])
    fig3.savefig(os.path.join(out_dir, f"{stem}_back.png"), dpi=160)
    print(f"saved graphs: {stem}_knee.png, {stem}_heel.png, {stem}_back.png")

    if show_plots:
        plt.show()                                         # blocks until the graph windows are closed
    else:
        plt.close("all")

# ----------------------------------------------------------------- main processing
def process_video(video_path, show=True, out_dir="outputs", show_plots=False):
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
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    if show:
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, on_mouse, stop_button_rect)

    state = "S1"
    prev = None                      # previous smoothed (knee, hip, back, ankle)
    reps = 0                         # reps with no flagged error
    rep_id = 0
    in_rep = reached_depth = False
    rep_start_t = 0.0
    min_knee = 999.0
    frame_scores = []
    err_streak, err_max = {}, {}     # current run / longest run of consecutive bad frames, per error code
    vk, vb = Velocity(VEL_WINDOW_S), Velocity(VEL_WINDOW_S)     # knee-angle and back-lean velocity
    rep_series = []
    last_result = "-"
    heel_buf = {"LEFT": deque(maxlen=HEEL_BASE_FRAMES), "RIGHT": deque(maxlen=HEEL_BASE_FRAMES)}  # standing heel y (px)
    scale_buf = deque(maxlen=HEEL_BASE_FRAMES)                  # standing shank length (px)
    heel_s = {"LEFT": None, "RIGHT": None}                      # smoothed heel y (px) per side
    heel_base, heel_scale = {}, None                            # frozen at the start of each rep
    heel_floor = {"LEFT": None, "RIGHT": None}                  # fallback baseline: lowest heel seen this rep
    heel_rise_max = 0.0
    knee_toe_max = -9.0
    accuracies, last_acc = [], 0.0
    rep_rows, frame_rows = [], []
    frame_idx = 0

    def snapshot(feedback, level, **extra):
        """Everything the side panel shows, read from the current state of the loop."""
        d = {"feedback": feedback, "level": level, "state": state, "reps": reps, "attempts": rep_id,
             "last_acc": last_acc, "avg_acc": (sum(accuracies) / len(accuracies)) if accuracies else None,
             "last_result": last_result, "angles": None, "vels": None, "heel_rise": None}
        d.update(extra)
        return d

    # a NEW landmarker per video: VIDEO mode needs monotonically increasing timestamps
    with vision.PoseLandmarker.create_from_options(options) as detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            this_frame = frame_idx
            t_s = this_frame / fps
            ts_ms = int(this_frame * 1000 / fps)
            frame_idx += 1                                   # increment BEFORE any `continue`

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = detector.detect_for_video(mp_image, ts_ms)

            disp = fit(frame)
            dh, dw = disp.shape[:2]
            row = {"frame": this_frame, "time_s": round(t_s, 3), "detected": 0, "state": state,
                   "feedback": "", "errors": "", "knee": "", "hip": "", "back": "", "ankle": ""}
            

            # ---- validity checks (all in one place, no stray `continue`s)
            skip_msg = None
            if not results.pose_landmarks:
                skip_msg = "No Person Detected"
            else:
                lm = results.pose_landmarks[0]               # list of landmarks for pose 0
                left_vis, right_vis = lm[LEFT_KNEE].visibility, lm[RIGHT_KNEE].visibility
                if max(left_vis, right_vis) < VIS_MIN:
                    skip_msg = "Cannot Detect Pose - Adjust Position"
                else:
                    ok, msg = is_side_view(lm)
                    if not ok:
                        skip_msg = msg
            if skip_msg:
                row["feedback"] = skip_msg
                frame_rows.append(row)
                err_streak.clear()                       # an unusable frame breaks every streak
                vk.reset(); vb.reset()
                heel_s["LEFT"] = heel_s["RIGHT"] = None
                put_wrapped(disp, skip_msg, 16, 40, dw - 32, 0.7, C_RED, 28, 2, 3)
                if present(make_canvas(disp, snapshot(skip_msg, "error")), show):
                    break
                continue

            side_name = "LEFT" if left_vis > right_vis else "RIGHT"
            idx = ((LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE, LEFT_FOOT_INDEX) if left_vis > right_vis
                   else (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE, RIGHT_FOOT_INDEX))
            shoulder, hip, knee, ankle, foot = (lm[i] for i in idx)

            sx, sy, hx, hy = shoulder.x * w, shoulder.y * h, hip.x * w, hip.y * h
            kx, ky, ax, ay = knee.x * w, knee.y * h, ankle.x * w, ankle.y * h
            fx, fy = foot.x * w, foot.y * h

            raw = (calculate_jointangles(hx, hy, kx, ky, ax, ay),       # knee
                   calculate_jointangles(sx, sy, hx, hy, kx, ky),       # hip
                   calculate_backlean(sx, sy, hx, hy),                  # back lean
                   calculate_jointangles(kx, ky, ax, ay, fx, fy))       # ankle
            if any(np.isnan(v) for v in raw):
                row["feedback"] = "NaN angle"
                frame_rows.append(row)
                err_streak.clear()
                vk.reset(); vb.reset()
                heel_s["LEFT"] = heel_s["RIGHT"] = None
                if present(make_canvas(disp, snapshot("Angle error - skipping frame", "warn")), show):
                    break
                continue

            if prev is None:
                prev = raw
            knee_angle, hip_angle, back_angle, ankle_angle = (
                ALPHA * p + (1 - ALPHA) * r for p, r in zip(prev, raw))
            going_down = knee_angle < prev[0]
            prev = (knee_angle, hip_angle, back_angle, ankle_angle)
            # knee vs TOE (not ankle), signed toward the facing direction, divided by shank length -> camera-independent
            heel_lm = lm[LEFT_HEEL if side_name == "LEFT" else RIGHT_HEEL]
            face_dir = 1 if foot.x >= heel_lm.x else -1
            shank_px = float(np.hypot(kx - ax, ky - ay))
            knee_over_toe = face_dir * (kx - fx) / shank_px if shank_px > 1e-6 else 0.0
            knee_vel = vk.update(t_s, knee_angle)            # deg/s, negative = descending
            back_vel = vb.update(t_s, back_angle)            # deg/s
            for sd, i_heel in (("LEFT", LEFT_HEEL), ("RIGHT", RIGHT_HEEL)):      # smoothed heel height, both feet
                if lm[i_heel].visibility >= VIS_MIN:
                    y = lm[i_heel].y * h
                    heel_s[sd] = y if heel_s[sd] is None else ALPHA * heel_s[sd] + (1 - ALPHA) * y
                else:
                    heel_s[sd] = None

            # ---- state machine (state is updated FIRST so scoring uses the current frame's state)
            if knee_angle > STAND_KNEE:
                if in_rep and (reached_depth or min_knee < PARTIAL_MIN_KNEE):
                    rep_id += 1
                    metrics, bounced = rep_metrics(rep_series, t_s) if rep_series else ({}, False)
                    metrics = dict(metrics, m_heel_lift_max=round(heel_rise_max, 3),
                                   m_knee_toe_max=round(knee_toe_max, 3) if knee_toe_max > -9 else "")
                    errs = sorted(c for c, n in err_max.items() if n >= (VEL_ERR_MIN_FRAMES if c in VEL_MSG else ERR_MIN_FRAMES))
                    if reached_depth and bounced:                  # BOUNCE is a single event, judged once per rep
                        errs = sorted(set(errs) | {"BOUNCE"})
                    if reached_depth and abs(metrics.get("m_bottom_lean_shift", 0.0)) > BUTT_WINK_SHIFT:
                        errs = sorted(set(errs) | {"BUTT_WINK"})
                    if not reached_depth:
                        errs = ["SHALLOW"]
                    elif min_knee > DEPTH_TARGET_KNEE:
                        errs = sorted(set(errs) | {"SQUAT_DEEPER"})
                    errs = [e for e in errs if e in ENABLED_ERRORS]
                    failing = [e for e in errs if e not in ADVISORY_ERRORS]
                    acc = (sum(frame_scores) / len(frame_scores) * 100) if frame_scores else 0.0
                    last_result = "OK" if not errs else ";".join(errs)
                    if not failing:
                        reps += 1
                    if reached_depth:
                        accuracies.append(acc)
                        last_acc = acc
                    rep_rows.append({"video": video_name, "rep_id": rep_id,
                                     "start_s": round(rep_start_t, 3), "end_s": round(t_s, 3),
                                     "label": "incorrect" if failing else "correct",
                                     "errors": ";".join(errs), "accuracy_pct": round(acc, 1),
                                     "reached_depth": int(reached_depth), **metrics})
                in_rep = False
                state = "S1"
            else:
                if not in_rep:                                # a new rep attempt begins
                    in_rep, reached_depth, min_knee = True, False, 999.0
                    rep_start_t = t_s
                    frame_scores, err_streak, err_max = [], {}, {}
                    rep_series = []
                    # freeze the flat-foot baseline from the standing frames just before this rep
                    heel_base = {sd: float(np.percentile(b, 80)) for sd, b in heel_buf.items() if len(b) >= HEEL_BASE_MIN_FRAMES}
                    heel_scale = float(np.median(scale_buf)) if len(scale_buf) >= HEEL_BASE_MIN_FRAMES else None
                    heel_floor = {"LEFT": None, "RIGHT": None}
                    heel_rise_max = 0.0
                    knee_toe_max = -9.0
                min_knee = min(min_knee, knee_angle)
                if knee_angle < BOTTOM_KNEE:
                    state, reached_depth = "S3", True
                else:
                    state = "S2" if going_down else "S4"

            if in_rep:
                rep_series.append((t_s, knee_angle, knee_vel, back_vel, -back_angle * face_dir, hip_angle))   # -back*face = lean toward facing direction
                if state == "S3":
                    knee_toe_max = max(knee_toe_max, knee_over_toe)

            # ---- momentum / swing (velocity based, so it needs no pose rules of its own)
            vel_errs = []
            if in_rep:
                if state in ("S2", "S3") and knee_vel < -FAST_DESCENT_VEL:
                    vel_errs.append("FAST_DESCENT")
                if abs(back_vel) > TORSO_SWING_VEL:
                    vel_errs.append("TORSO_SWING")

            if not in_rep:                               # standing: keep learning where the flat heels sit
                for sd in ("LEFT", "RIGHT"):
                    if heel_s[sd] is not None:
                        heel_buf[sd].append(heel_s[sd])
                scale_buf.append(float(np.hypot(kx - ax, ky - ay)))

            # ---- heel lift.  Baseline = 80th percentile of heel y (i.e. near its LOWEST position, since image y
            # ---- grows downward) over the standing frames just before the rep.  A high percentile, not the median,
            # ---- so a heel that already started rising before the knee crossed STAND_KNEE can't drag the baseline
            # ---- up.  If there were no standing frames, fall back to the lowest heel position seen in the rep.
            heel_errs, heel_rise, heel_ref = [], None, None
            if in_rep and heel_s[side_name] is not None:
                y = heel_s[side_name]
                prev_floor = heel_floor[side_name]
                heel_floor[side_name] = y if prev_floor is None else max(prev_floor, y)    # image y grows downward
                heel_ref = heel_base.get(side_name, heel_floor[side_name])
                scale = heel_scale or float(np.hypot(kx - ax, ky - ay))
                if scale > 1e-6:
                    heel_rise = (heel_ref - y) / scale           # > 0 = heel is ABOVE its flat-foot position
                    heel_rise_max = max(heel_rise_max, heel_rise)
                    if heel_rise > HEEL_LIFT_RATIO:
                        heel_errs.append("HEEL_LIFT")

            # ---- feedback + scoring
            feedback, level, frame_errs = "", "ok", []
            if state == "S1":
                feedback = "Standing - Start Your Squat"

            elif state == "S2":
                if abs(back_angle) > 35:
                    feedback, level = "Keep Back Straight While Descending", "warn"
                elif knee_angle > 120:
                    feedback, level = "Keep Descending", "warn"
                else:
                    feedback = "Good Descent"

            elif state == "S3":
                ks = 1.0 if knee_angle <= DEPTH_TARGET_KNEE else max(0, 1 - (knee_angle - DEPTH_TARGET_KNEE) / 40)   # deeper than 85 is not penalised
                hs = max(0, 1 - abs(hip_angle - 80) / 40)
                bs = 1.0 if abs(back_angle) <= LEAN_FORWARD_BACK else max(0, 1 - (abs(back_angle) - LEAN_FORWARD_BACK) / 20)
                as_ = max(0, 1 - abs(ankle_angle - 80) / 30)
                frame_scores.append(ks * 0.35 + bs * 0.30 + hs * 0.20 + as_ * 0.15)

                errs_now = s3_errors(back_angle, knee_angle, hip_angle, ankle_angle, knee_over_toe)
                frame_errs = [c for c, _ in errs_now]
                if errs_now:
                    feedback, level = errs_now[0][1], "error"
                elif knee_angle > DEPTH_TARGET_KNEE:
                    feedback, level = "Squat Deeper", "warn"
                elif 30 <= abs(back_angle) <= 45:
                    feedback = "Good Torso Posture"
                elif 60 < knee_angle <= DEPTH_TARGET_KNEE:
                    feedback = "Good Squat Depth"
                elif 50 <= hip_angle <= 90:
                    feedback = "Good Hip Engagement"
                elif 70 <= ankle_angle <= 100:
                    feedback = "Good Posture"
                else:
                    feedback = "Good Form - Hold Position"

            elif state == "S4":
                if abs(back_angle) > 35:
                    feedback, level = "Keep Back Straight While Rising", "warn"
                else:
                    feedback = "Good - Drive Through Heels"

            extra_errs = [c for c in heel_errs + vel_errs if c in ENABLED_ERRORS]
            frame_errs = frame_errs + extra_errs
            if extra_errs and level != "error":
                feedback, level = ALERT_MSG[extra_errs[0]], "warn"

            # ---- consecutive-frame tracking, for EVERY error code in every state
            for code in list(err_streak):                # a streak ends the moment the error stops firing
                if code not in frame_errs:
                    err_streak[code] = 0
            for code in frame_errs:
                err_streak[code] = err_streak.get(code, 0) + 1
                err_max[code] = max(err_max.get(code, 0), err_streak[code])
            if in_rep:      # an error confirmed earlier in this rep stays on screen, in red, until the rep ends
                confirmed = [c for c in LIVE_PRIORITY if c in ENABLED_ERRORS and
                             err_max.get(c, 0) >= (VEL_ERR_MIN_FRAMES if c in VEL_MSG else ERR_MIN_FRAMES)]
                if confirmed:
                    feedback, level = LIVE_MSG[confirmed[0]], "error"
                    if len(confirmed) > 1:
                        feedback += f"  (+{len(confirmed) - 1} more)"

            if in_rep and state == "S4" and knee_angle > min_knee + 10:    # clearly rising: the bottom is known now
                if not reached_depth:
                    feedback, level = "Shallow Squat Depth", "error"
                elif min_knee > DEPTH_TARGET_KNEE and "SQUAT_DEEPER" in ENABLED_ERRORS:
                    feedback, level = "Squat Deeper", "error"
            # ---- overlay
            pts = [(int(p.x * dw), int(p.y * dh)) for p in (shoulder, hip, knee, ankle, foot)]
            cv2.polylines(disp, [np.array(pts, np.int32)], False, (255, 200, 0), 3)
            for p in pts:
                cv2.circle(disp, p, 6, (0, 255, 255), -1)
            if heel_s[side_name] is not None:             # green/red dot on the heel; orange tick = flat-foot baseline
                hp = (int(lm[LEFT_HEEL if side_name == "LEFT" else RIGHT_HEEL].x * dw), int(heel_s[side_name] / h * dh))
                cv2.circle(disp, hp, 7, (0, 0, 255) if heel_errs else (0, 255, 0), -1)
                if heel_ref is not None:
                    yb = int(heel_ref / h * dh)
                    cv2.line(disp, (hp[0] - 35, yb), (hp[0] + 35, yb), (0, 165, 255), 2)
            face = 1 if foot.x >= ankle.x else -1          # +1 = the person faces right in the image
            video_label(disp, f"B {back_angle:.0f}", pts[0], -face)                      # back lean (at the shoulder)
            video_label(disp, f"H {hip_angle:.0f}", pts[1], -face)                       # hip
            video_label(disp, f"K {knee_angle:.0f}", pts[2], face)                       # knee
            video_label(disp, f"A {ankle_angle:.0f}", (pts[3][0], pts[3][1] - 26), face)  # ankle (clear of the foot)
                        # ---- state label, top-left corner of the video
            s_label = STATE_NAMES.get(state, state)                                      # e.g. "S3 bottom"
            s_color = tuple(int(STATE_COLORS[state][i:i + 2], 16) for i in (5, 3, 1))    # same colours as the graph strip, hex -> BGR
            (tw, th), _ = cv2.getTextSize(s_label, FONT, 0.8, 2)
            roi = disp[8:th + 28, 8:tw + 28]
            roi[:] = (roi * 0.35).astype(np.uint8)                                       # darken a box so it is readable on any background
            cv2.putText(disp, s_label, (16, th + 16), FONT, 0.8, s_color, 2, cv2.LINE_AA)
            info = snapshot(feedback, level, angles=(knee_angle, hip_angle, back_angle, ankle_angle),
                            vels=(knee_vel, back_vel), heel_rise=heel_rise)

            row.update(detected=1, state=state, feedback=feedback, errors=";".join(frame_errs),
                       knee=round(knee_angle, 1), hip=round(hip_angle, 1),
                       back=round(back_angle, 1), ankle=round(ankle_angle, 1),
                       knee_vel=round(knee_vel, 1), back_vel=round(back_vel, 1),
                       heel_rise="" if heel_rise is None else round(heel_rise, 3))
            frame_rows.append(row)

            if present(make_canvas(disp, info), show):
                break
    
    cap.release()
    write_csv(os.path.join(out_dir, f"{stem}_frames.csv"), frame_rows, FRAME_FIELDS)
    write_csv(os.path.join(out_dir, f"{stem}_reps.csv"), rep_rows, REP_FIELDS)
    try:
        make_session_plots(frame_rows, rep_rows, stem, out_dir, show_plots)
    except Exception as e:                                   # a plotting problem must never lose the CSVs
        print(f"Could not make graphs: {e}")
    return {"reps": reps, "attempts": rep_id, "rep_rows": rep_rows}



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=VIDEO_PATH, help="video file or folder of videos")
    ap.add_argument("--out", default="outputs")
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
    #if show and len(videos) == 1 and last:
        #show_summary(last["reps"], last["avg_accuracy"])


if __name__ == "__main__":
    main()