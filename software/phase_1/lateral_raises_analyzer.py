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

# MediaPipe pose landmark indices (person's own left/right, video NOT mirrored)
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
ARM_IDX = {"LEFT": (L_HIP, L_SHOULDER, L_ELBOW, L_WRIST),
           "RIGHT": (R_HIP, R_SHOULDER, R_ELBOW, R_WRIST)}

REST_ENTER = 30        # shoulder angle above this = arm leaves rest (rep starts)
REST_EXIT = 25         # ...below this = back at rest (rep ends); gap avoids chatter around one value
TOP_MIN = 85           # must reach this to be a full rep
TOP_MAX = 100          # above this = raised too high
PARTIAL_MIN_PEAK = 55  # a rise that never exceeds this is ignored (just a wobble)
ELBOW_MIN = 150        # elbow angle below this = elbow bent
ELBOW_CHECK_MIN = 40   # only judge the elbow once the arm is clearly raised
SYM_MAX_DIFF = 15      # left/right shoulder-angle difference allowed
ERR_MIN_FRAMES = 5     # an error must persist this many frames to flag the rep (1 = flag instantly)
ALPHA = 0.3            # smoothing weight of previous value
VIS_MIN = 0.5

window_name = "PowerPulse - Lateral Raise Detector"
summary_window_name = "Lateral Raise Summary"
stop_flag = False
stop_button_rect = [0, 0, 0, 0]
summary_close_flag = False
summary_close_rect = (220, 200, 330, 245)

REP_FIELDS = ["video", "rep_id", "arm", "start_s", "end_s", "label", "errors",
              "peak_angle", "min_elbow_angle", "reached_top"]
FRAME_FIELDS = ["frame", "time_s", "detected", "l_shoulder", "l_elbow", "l_state", "l_errors",
                "r_shoulder", "r_elbow", "r_state", "r_errors", "sym_diff"]

COLORS = {
    "good": (0, 220, 120), "bad": (0, 80, 220), "warning": (0, 165, 255),
    "info": (220, 220, 220), "accent": (255, 200, 0), "bg": (15, 15, 25), "panel": (30, 30, 45),
}
FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_SM = cv2.FONT_HERSHEY_SIMPLEX
PANEL_W, PANEL_H = 230, 345


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


# ----------------------------------------------------------------- rep counter
class RepCounter:
    """
    One per arm.  DOWN -> RISING -> UP -> LOWERING -> DOWN.
    A rep ATTEMPT runs from leaving rest to returning to rest.  Reps that never reach TOP_MIN are
    logged as NOT_HIGH_ENOUGH instead of being silently dropped (the old version counted a rep the
    moment the arm hit 85 deg, whatever happened afterwards, and ignored partial reps entirely).
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
        self.start_t = 0.0
        self.peak = 0.0
        self.min_elbow = 999.0
        self.err_counts = {}

    def flashing(self, t):
        return t - self.last_flash < 0.8

    def update(self, angle, elbow, sym_bad, t, video_name):
        """Feed one frame. Returns the error codes firing on this frame."""
        if angle is None:
            return []
        if self.state == "DOWN":
            if angle <= REST_ENTER:
                return []
            self._clear()
            self.state, self.start_t = "RISING", t

        self.peak = max(self.peak, angle)
        frame_errs = []
        if elbow is not None and angle >= ELBOW_CHECK_MIN:
            self.min_elbow = min(self.min_elbow, elbow)
            if elbow < ELBOW_MIN:
                frame_errs.append("ELBOW_BENT")
        if angle > TOP_MAX:
            frame_errs.append("TOO_HIGH")
        if sym_bad:
            frame_errs.append("UNEVEN_RAISE")
        for c in frame_errs:
            self.err_counts[c] = self.err_counts.get(c, 0) + 1

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
        errs = sorted(c for c, n in self.err_counts.items() if n >= ERR_MIN_FRAMES)
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
            "reached_top": int(reached)})


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
    (tw, th), _ = cv2.getTextSize("STOP", FONT_SM, 0.85, 2)
    cv2.putText(frame, "STOP", (x1 + (x2 - x1 - tw) // 2, y1 + (y2 - y1 + th) // 2),
                FONT_SM, 0.85, (255, 255, 255), 2)


def fit(frame, max_w=1000, max_h=700):
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
    cv2.putText(img, "SUMMARY", (20, 40), FONT_SM, 0.9, (255, 255, 255), 2)
    cv2.putText(img, f"Left : {left.reps} correct / {left.attempts}", (20, 90), FONT_SM, 0.7, (255, 255, 255), 2)
    cv2.putText(img, f"Right: {right.reps} correct / {right.attempts}", (20, 130), FONT_SM, 0.7, (255, 255, 255), 2)
    rounded_rect(img, summary_close_rect, (0, 0, 255), radius=12)
    cv2.putText(img, "Close", (summary_close_rect[0] + 25, summary_close_rect[1] + 30), FONT_SM, 0.7, (255, 255, 255), 2)
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


# ----------------------------------------------------------------- drawing helpers
# NB: cv2 Hershey fonts only render plain ASCII - the old unicode arrows / ticks / degree signs
# were drawn as "?" so everything below is ASCII.
def draw_rounded_rect_overlay(img, x1, y1, x2, y2, radius, color, alpha=0.65):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    for cx, cy in ((x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
                   (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)):
        cv2.circle(overlay, (cx, cy), radius, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def put_text_shadow(img, text, pos, font, scale, color, thickness=1):
    x, y = pos
    cv2.putText(img, text, (x + 1, y + 1), font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_angle_arc(img, center, angle, radius=28, color=(255, 200, 0)):
    cv2.ellipse(img, center, (radius, radius), -90, 0, min(int(angle), 180), color, 2, cv2.LINE_AA)


def draw_side_panel(img, x0, y0, side, counter, sh_ang, el_ang, wrist_y, shoulder_y, form_ok, sym_ok, t):
    pw, ph = PANEL_W - 20, PANEL_H - 15
    draw_rounded_rect_overlay(img, x0, y0, x0 + pw, y0 + ph, 10, COLORS["panel"], alpha=0.9)
    px, py = x0, y0

    side_color = COLORS["good"] if side == "LEFT" else COLORS["accent"]
    put_text_shadow(img, f"{side} ARM", (px + 10, py + 28), FONT, 0.65, side_color, 1)
    rep_color = (0, 255, 255) if counter.flashing(t) else COLORS["good"]
    put_text_shadow(img, f"REPS: {counter.reps} / {counter.attempts}", (px + 10, py + 58), FONT, 0.65, rep_color, 1)

    if sh_ang is not None:
        sa_color = (COLORS["good"] if TOP_MIN <= sh_ang <= TOP_MAX else
                    COLORS["warning"] if REST_ENTER <= sh_ang < TOP_MIN else COLORS["info"])
        put_text_shadow(img, f"Shoulder: {int(sh_ang)} deg", (px + 10, py + 92), FONT_SM, 0.52, sa_color, 1)
        bx, by, bw = px + 10, py + 100, pw - 20
        cv2.rectangle(img, (bx, by), (bx + bw, by + 6), (60, 60, 80), -1)
        cv2.rectangle(img, (bx, by), (bx + int(np.clip(sh_ang / 100, 0, 1) * bw), by + 6), sa_color, -1)
    else:
        put_text_shadow(img, "Shoulder: --", (px + 10, py + 92), FONT_SM, 0.52, COLORS["info"], 1)

    if el_ang is not None:
        ok = el_ang >= ELBOW_MIN
        put_text_shadow(img, f"Elbow: {int(el_ang)} deg {'OK' if ok else 'BEND!'}", (px + 10, py + 135),
                        FONT_SM, 0.50, COLORS["good"] if ok else COLORS["bad"], 1)
    else:
        put_text_shadow(img, "Elbow: --", (px + 10, py + 135), FONT_SM, 0.50, COLORS["info"], 1)

    if wrist_y is not None and shoulder_y is not None:
        level = abs(wrist_y - shoulder_y) < 0.06
        put_text_shadow(img, f"Wrist: {'at level' if level else 'not level'}", (px + 10, py + 165),
                        FONT_SM, 0.50, COLORS["good"] if level else COLORS["warning"], 1)
    else:
        put_text_shadow(img, "Wrist: --", (px + 10, py + 165), FONT_SM, 0.50, COLORS["info"], 1)

    put_text_shadow(img, "Form: OK" if form_ok else "Form: Fix elbow", (px + 10, py + 198), FONT_SM, 0.50,
                    COLORS["good"] if form_ok else COLORS["bad"], 1)
    put_text_shadow(img, "Sym:  Even" if sym_ok else "Sym:  Uneven", (px + 10, py + 226), FONT_SM, 0.50,
                    COLORS["good"] if sym_ok else COLORS["warning"], 1)

    state_map = {"DOWN": "v DOWN", "RISING": "^ RISING", "UP": "= TOP", "LOWERING": "v LOWERING"}
    sc = {"DOWN": COLORS["info"], "RISING": COLORS["warning"], "UP": COLORS["good"], "LOWERING": COLORS["warning"]}
    put_text_shadow(img, state_map[counter.state], (px + 10, py + 260), FONT, 0.60, sc[counter.state], 1)
    put_text_shadow(img, f"Last: {counter.last_result}", (px + 10, py + 288), FONT_SM, 0.42, COLORS["info"], 1)
    if counter.flashing(t):
        put_text_shadow(img, "REP!", (px + pw // 2 - 30, py + ph - 14), FONT, 0.85, (0, 255, 255), 2)


# ----------------------------------------------------------------- main processing
def process_video(video_path, show=True, out_dir="outputs_raise"):
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
    left_panel_arm = "RIGHT"        # video is not mirrored: the person's LEFT arm is on the image's right
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

            if results.pose_landmarks:
                lm = results.pose_landmarks[0]
                left_panel_arm = "LEFT" if lm[L_SHOULDER].x < lm[R_SHOULDER].x else "RIGHT"
                for side in ("LEFT", "RIGHT"):
                    got = arm_angles(lm, side, w, h)
                    if got is None:
                        continue
                    raw = got[:2]
                    if smooth[side] is None:
                        smooth[side] = raw
                    smooth[side] = tuple(ALPHA * p + (1 - ALPHA) * r for p, r in zip(smooth[side], raw))
                    arm[side] = (smooth[side][0], smooth[side][1], got[2])

            # ---- symmetry (only when both arms are off rest, so a one-arm exercise isn't flagged)
            a_l, a_r = arm["LEFT"], arm["RIGHT"]
            sym_diff, sym_ok, sym_bad = None, True, False
            if a_l and a_r:
                sym_diff = abs(a_l[0] - a_r[0])
                sym_ok = sym_diff < SYM_MAX_DIFF
                sym_bad = (not sym_ok) and min(a_l[0], a_r[0]) > REST_ENTER

            # ---- rep counters + logging
            row = {"frame": this_frame, "time_s": round(t_s, 3), "detected": int(bool(results.pose_landmarks)),
                   "sym_diff": "" if sym_diff is None else round(sym_diff, 1)}
            for side, key in (("LEFT", "l"), ("RIGHT", "r")):
                a = arm[side]
                c = counters[side]
                errs = c.update(a[0], a[1], sym_bad, t_s, video_name) if a else []
                row[f"{key}_shoulder"] = "" if not a else round(a[0], 1)
                row[f"{key}_elbow"] = "" if not a else round(a[1], 1)
                row[f"{key}_state"] = c.state
                row[f"{key}_errors"] = ";".join(errs)
            frame_rows.append(row)

            # ---- drawing (all on the resized frame; panels live on a canvas beside the video so
            #      they never cover the person, even for portrait phone videos)
            for side in ("LEFT", "RIGHT"):
                a = arm[side]
                if not a:
                    continue
                form_ok = a[1] >= ELBOW_MIN
                col = COLORS["good"] if form_ok else COLORS["bad"]
                pts = [(int(p.x * dw), int(p.y * dh)) for p in a[2]]      # hip, shoulder, elbow, wrist
                for i in range(3):
                    cv2.line(disp, pts[i], pts[i + 1], col, 3, cv2.LINE_AA)
                for p in pts:
                    cv2.circle(disp, p, 5, col, -1, cv2.LINE_AA)
                    cv2.circle(disp, p, 7, (255, 255, 255), 1, cv2.LINE_AA)
                draw_angle_arc(disp, pts[1], a[0], 30, COLORS["accent"])
                draw_angle_arc(disp, pts[2], a[1], 20, col)
                dx = -40 if side == "RIGHT" else 8      # keep labels outside the body
                put_text_shadow(disp, f"{int(a[0])}", (pts[1][0] + dx, pts[1][1] - 15), FONT_SM, 0.5, COLORS["accent"], 1)
                put_text_shadow(disp, f"{int(a[1])}", (pts[2][0] + dx, pts[2][1] + 18), FONT_SM, 0.5, col, 1)

            if not results.pose_landmarks:
                put_text_shadow(disp, "No person detected - step into frame", (20, dh // 2), FONT, 0.6, COLORS["warning"], 1)

            draw_rounded_rect_overlay(disp, max(dw // 2 - 160, 0), 8, min(dw // 2 + 160, dw), 46, 8, COLORS["panel"], 0.75)
            put_text_shadow(disp, "LATERAL RAISE DETECTOR", (max(dw // 2 - 148, 4), 36), FONT, 0.6, COLORS["accent"], 1)
            if sym_bad:
                bx, by = max(dw // 2 - 140, 10), dh - 70
                draw_rounded_rect_overlay(disp, bx - 8, by - 24, bx + 290, by + 10, 6, (0, 40, 100), 0.80)
                put_text_shadow(disp, "! UNEVEN RAISE - check symmetry", (bx, by), FONT_SM, 0.55, COLORS["bad"], 1)

            canvas = np.full((max(dh, PANEL_H + 20), dw + 2 * PANEL_W, 3), COLORS["bg"], np.uint8)
            canvas[:dh, PANEL_W:PANEL_W + dw] = disp
            other = "RIGHT" if left_panel_arm == "LEFT" else "LEFT"
            for x0, side in ((10, left_panel_arm), (PANEL_W + dw + 10, other)):
                a = arm[side]
                draw_side_panel(canvas, x0, 10, side, counters[side],
                                a[0] if a else None, a[1] if a else None,
                                a[2][3].y if a else None, a[2][1].y if a else None,
                                (a[1] >= ELBOW_MIN) if a else True, sym_ok, t_s)

            if present(canvas, show):
                break

    cap.release()
    rep_rows = sorted(counters["LEFT"].records + counters["RIGHT"].records,
                      key=lambda r: (r["start_s"], r["arm"]))
    write_csv(os.path.join(out_dir, f"{stem}_frames.csv"), frame_rows, FRAME_FIELDS)
    write_csv(os.path.join(out_dir, f"{stem}_reps.csv"), rep_rows, REP_FIELDS)
    L, R = counters["LEFT"], counters["RIGHT"]
    print(f"{video_name}: left {L.reps}/{L.attempts} correct, right {R.reps}/{R.attempts} correct")
    return {"left": L, "right": R, "rep_rows": rep_rows}


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="video file or folder of videos")
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
        last = process_video(v, show=show, out_dir=args.out)
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