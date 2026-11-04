"""
PowerPulse - Lunge Analyzer (video-file version + logging for trainer comparison)

Usage:
  python powerpulse_lunge_fixed.py --input path/to/video.mp4
  python powerpulse_lunge_fixed.py --input path/to/folder --no-display      # batch, headless

Outputs (in --out folder, default ./outputs_lunge):
  <video>_frames.csv        per-frame angles / state / feedback (for debugging)
  <video>_reps.csv          one row per rep with predicted label + error codes
  all_reps_predicted.csv    all videos combined  -> feed this to compare_labels.py
"""
import argparse
import csv
import glob
import math
import os
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = "pose_landmarker_heavy.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_heavy/float16/latest/"
    "pose_landmarker_heavy.task"
)

LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_HEEL, RIGHT_HEEL = 29, 30
LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX = 31, 32

STAND_KNEE = 160          #both knees above this = standing
BOTTOM_KNEE = 105         #front knee at/below this = bottom phase (S3)
PARTIAL_MIN_KNEE = 130    #a rep that never goes below this is ignored (just a wobble)
ALPHA = 0.3
VIS_MIN = 0.6
ERR_MIN_FRAMES = 5        #an error must persist this many S3 frames to flag the rep (1 = old behaviour)
STRIDE_MIN_RATIO = 0.5    #ankle separation / torso length below this = feet together ("NONE")
FEEDBACK_THRESHOLD = 8    #frames a message must persist before it is displayed

window_name = "PowerPulse - Lunge Analyzer"
summary_window_name = "Lunge Summary"
stop_flag = False
stop_button_rect = [0, 0, 0, 0]
summary_close_flag = False
summary_close_rect = (220, 200, 330, 245)

REP_FIELDS = ["video", "rep_id", "start_s", "end_s", "label", "errors", "accuracy_pct", "reached_depth", "leg"]
FRAME_FIELDS = ["frame", "time_s", "detected", "state", "leg", "feedback", "errors",
                "front_knee", "back_knee", "hip", "back"]


def download_model():
    if os.path.exists(MODEL_PATH):
        print(f"Model already present: {MODEL_PATH}")
        return
    print(f"Downloading model -> {MODEL_PATH}")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.")


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


def fit(frame, max_w=1100, max_h=700):
    h, w = frame.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        return cv2.resize(frame, (int(w * scale), int(h * scale)))
    return frame.copy()


def present(disp, show):
    """Show a frame, handle keys. Returns True if the user wants to quit."""
    if not show:
        return False
    stop_button_rect[:] = get_stop_button_rect(disp)  
    draw_stop_button(disp)
    cv2.imshow(window_name, disp)
    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), 27):
        return True
    if key == ord(' '):                                  
        while not stop_flag:
            k = cv2.waitKey(50) & 0xFF
            if k == ord(' '):
                break
            if k in (ord('q'), 27):
                return True
    return stop_flag


def make_summary_image(reps, attempts, avg_accuracy, left_reps, right_reps):
    img = np.zeros((260, 350, 3), np.uint8)
    img[:] = (40, 40, 40)
    cv2.putText(img, "SUMMARY", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(img, f"Correct reps: {reps} / {attempts}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(img, f"Left: {left_reps}  Right: {right_reps}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(img, f"Accuracy: {avg_accuracy:.1f}%", (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    rounded_rect(img, summary_close_rect, (0, 0, 255), radius=12)
    cv2.putText(img, "Close", (summary_close_rect[0] + 25, summary_close_rect[1] + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


def show_summary(reps, attempts, avg_accuracy, left_reps, right_reps):
    global summary_close_flag
    summary_close_flag = False
    cv2.namedWindow(summary_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(summary_window_name, 350, 260)
    cv2.setMouseCallback(summary_window_name, on_summary_mouse, summary_close_rect)
    while True:
        cv2.imshow(summary_window_name, make_summary_image(reps, attempts, avg_accuracy, left_reps, right_reps))
        if (cv2.waitKey(20) & 0xFF) == 27 or summary_close_flag:
            break
    cv2.destroyWindow(summary_window_name)


def calculate_jointangles(x1, y1, x2, y2, x3, y3):
    """Angle at point 2 (degrees) between 2->1 and 2->3. Returns NaN if points coincide."""
    v1 = np.array([x1 - x2, y1 - y2], dtype=float)
    v2 = np.array([x3 - x2, y3 - y2], dtype=float)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom < 1e-6:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))))


def calculate_backlean(shoulder_x, shoulder_y, hip_x, hip_y):
    dx, dy = shoulder_x - hip_x, shoulder_y - hip_y
    lean = float(np.degrees(np.arctan2(abs(dx), -dy)))
    return lean if shoulder_x < hip_x else -lean


def is_side_view(lm):
    if abs(lm[LEFT_SHOULDER].x - lm[RIGHT_SHOULDER].x) > 0.25:
        return False, "Please Turn Sideways to the Camera"
    return True, "Good Position"


def detect_leading_leg(lm, w, torso_px):
    """
    Returns (leg, facing). facing = +1 if the person faces right in the image, -1 if left.
    The leading leg is the one whose ankle is further ahead in the facing direction.
    (The old version compared |ankle - hip_centre| for each leg, but in a lunge the hips sit
    midway between the feet, so both distances are about equal and it kept returning NONE.)
    """
    toe_minus_heel = ((lm[LEFT_FOOT_INDEX].x - lm[LEFT_HEEL].x) +
                      (lm[RIGHT_FOOT_INDEX].x - lm[RIGHT_HEEL].x))
    facing = 1 if toe_minus_heel >= 0 else -1
    diff_px = (lm[LEFT_ANKLE].x - lm[RIGHT_ANKLE].x) * w
    if abs(diff_px) < STRIDE_MIN_RATIO * torso_px:
        return "NONE", facing
    return ("LEFT" if diff_px * facing > 0 else "RIGHT"), facing


def s3_errors(back, front_knee, back_knee, knee_over_toe):
    """All bottom-phase errors firing this frame, in priority order: (code, message)."""
    errs = []
    if abs(back) > 30:       errs.append(("LEAN_FORWARD", "Torso Leaning - Keep Upright"))
    if knee_over_toe > 0.08: errs.append(("KNEES_FORWARD", "Front Knee Too Far Forward"))
    if back_knee > 135:      errs.append(("BACK_KNEE_HIGH", "Lower Back Knee - Drop Closer to Ground"))
    if back_knee < 80:       errs.append(("BACK_KNEE_LOW", "Back Knee Too Low - Don't Touch Ground"))
    if front_knee < 75:      errs.append(("TOO_DEEP", "Too Deep - Come Up Slightly"))
    return errs


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)


def process_video(video_path, show=True, out_dir="outputs_lunge"):
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

    state = "S1"
    prev = None                         
    reps = left_reps = right_reps = 0   
    rep_id = 0
    in_rep = reached_depth = False
    rep_leg = last_leg = None
    facing = 1
    rep_start_t, min_knee = 0.0, 999.0
    frame_scores, err_counts = [], {}
    accuracies, last_acc = [], 0.0
    shown, pending, pending_n = ("Analyzing...", "warn"), None, 0
    rep_rows, frame_rows = [], []
    frame_idx = 0

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
            row = {"frame": this_frame, "time_s": round(t_s, 3), "detected": 0, "state": state, "leg": "",
                   "feedback": "", "errors": "", "front_knee": "", "back_knee": "", "hip": "", "back": ""}

            
            skip_msg = None
            if not results.pose_landmarks:
                skip_msg = "No Person Detected"
            else:
                lm = results.pose_landmarks[0]
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
                cv2.putText(disp, skip_msg, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if present(disp, show):
                    break
                continue

            
            torso_px = math.hypot(
                ((lm[LEFT_SHOULDER].x + lm[RIGHT_SHOULDER].x) - (lm[LEFT_HIP].x + lm[RIGHT_HIP].x)) / 2 * w,
                ((lm[LEFT_SHOULDER].y + lm[RIGHT_SHOULDER].y) - (lm[LEFT_HIP].y + lm[RIGHT_HIP].y)) / 2 * h)
            leading, new_facing = detect_leading_leg(lm, w, torso_px)
            if not in_rep:
                facing = new_facing
                if leading != "NONE":
                    last_leg = leading
            active = rep_leg if in_rep else (last_leg or ("LEFT" if left_vis >= right_vis else "RIGHT"))

            if active == "LEFT":
                i_sh, i_fh, i_fk, i_fa, i_ff = LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE, LEFT_FOOT_INDEX
                i_bh, i_bk, i_ba = RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE
            else:
                i_sh, i_fh, i_fk, i_fa, i_ff = RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE, RIGHT_FOOT_INDEX
                i_bh, i_bk, i_ba = LEFT_HIP, LEFT_KNEE, LEFT_ANKLE
            sh, fh, fk, fa, ff = lm[i_sh], lm[i_fh], lm[i_fk], lm[i_fa], lm[i_ff]
            bh, bk, ba = lm[i_bh], lm[i_bk], lm[i_ba]

            front_knee_raw = calculate_jointangles(fh.x * w, fh.y * h, fk.x * w, fk.y * h, fa.x * w, fa.y * h)
            if not in_rep and leading == "NONE":
                back_knee_raw = 180.0                       # feet together: no back leg to measure
            else:
                back_knee_raw = calculate_jointangles(bh.x * w, bh.y * h, bk.x * w, bk.y * h, ba.x * w, ba.y * h)
            hip_raw = calculate_jointangles(sh.x * w, sh.y * h, fh.x * w, fh.y * h, fk.x * w, fk.y * h)
            back_raw = calculate_backlean(sh.x * w, sh.y * h, fh.x * w, fh.y * h)

            raw = (front_knee_raw, back_knee_raw, hip_raw, back_raw)
            if any(np.isnan(v) for v in raw):
                row["feedback"] = "NaN angle"
                frame_rows.append(row)
                if present(disp, show):
                    break
                continue

            if prev is None:
                prev = raw
            front_knee_angle, back_knee_angle, hip_angle, back_angle = (
                ALPHA * p + (1 - ALPHA) * r for p, r in zip(prev, raw))
            going_down = front_knee_angle < prev[0]          # smoothed vs smoothed
            prev = (front_knee_angle, back_knee_angle, hip_angle, back_angle)

            knee_over_toe = (fk.x - ff.x) * facing          

            
            if front_knee_angle > STAND_KNEE and back_knee_angle > STAND_KNEE:
                if in_rep and (reached_depth or min_knee < PARTIAL_MIN_KNEE):
                    rep_id += 1
                    errs = sorted(c for c, n in err_counts.items() if n >= ERR_MIN_FRAMES)
                    if not reached_depth:
                        errs = ["SHALLOW"]
                    acc = (sum(frame_scores) / len(frame_scores) * 100) if frame_scores else 0.0
                    if not errs:
                        reps += 1
                    if rep_leg == "LEFT":
                        left_reps += 1
                    elif rep_leg == "RIGHT":
                        right_reps += 1
                    if reached_depth:
                        accuracies.append(acc)
                        last_acc = acc
                    rep_rows.append({"video": video_name, "rep_id": rep_id,
                                     "start_s": round(rep_start_t, 3), "end_s": round(t_s, 3),
                                     "label": "incorrect" if errs else "correct",
                                     "errors": ";".join(errs), "accuracy_pct": round(acc, 1),
                                     "reached_depth": int(reached_depth), "leg": rep_leg})
                in_rep = False
                state = "S1"
            elif front_knee_angle <= STAND_KNEE:
                if not in_rep:                                
                    in_rep, reached_depth, min_knee = True, False, 999.0
                    rep_leg, rep_start_t = active, t_s
                    frame_scores, err_counts = [], {}
                min_knee = min(min_knee, front_knee_angle)
                if front_knee_angle <= BOTTOM_KNEE:
                    state, reached_depth = "S3", True
                else:
                    state = "S2" if going_down else "S4"
            

            
            feedback, level, frame_errs = "", "ok", []
            if state == "S1":
                feedback = ("Standing - Step Forward to Begin Lunge" if leading == "NONE"
                            else "Return to Standing Position")

            elif state == "S2":
                if abs(back_angle) > 25:
                    feedback, level = "Keep Torso Upright While Descending", "warn"
                else:
                    feedback = "Good Descent - Keep Lowering"

            elif state == "S3":
                fks = max(0, 1 - abs(front_knee_angle - 90) / 40)
                bks = max(0, 1 - abs(back_knee_angle - 100) / 40)
                bs = max(0, 1 - abs(back_angle) / 30)
                hs = max(0, 1 - abs(hip_angle - 85) / 40)
                frame_scores.append(fks * 0.35 + bs * 0.30 + bks * 0.20 + hs * 0.15)

                errs_now = s3_errors(back_angle, front_knee_angle, back_knee_angle, knee_over_toe)
                for code, _ in errs_now:
                    err_counts[code] = err_counts.get(code, 0) + 1
                frame_errs = [c for c, _ in errs_now]
                if errs_now:
                    feedback, level = errs_now[0][1], "error"
                elif 85 <= front_knee_angle <= 95:
                    feedback = "Perfect Depth"
                elif abs(back_angle) <= 15:
                    feedback = "Excellent Posture"
                else:
                    feedback = "Good Form - Hold Position"

            elif state == "S4":
                if not reached_depth:
                    feedback, level = "Go Deeper Next Rep", "warn"
                elif abs(back_angle) > 25:
                    feedback, level = "Keep Torso Upright While Rising", "warn"
                else:
                    feedback = "Good - Push Through Front Heel"

            
            cand = (feedback, level)
            if cand == pending:
                pending_n += 1
            else:
                pending, pending_n = cand, 1
            if pending_n >= FEEDBACK_THRESHOLD:
                shown = pending
            disp_text, disp_level = shown
            color = {"ok": (0, 255, 0), "warn": (0, 165, 255), "error": (0, 0, 255)}[disp_level]

            
            def to_px(p):
                return (int(p.x * dw), int(p.y * dh))
            cv2.polylines(disp, [np.array([to_px(p) for p in (sh, fh, fk, fa, ff)], np.int32)],
                          False, (255, 200, 0), 3)
            cv2.polylines(disp, [np.array([to_px(p) for p in (bh, bk, ba)], np.int32)],
                          False, (200, 120, 255), 3)
            for p in (sh, fh, fk, fa, ff, bh, bk, ba):
                cv2.circle(disp, to_px(p), 5, (0, 255, 255), -1)

            lead_txt = rep_leg if in_rep else leading
            cv2.putText(disp, disp_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(disp, f"Correct reps: {reps}  (attempts: {rep_id})", (50, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(disp, f"Left: {left_reps} | Right: {right_reps}", (50, 125),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(disp, f"Leading: {lead_txt}", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(disp, f"State: {state}", (50, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            if accuracies:
                cv2.putText(disp, f"Last Rep: {last_acc:.1f}%", (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(disp, f"Session Avg: {sum(accuracies) / len(accuracies):.1f}%", (50, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            row.update(detected=1, state=state, leg=active, feedback=feedback, errors=";".join(frame_errs),
                       front_knee=round(front_knee_angle, 1), back_knee=round(back_knee_angle, 1),
                       hip=round(hip_angle, 1), back=round(back_angle, 1))
            frame_rows.append(row)

            if present(disp, show):
                break

    cap.release()
    write_csv(os.path.join(out_dir, f"{stem}_frames.csv"), frame_rows, FRAME_FIELDS)
    write_csv(os.path.join(out_dir, f"{stem}_reps.csv"), rep_rows, REP_FIELDS)
    avg_acc = sum(accuracies) / len(accuracies) if accuracies else 0.0
    print(f"{video_name}: {rep_id} attempts, {reps} correct, avg accuracy {avg_acc:.1f}%")
    return {"reps": reps, "attempts": rep_id, "avg_accuracy": avg_acc,
            "left": left_reps, "right": right_reps, "rep_rows": rep_rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="video file or folder of videos")
    ap.add_argument("--out", default="outputs_lunge")
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
        show_summary(last["reps"], last["attempts"], last["avg_accuracy"], last["left"], last["right"])


if __name__ == "__main__":
    main()