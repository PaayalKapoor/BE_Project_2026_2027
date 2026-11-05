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
VIDEO_PATH = r"C:\Users\ADMIN\Documents\GitHub\BE_Project_2026_2027\videos\Supine_Heel_Slides\Sayalee_SHS_R_2.mp4"

LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX = 31, 32

STAND_KNEE = 165        #knee angle above this = standing
BOTTOM_KNEE = 95        #knee angle below this = bottom phase (S3)
PARTIAL_MIN_KNEE = 130  #a rep that never goes below this is ignored (just a wobble)
ALPHA = 0.3             #smoothing weight of previous value
VIS_MIN = 0.6
ERR_MIN_FRAMES = 5      #an error must persist this many S3 frames to flag the rep 

window_name = "PowerPulse - Squat Analyzer"
summary_window_name = "Squat Summary"
stop_flag = False
stop_button_rect = [0, 0, 0, 0]
summary_close_flag = False
summary_close_rect = (200, 130, 310, 175)

REP_FIELDS = ["video", "rep_id", "start_s", "end_s", "label", "errors", "accuracy_pct", "reached_depth"]
FRAME_FIELDS = ["frame", "time_s", "detected", "state", "feedback", "errors", "knee", "hip", "back", "ankle"]

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
    stop_button_rect[:] = get_stop_button_rect(disp)   
    draw_stop_button(disp)
    cv2.imshow(window_name, disp)
    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), 27):
        return True
    if key == ord(' '):    #space = pause / resume
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


def show_summary(reps, avg_accuracy):
    global summary_close_flag
    summary_close_flag = False
    cv2.namedWindow(summary_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(summary_window_name, 340, 220)
    cv2.setMouseCallback(summary_window_name, on_summary_mouse, summary_close_rect)
    while True:
        cv2.imshow(summary_window_name, make_summary_image(reps, avg_accuracy))
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
    cosine = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def calculate_backlean(shoulder_x, shoulder_y, hip_x, hip_y):
    """Angle of hip->shoulder vector from vertical; sign depends on lean direction."""
    dx, dy = shoulder_x - hip_x, shoulder_y - hip_y
    lean = float(np.degrees(np.arctan2(abs(dx), -dy)))   #image y points down, so up = -dy
    return lean if shoulder_x < hip_x else -lean


def is_side_view(lm):
    width = abs(lm[LEFT_SHOULDER].x - lm[RIGHT_SHOULDER].x)
    if width > 0.25:
        return False, "Please Turn Sideways to the Camera"
    return True, "Good Position"


def s3_errors(back, knee, hip, ankle, knee_over_toe):
    """All bottom-phase errors firing this frame, in priority order: (code, message)."""
    errs = []
    if abs(back) > 45:            errs.append(("LEAN_FORWARD", "Lean Less Forward"))
    if knee < 70:                 errs.append(("TOO_DEEP", "Too Deep - Control The Squat"))
    if hip < 50:                  errs.append(("HIP_FLEXION", "Too Much Hip Flexion"))
    if ankle < 70:                errs.append(("ANKLE_MOBILITY", "Limited Mobility - Fix Posture"))
    if abs(knee_over_toe) > 0.08: errs.append(("KNEES_FORWARD", "Knees Too Far Forward"))
    if hip > 90:                  errs.append(("HIPS_BACK", "Push Hips Back More"))
    if ankle > 100:               errs.append(("WEIGHT_BACK", "Weight Shifting Backward"))
    return errs


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)


def process_video(video_path, show=True, out_dir="outputs"):
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
    prev = None                      #previous smoothed (knee, hip, back, ankle)
    reps = 0                         #reps with no flagged error
    rep_id = 0
    in_rep = reached_depth = False
    rep_start_t = 0.0
    min_knee = 999.0
    frame_scores, err_counts = [], {}
    accuracies, last_acc = [], 0.0
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
            row = {"frame": this_frame, "time_s": round(t_s, 3), "detected": 0, "state": state,
                   "feedback": "", "errors": "", "knee": "", "hip": "", "back": "", "ankle": ""}

            
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

            idx = ((LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE, LEFT_FOOT_INDEX) if left_vis > right_vis
                   else (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE, RIGHT_FOOT_INDEX))
            shoulder, hip, knee, ankle, foot = (lm[i] for i in idx)

            sx, sy, hx, hy = shoulder.x * w, shoulder.y * h, hip.x * w, hip.y * h
            kx, ky, ax, ay = knee.x * w, knee.y * h, ankle.x * w, ankle.y * h
            fx, fy = foot.x * w, foot.y * h

            raw = (calculate_jointangles(hx, hy, kx, ky, ax, ay),       #knee
                   calculate_jointangles(sx, sy, hx, hy, kx, ky),       #hip
                   calculate_backlean(sx, sy, hx, hy),                  #back lean
                   calculate_jointangles(kx, ky, ax, ay, fx, fy))       #ankle
            if any(np.isnan(v) for v in raw):
                row["feedback"] = "NaN angle"
                frame_rows.append(row)
                if present(disp, show):
                    break
                continue

            if prev is None:
                prev = raw
            knee_angle, hip_angle, back_angle, ankle_angle = (
                ALPHA * p + (1 - ALPHA) * r for p, r in zip(prev, raw))
            going_down = knee_angle < prev[0]
            prev = (knee_angle, hip_angle, back_angle, ankle_angle)
            knee_over_toe = knee.x - ankle.x

            
            if knee_angle > STAND_KNEE:
                if in_rep and (reached_depth or min_knee < PARTIAL_MIN_KNEE):
                    rep_id += 1
                    errs = sorted(c for c, n in err_counts.items() if n >= ERR_MIN_FRAMES)
                    if not reached_depth:
                        errs = ["SHALLOW"]
                    acc = (sum(frame_scores) / len(frame_scores) * 100) if frame_scores else 0.0
                    if not errs:
                        reps += 1
                    if reached_depth:
                        accuracies.append(acc)
                        last_acc = acc
                    rep_rows.append({"video": video_name, "rep_id": rep_id,
                                     "start_s": round(rep_start_t, 3), "end_s": round(t_s, 3),
                                     "label": "incorrect" if errs else "correct",
                                     "errors": ";".join(errs), "accuracy_pct": round(acc, 1),
                                     "reached_depth": int(reached_depth)})
                in_rep = False
                state = "S1"
            else:
                if not in_rep:                                
                    in_rep, reached_depth, min_knee = True, False, 999.0
                    rep_start_t = t_s
                    frame_scores, err_counts = [], {}
                min_knee = min(min_knee, knee_angle)
                if knee_angle < BOTTOM_KNEE:
                    state, reached_depth = "S3", True
                else:
                    state = "S2" if going_down else "S4"

            
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
                ks = max(0, 1 - abs(knee_angle - 85) / 40)
                hs = max(0, 1 - abs(hip_angle - 80) / 40)
                bs = max(0, 1 - abs(back_angle) / 45)
                as_ = max(0, 1 - abs(ankle_angle - 80) / 30)
                frame_scores.append(ks * 0.35 + bs * 0.30 + hs * 0.20 + as_ * 0.15)

                errs_now = s3_errors(back_angle, knee_angle, hip_angle, ankle_angle, knee_over_toe)
                for code, _ in errs_now:
                    err_counts[code] = err_counts.get(code, 0) + 1
                frame_errs = [c for c, _ in errs_now]
                if errs_now:
                    feedback, level = errs_now[0][1], "error"
                elif 30 <= abs(back_angle) <= 45:
                    feedback = "Good Torso Posture"
                elif 60 < knee_angle < 90:
                    feedback = "Good Squat Depth"
                elif 50 <= hip_angle <= 90:
                    feedback = "Good Hip Engagement"
                elif 70 <= ankle_angle <= 100:
                    feedback = "Good Posture"
                else:
                    feedback = "Good Form - Hold Position"

            elif state == "S4":
                if abs(back_angle) > 40:
                    feedback, level = "Keep Back Straight While Rising", "warn"
                else:
                    feedback = "Good - Drive Through Heels"

            color = {"ok": (0, 255, 0), "warn": (0, 165, 255), "error": (0, 0, 255)}[level]

            
            pts = [(int(p.x * dw), int(p.y * dh)) for p in (shoulder, hip, knee, ankle, foot)]
            cv2.polylines(disp, [np.array(pts, np.int32)], False, (255, 200, 0), 3)
            for p in pts:
                cv2.circle(disp, p, 6, (0, 255, 255), -1)
            cv2.putText(disp, feedback, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(disp, "State: " + state, (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(disp, f"Correct reps: {reps}  (attempts: {rep_id})", (50, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            if accuracies:
                cv2.putText(disp, f"Last Rep: {last_acc:.1f}%", (50, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(disp, f"Session Avg: {sum(accuracies) / len(accuracies):.1f}%", (50, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            row.update(detected=1, state=state, feedback=feedback, errors=";".join(frame_errs),
                       knee=round(knee_angle, 1), hip=round(hip_angle, 1),
                       back=round(back_angle, 1), ankle=round(ankle_angle, 1))
            frame_rows.append(row)

            if present(disp, show):
                break

    cap.release()
    write_csv(os.path.join(out_dir, f"{stem}_frames.csv"), frame_rows, FRAME_FIELDS)
    write_csv(os.path.join(out_dir, f"{stem}_reps.csv"), rep_rows, REP_FIELDS)
    avg_acc = sum(accuracies) / len(accuracies) if accuracies else 0.0
    print(f"{video_name}: {rep_id} attempts, {reps} correct, avg accuracy {avg_acc:.1f}%")
    return {"reps": reps, "attempts": rep_id, "avg_accuracy": avg_acc, "rep_rows": rep_rows}


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
        last = process_video(v, show=show, out_dir=args.out)
        all_rows += last["rep_rows"]
        if stop_flag:
            break

    os.makedirs(args.out, exist_ok=True)
    write_csv(os.path.join(args.out, "all_reps_predicted.csv"), all_rows, REP_FIELDS)
    cv2.destroyAllWindows()
    if show and len(videos) == 1 and last:
        show_summary(last["reps"], last["avg_accuracy"])


if __name__ == "__main__":
    main()