import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy.signal import savgol_filter, find_peaks
import matplotlib.pyplot as plt

# ============================== CONFIG ==============================
VIDEO_PATH = r"C:\Users\sayal\Downloads\Participant_102_BIC_L.mp4"
MODEL_PATH = r"D:\Sayalee\Projects\Major_project\models\pose_landmarker_heavy.task"
OUTPUT_CSV = r"D:\Sayalee\Projects\Major_project\Features\bicepcurls.csv"
SIDE = "left"                       # the arm/side facing the camera

# --- filtering ---
SG_WINDOW = 11
SG_POLY = 3

# --- rep detection / states ---
REP_PROMINENCE_DEG = 40              # min depth of an elbow-angle valley to count as a rep
MIN_REP_SEC = 0.8                    # min time between two valleys
STATE_FRAC = 0.15                    # S1/S3 margin as fraction of rep range
MIN_MARGIN_DEG = 5                   # floor on that margin
MIN_REP_RANGE = 40                   # below this elbow range the rep is INVALID
BASELINE_FRAMES = 5                  # first frames of a rep used as baseline

# --- gates ---
MIN_VISIBILITY = 0.5                 # mean visibility of key landmarks in a rep
SIDE_VIEW_MAX_SHOULDER_GAP = 0.12    # |left_sh_x - right_sh_x| (height units); larger = not a side view

# --- rule thresholds (PLACEHOLDERS: tune from recorded reps) ---
START_LEAN_MAX = 20                  # S1: absolute torso angle from vertical (deg)
START_DRIFT_MAX = 25                 # S1: absolute upper-arm angle vs torso (deg)
LEAN_BACK_MAX = 12                   # S2/S3/S4: backward lean change from rep start (deg)
LEAN_FWD_MAX = 20                    # S2/S3: forward lean change from rep start (deg)
HIP_SHIFT_MAX = 0.12                 # S2: forward hip shift / torso length
DRIFT_MAX = 20                       # S2/S3: forward elbow drift change from rep start (deg)
SHRUG_MAX = 0.08                     # S2/S3: shoulder rise / torso length
DROP_VEL_MAX = 250                   # S4: elbow extension speed (deg/s)
ROM_TOP_MAX = 60                     # min elbow angle in S3 must get below this
ROM_BOTTOM_MIN = 150                 # max elbow angle in S1 must get above this

# --- persistence (ms a violation must last before it counts) ---
PERSIST_MS_DEFAULT = 500
PERSIST_MS = {
    "shrug": 300,
    "drop_weight": 150,
}

LANDMARK_INDICES = {
    "right": dict(shoulder=12, elbow=14, wrist=16, hip=24, knee=26, ankle=28, foot=32),
    "left":  dict(shoulder=11, elbow=13, wrist=15, hip=23, knee=25, ankle=27, foot=31),
}
OTHER_SHOULDER = {"right": 11, "left": 12}


# ======================= 1. LANDMARK EXTRACTION =======================
def extract_landmarks(video_path, side):
    print("Extracting landmarks")
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    lms = LANDMARK_INDICES[side]
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    rows = []

    with vision.PoseLandmarker.create_from_options(options) as detector:
        frame_number = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            h, w, _ = frame.shape
            aspect = w / h
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect_for_video(mp_image, int(frame_number * 1000 / fps))

            row = {"frame": frame_number}
            if result.pose_landmarks:
                p = result.pose_landmarks[0]
                for name, idx in lms.items():
                    # x is multiplied by aspect so x and y share the same scale (angles stay true)
                    row[f"{name}_x"] = p[idx].x * aspect
                    row[f"{name}_y"] = p[idx].y
                    row[f"{name}_vis"] = p[idx].visibility
                row["other_shoulder_x"] = p[OTHER_SHOULDER[side]].x * aspect
            frame_number += 1
            rows.append(row)
    cap.release()

    df = pd.DataFrame(rows).set_index("frame")
    vis_cols = [c for c in df.columns if c.endswith("_vis")]
    coord_cols = [c for c in df.columns if c not in vis_cols]
    df[coord_cols] = df[coord_cols].interpolate(method="linear").ffill().bfill()
    df[vis_cols] = df[vis_cols].fillna(0.0)      # undetected frames count as not visible
    print(f"Extracted {len(df)} frames at {fps:.1f} fps")
    return df, fps


def smooth_coords(df):
    print("Smoothening Coordinates")
    out = df.copy()
    if len(df) < SG_WINDOW:
        return out
    for col in df.columns:
        if col.endswith("_vis"):
            continue
        out[col] = savgol_filter(df[col].values, SG_WINDOW, SG_POLY, mode="interp")
    return out


def is_side_view(df):
    gap = np.median(np.abs(df["shoulder_x"] - df["other_shoulder_x"]))
    return gap <= SIDE_VIEW_MAX_SHOULDER_GAP, gap


# ======================= 2. FRAME-LEVEL FEATURES =======================
def angle_at(a, b, c):
    ba, bc = a - b, c - b
    cos = (ba * bc).sum(axis=1) / (np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def compute_features(c, fps):
    print("Computing features")
    pt = lambda n: np.c_[c[f"{n}_x"].values, c[f"{n}_y"].values]
    sh, el, wr, hip, ank = pt("shoulder"), pt("elbow"), pt("wrist"), pt("hip"), pt("ankle")

    # +1 if the person faces +x in the image, -1 if -x (toe is ahead of the ankle)
    fwd = 1.0 if np.median(c["foot_x"].values - c["ankle_x"].values) > 0 else -1.0
    torso_len = np.linalg.norm(sh - hip, axis=1) + 1e-8

    f = pd.DataFrame(index=c.index)
    f["elbow_angle"] = angle_at(sh, el, wr)

    # torso angle from vertical: + = leaning forward, - = leaning back
    f["torso_angle"] = np.degrees(np.arctan2((sh[:, 0] - hip[:, 0]) * fwd, hip[:, 1] - sh[:, 1]))

    # upper arm angle relative to the torso line: + = elbow forward of torso
    ua = np.degrees(np.arctan2((el[:, 0] - sh[:, 0]) * fwd, el[:, 1] - sh[:, 1]))
    f["upper_arm_rel"] = ua + f["torso_angle"]

    f["hip_shift"] = (hip[:, 0] - ank[:, 0]) * fwd / torso_len     # + = hip forward of ankle
    f["shoulder_up"] = -sh[:, 1] / torso_len                       # + = shoulder higher

    f["elbow_vel"] = savgol_filter(f["elbow_angle"].values, SG_WINDOW, SG_POLY,
                                   deriv=1, delta=1 / fps, mode="interp")   # + = extending (lowering)

    f["vis_min"] = c[["shoulder_vis", "elbow_vis", "wrist_vis", "hip_vis"]].min(axis=1)
    return f


# ======================= 3. REP DETECTION =======================
def detect_rep(f, fps):
    print("Detecting repetitions")
    sig = f["elbow_angle"].values
    valleys, _ = find_peaks(-sig, prominence=REP_PROMINENCE_DEG, distance=int(MIN_REP_SEC * fps))
    reps = []
    for i, v in enumerate(valleys):
        left_start = 0 if i == 0 else valleys[i - 1]
        right_end = len(sig) if i == len(valleys) - 1 else valleys[i + 1]
        left = left_start + int(np.argmax(sig[left_start:v + 1]))
        right = v + int(np.argmax(sig[v:right_end]))
        reps.append(dict(rep_id=i + 1, rep_start=left, rep_end=right, valley=v))
    print(f"{len(reps)} reps detected")
    return reps


# ======================= 4. STATES + BASELINE DELTAS =======================
def label_reps(f, reps):
    f = f.copy()
    f["rep_id"] = 0
    f["state"] = "NONE"
    for col in ["lean_back", "lean_fwd", "drift", "hip", "shrug"]:
        f[col] = np.nan

    for rep in reps:
        s, e, v = rep["rep_start"], rep["rep_end"], rep["valley"]
        if e <= s:
            continue
        f.loc[s:e - 1, "rep_id"] = rep["rep_id"]

        # per-rep baselines (rep starts at a peak extension, so these are the "at rest" values)
        base = f.loc[s:s + BASELINE_FRAMES - 1, ["torso_angle", "upper_arm_rel", "hip_shift", "shoulder_up"]].median()
        seg = f.loc[s:e - 1]
        f.loc[s:e - 1, "lean_back"] = -(seg["torso_angle"] - base["torso_angle"])
        f.loc[s:e - 1, "lean_fwd"] = seg["torso_angle"] - base["torso_angle"]
        f.loc[s:e - 1, "drift"] = seg["upper_arm_rel"] - base["upper_arm_rel"]
        f.loc[s:e - 1, "hip"] = seg["hip_shift"] - base["hip_shift"]
        f.loc[s:e - 1, "shrug"] = seg["shoulder_up"] - base["shoulder_up"]

        # states from this rep's own range (adaptive thresholds)
        angles = f.loc[s:e, "elbow_angle"].values
        rep_max = min(angles[0], angles[-1])
        rep_min = angles.min()
        rep_range = rep_max - rep_min
        if rep_range < MIN_REP_RANGE:
            f.loc[s:e - 1, "state"] = "INVALID"
            continue
        margin = max(STATE_FRAC * rep_range, MIN_MARGIN_DEG)
        before_valley = np.arange(len(angles)) <= (v - s)
        states = np.where(angles >= rep_max - margin, "S1",
                 np.where(angles <= rep_min + margin, "S3",
                 np.where(before_valley, "S2", "S4")))
        f.loc[s:e - 1, "state"] = states[:-1]
    return f


# ======================= 5. RULES (state-wise, per frame) =======================
# S1 = extended, S2 = curling up, S3 = peak contraction, S4 = lowering
def get_frame_errors(state, r): #this function returns a dictionary, if the dict is emtpy means frame is clean
    errors = {}
    #r in one frame. in evaluate frame() r is one rep's frame 
    if state == "S1":
        if abs(r.torso_angle) > START_LEAN_MAX:
            errors["start_lean"] = "Stand upright: reset posture"
        if r.upper_arm_rel > START_DRIFT_MAX:
            errors["start_drift"] = "Bring elbows back to your sides"

    elif state == "S2":
        if r.lean_back > LEAN_BACK_MAX:
            errors["swing_lean"] = "Don't lean backward"
        if r.lean_fwd > LEAN_FWD_MAX:
            errors["lean_fwd"] = "Don't lean forward"
        if r.hip > HIP_SHIFT_MAX:
            errors["swing_hip"] = "Keep your hips still"
        if r.drift > DRIFT_MAX:
            errors["elbow_drift"] = "Pin your elbow to your side"
        if r.shrug > SHRUG_MAX:
            errors["shrug"] = "Relax your shoulders"

    elif state == "S3":
        if r.lean_back > LEAN_BACK_MAX:
            errors["swing_lean"] = "Don't rock back"
        if r.drift > DRIFT_MAX:
            errors["elbow_drift"] = "Elbow drifted forward: tuck back"
        if r.shrug > SHRUG_MAX:
            errors["shrug"] = "Relax your shoulders"

    elif state == "S4":
        if r.elbow_vel > DROP_VEL_MAX:
            errors["drop_weight"] = "Lower slowly: control the way down"
        if r.lean_back > LEAN_BACK_MAX:
            errors["swing_lean"] = "Don't rock back"

    return errors


# ======================= 6. REP-LEVEL CHECKS + VERDICT =======================
def longest_run(flags):
    best = cur = 0
    for x in flags:
        cur = cur + 1 if x else 0
        best = max(best, cur)
    return best


def check_rom(rep):
    errors = []
    s3 = rep.loc[rep["state"] == "S3", "elbow_angle"]
    s1 = rep.loc[rep["state"] == "S1", "elbow_angle"]
    if s3.empty or s3.min() > ROM_TOP_MAX:
        errors.append("Curl higher: full range of motion")
    if s1.empty or s1.max() < ROM_BOTTOM_MIN:
        errors.append("Extend your arm fully at the bottom")
    return errors


def check_sequence(rep):
    # S2 -> S3 -> S4 must first appear in this order
    first = {}
    for i, s in enumerate(rep["state"].values):
        first.setdefault(s, i)
    if not all(k in first for k in ("S2", "S3", "S4")):
        return ["Incomplete rep"]
    if not (first["S2"] < first["S3"] < first["S4"]):
        return ["Rep phases out of order"]
    return []


def evaluate_reps(f, fps, side_view_ok):
    print("Evaluating reps")
    results = []
    for rep_id, rep in f[f["rep_id"] > 0].groupby("rep_id"):
        # gates: no verdict is better than a wrong verdict
        if not side_view_ok:
            results.append(dict(rep_id=rep_id, verdict="NO_VERDICT", errors=["Camera is not a side view"]))
            continue
        if (rep["state"] == "INVALID").any() or rep["vis_min"].mean() < MIN_VISIBILITY:
            results.append(dict(rep_id=rep_id, verdict="NO_VERDICT", errors=["Arm not visible / rep not usable"]))
            continue

        # frame flags per error key
        flags, messages = {}, {}
        for r in rep.itertuples():
            frame_errors = get_frame_errors(r.state, r)
            for key, msg in frame_errors.items():
                messages[key] = msg
        for key in messages:
            flags[key] = [key in get_frame_errors(r.state, r) for r in rep.itertuples()]

        # persistence: a violation only counts if it lasts long enough
        fired = {}
        for key, flag_list in flags.items():
            min_frames = max(1, round(PERSIST_MS.get(key, PERSIST_MS_DEFAULT) * fps / 1000))
            if longest_run(flag_list) >= min_frames:
                fired[key] = messages[key]

        # merge lean + hip into one swing error so one swing is not penalised twice
        error_list = [m for k, m in fired.items() if not k.startswith("swing_")]
        swing = [k for k in fired if k.startswith("swing_")]
        if len(swing) == 2:
            error_list.append("Swinging: rocking back and driving hips")
        elif len(swing) == 1:
            error_list.append(fired[swing[0]])

        error_list += check_rom(rep)
        error_list += check_sequence(rep)

        results.append(dict(rep_id=rep_id,
                            verdict="INCORRECT" if error_list else "CORRECT",
                            errors=error_list))
    return pd.DataFrame(results)

def show_processed_video(video_path, coords, feats, results, fps, side):
    """Plays the video with the skeleton, live frame feedback and per-rep verdicts drawn on it.
    q = quit, p = pause."""
    cap = cv2.VideoCapture(video_path)
    rows = list(feats.itertuples())                       # one object per frame, same as get_frame_errors expects
    verdicts = {v.rep_id: v for v in results.itertuples()} if len(results) else {}
    end_frame = {}                                        # last frame of each rep
    for r in rows:
        if r.rep_id > 0:
            end_frame[r.rep_id] = r.Index

    bones = [("shoulder", "hip"), ("hip", "knee"), ("knee", "ankle"), ("ankle", "foot"),
             ("shoulder", "elbow"), ("elbow", "wrist")]
    joints = ["shoulder", "elbow", "wrist", "hip", "knee", "ankle", "foot"]
    GREEN, RED, GRAY, WHITE = (0, 220, 0), (0, 0, 255), (170, 170, 170), (255, 255, 255)

    def put(img, text, org, color, size, thick=2):       # text with a dark outline so it stays readable
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), thick + 3, cv2.LINE_AA)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, size, color, thick, cv2.LINE_AA)

    delay = max(1, int(1000 / fps))
    run = {}                                              # consecutive violating frames per error key
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or idx >= len(rows):
            break
        h, w = frame.shape[:2]
        sc = h / 720
        r = rows[idx]

        # ---- live frame feedback (same persistence rule as the verdict) ----
        errs = get_frame_errors(r.state, r)
        for k in list(run):
            if k not in errs:
                run[k] = 0
        for k in errs:
            run[k] = run.get(k, 0) + 1
        live = {k: m for k, m in errs.items()
                if run[k] >= max(1, round(PERSIST_MS.get(k, PERSIST_MS_DEFAULT) * fps / 1000))}
        msgs = [m for k, m in live.items() if not k.startswith("swing_")]
        swing = [k for k in live if k.startswith("swing_")]
        if len(swing) == 2:
            msgs.append("Swinging: rocking back and driving hips")
        elif swing:
            msgs.append(live[swing[0]])

        # ---- skeleton (coords x is aspect-corrected, so both axes scale by h) ----
        pts = {n: (int(coords[f"{n}_x"].iloc[idx] * h), int(coords[f"{n}_y"].iloc[idx] * h)) for n in joints}
        color = RED if msgs else GREEN
        for a, b in bones:
            cv2.line(frame, pts[a], pts[b], color, max(2, int(4 * sc)), cv2.LINE_AA)
        for n in joints:
            cv2.circle(frame, pts[n], max(3, int(6 * sc)), WHITE, -1, cv2.LINE_AA)
        put(frame, f"{r.elbow_angle:.0f}", (pts["elbow"][0] + 10, pts["elbow"][1]), WHITE, 0.7 * sc)

        # ---- top panel: current rep, state, live feedback ----
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (int(w * 0.55), int(150 * sc)), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)
        rep_txt = str(r.rep_id) if r.rep_id > 0 else "-"
        put(frame, f"Rep {rep_txt}   State: {r.state}", (15, int(35 * sc)), WHITE, 0.8 * sc)
        if r.rep_id > 0 and msgs:
            for i, m in enumerate(msgs[:3]):
                put(frame, m, (15, int((70 + 30 * i) * sc)), RED, 0.7 * sc)
        elif r.rep_id > 0:
            put(frame, "Good form", (15, int(70 * sc)), GREEN, 0.7 * sc)

        # ---- bottom panel: verdict of the last finished rep ----
        done = [rid for rid, e in end_frame.items() if idx > e]
        if done:
            rid = max(done)
            v = verdicts.get(rid)
            if v is not None:
                vcol = {"CORRECT": GREEN, "INCORRECT": RED}.get(v.verdict, GRAY)
                n_ok = sum(1 for d in done if d in verdicts and verdicts[d].verdict == "CORRECT")
                n_all = sum(1 for d in done if d in verdicts and verdicts[d].verdict != "NO_VERDICT")
                lines = v.errors[:3]
                top = h - int((60 + 28 * len(lines)) * sc)
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, top), (int(w * 0.65), h), (0, 0, 0), -1)
                frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)
                put(frame, f"Rep {rid}: {v.verdict}   ({n_ok}/{n_all} correct)", (15, top + int(32 * sc)), vcol, 0.8 * sc)
                for i, m in enumerate(lines):
                    put(frame, m, (15, top + int((62 + 28 * i) * sc)), WHITE, 0.65 * sc, 1)
                    
        display_frame= cv2.resize(frame,(1280,720))
        cv2.imshow("Bicep Curl Analyzer", cv2.resize(frame, (w // 3, h // 3)))
        key = cv2.waitKey(delay) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p"):
            cv2.waitKey(0)
        idx += 1

    cap.release()
    cv2.destroyAllWindows()



# ======================= 7. PLOT + MAIN =======================
def plot_result(f, results):
    plt.figure(figsize=(12, 5))
    plt.plot(f.index, f["elbow_angle"], label="Elbow angle")
    colors = {"CORRECT": "green", "INCORRECT": "red", "NO_VERDICT": "gray"}
    for row in results.itertuples():
        seg = f[f["rep_id"] == row.rep_id]
        if len(seg):
            plt.axvspan(seg.index[0], seg.index[-1], color=colors[row.verdict], alpha=0.15)
    plt.xlabel("Frame")
    plt.ylabel("Elbow Angle (deg)")
    plt.title("Repetition Evaluation-Bicep Curls")
    plt.legend()
    plt.show()


def process_video(video_path, side):
    raw_df, fps = extract_landmarks(video_path, side)
    coords = smooth_coords(raw_df)

    side_ok, gap = is_side_view(coords)
    print(f"Side-view check: shoulder gap {gap:.3f} -> {'OK' if side_ok else 'NOT a side view'}")

    feats = compute_features(coords, fps)
    feats["vis_min"] = coords[["shoulder_vis", "elbow_vis", "wrist_vis", "hip_vis"]].min(axis=1)

    reps = detect_rep(feats, fps)
    feats = label_reps(feats, reps)
    results = evaluate_reps(feats, fps, side_ok)
    show_processed_video(video_path, coords, feats, results, fps, side)

    print("\n=== REP RESULTS ===")
    for row in results.itertuples():
        print(f"Rep {row.rep_id}: {row.verdict}  {row.errors}")
    if len(results):
        valid = results[results["verdict"] != "NO_VERDICT"]
        if len(valid):
            print(f"\n{(valid['verdict'] == 'CORRECT').sum()}/{len(valid)} reps correct")

    feats.to_csv(OUTPUT_CSV)
    plot_result(feats, results)
    return feats, results


if __name__ == "__main__":
    process_video(VIDEO_PATH, SIDE)