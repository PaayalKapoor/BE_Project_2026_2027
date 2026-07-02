import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import cv2
import pandas as pd
import numpy as np
from scipy.signal import find_peaks, savgol_filter

VIDEO_PATH = "D:\Sayalee\Major_project\Dataset_Videos\Patient118.mp4"
SIDE = "right"

# ── Rep detection constants ──────────────────────────────────────────────────
REP_MIN_PROMINENCE = 15   # minimum degrees of flexion to count as a real rep
REP_MIN_DISTANCE   = 20   # minimum frames between valleys

# ── Savitzky-Golay filter constants ─────────────────────────────────────────
SG_WINDOW = 7    # must be odd. increase for more smoothing, decrease to preserve sharp changes
SG_ORDER  = 2    # polynomial order. 2 is standard for biological motion

LANDMARK_INDICES = {
    "right": dict(shoulder=12, hip=24, knee=26, ankle=28, foot=32),
    "left":  dict(shoulder=11, hip=23, knee=25, ankle=27, foot=31),
}

# ── Opposite side landmarks needed for pelvis hiking calculation ─────────────
OPPOSITE_HIP = {"right": 23, "left": 24}


# ─────────────────────────────────────────────────────────────────────────────
# EXISTING FUNCTIONS (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def extract_landmarks(video_path, side):
    base_options = python.BaseOptions(
        model_asset_path="D:\Sayalee\Major_project\models\pose_landmarker_heavy.task"
    )
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.6,
        min_tracking_confidence=0.5,
        output_segmentation_masks=True
    )

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    lms_indices      = LANDMARK_INDICES[side]
    opposite_hip_idx = OPPOSITE_HIP[side]
    all_frames_data       = []
    all_frames_data_world = []

    with vision.PoseLandmarker.create_from_options(options) as detector:
        frame_number = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            height, width, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(frame_number * 1000 / fps)
            result = detector.detect_for_video(mp_image, timestamp_ms)

            frame_data       = {"frame": frame_number, "detected": False}
            frame_data_world = {"frame": frame_number, "detected": False}

            if result.pose_landmarks:
                landmarks       = result.pose_landmarks[0]
                world_landmarks = result.pose_world_landmarks[0]

                pixel_lms = []
                for lm in landmarks:
                    pixel_lms.append((int(lm.x * width), int(lm.y * height)))

                cv2.line(frame, pixel_lms[24], pixel_lms[26], (255, 0, 0), 2)
                cv2.line(frame, pixel_lms[26], pixel_lms[28], (255, 0, 0), 2)
                cv2.line(frame, pixel_lms[28], pixel_lms[32], (255, 0, 0), 2)
                cv2.line(frame, pixel_lms[12], pixel_lms[24], (255, 0, 0), 2)

                frame_data["detected"]       = True
                frame_data_world["detected"] = True

                for joint_name, joint_index in lms_indices.items():
                    lm  = landmarks[joint_index]
                    lmw = world_landmarks[joint_index]

                    frame_data[f"{joint_name}_x"]          = lm.x
                    frame_data[f"{joint_name}_y"]          = lm.y
                    frame_data[f"{joint_name}_z"]          = lm.z
                    frame_data[f"{joint_name}_visibility"] = lm.visibility

                    frame_data_world[f"{joint_name}_x"] = lmw.x
                    frame_data_world[f"{joint_name}_y"] = lmw.y
                    frame_data_world[f"{joint_name}_z"] = lmw.z

                    x_px = int(lm.x * width)
                    y_px = int(lm.y * height)
                    cv2.circle(frame, (x_px, y_px), 5, (0, 255, 0), -1)
                    text = f"{joint_name} X:{lmw.x:.2f} Y:{lmw.y:.2f} Vis:{lm.visibility:.2f}"
                    cv2.putText(frame, text, (x_px + 10, y_px),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                # Also store opposite hip y for pelvic hiking (normalised coords)
                opp_lm = landmarks[opposite_hip_idx]
                frame_data["opposite_hip_y"] = opp_lm.y
                frame_data_world["opposite_hip_y"] = world_landmarks[opposite_hip_idx].y

            else:
                for joint_name in lms_indices.keys():
                    frame_data[f"{joint_name}_x"]          = None
                    frame_data[f"{joint_name}_y"]          = None
                    frame_data[f"{joint_name}_z"]          = None
                    frame_data[f"{joint_name}_visibility"] = None

                    frame_data_world[f"{joint_name}_x"] = None
                    frame_data_world[f"{joint_name}_y"] = None
                    frame_data_world[f"{joint_name}_z"] = None

                frame_data["opposite_hip_y"]       = None
                frame_data_world["opposite_hip_y"] = None

            all_frames_data.append(frame_data)
            all_frames_data_world.append(frame_data_world)

            cv2.imshow("Pose_Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_number += 1

    cap.release()
    cv2.destroyAllWindows()

    df  = pd.DataFrame(all_frames_data).set_index("frame")
    dfw = pd.DataFrame(all_frames_data_world).set_index("frame")
    print(df.head())
    print(dfw.head())
    return df, dfw


def compute_angle(a, b, c):
    """Angle at joint B given three anatomical points A-B-C. Returns degrees."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine = np.clip(cosine, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — CALCULATE ANGLES
# ─────────────────────────────────────────────────────────────────────────────

def calculate_angles(lm_df, lm_df_world):
    """
    Computes per-frame joint angles from the landmark dataframe.

    Angles computed
    ───────────────
    knee_flexion_angle   : angle at knee between hip-knee-ankle  (world landmarks)
    hip_flexion_angle    : angle at hip between shoulder-hip-knee (world landmarks)
    ankle_angle          : angle at ankle between knee-ankle-foot (world landmarks)
    pelvic_hike_norm     : lateral pelvic tilt normalised by torso length
                           = (exercising_hip_y - opposite_hip_y) / torso_length
                           positive = exercising side is LOWER (hiking upward means
                           exercising hip rises = its y in image coords decreases,
                           so the difference becomes negative — sign tells direction)
                           Uses normalised image coords (lm_df) not world coords
                           because vertical pixel position is more stable than
                           world y for a supine patient from a side camera

    Returns a DataFrame indexed by frame with one column per angle.
    Frames where any required landmark was missing get NaN.
    """

    angle_rows = []

    for frame_idx, row in lm_df.iterrows():
        # Skip frames where pose was not detected
        if not row.get("detected", False):
            angle_rows.append({
                "frame":               frame_idx,
                "knee_flexion_angle":  np.nan,
                "hip_flexion_angle":   np.nan,
                "ankle_angle":         np.nan,
                "pelvic_hike_norm":    np.nan,
            })
            continue

        world_row = lm_df_world.loc[frame_idx]

        # ── Pull world-coordinate 3D points for angle computation ────────────
        # Using world landmarks for angles gives physically meaningful geometry
        # because world coords are in metres relative to the hip centre,
        # removing perspective distortion that affects normalised image coords.

        def wpt(joint):
            """Returns (x, y, z) world coordinates for a joint name."""
            return (
                world_row[f"{joint}_x"],
                world_row[f"{joint}_y"],
                world_row[f"{joint}_z"],
            )

        shoulder = wpt("shoulder")
        hip      = wpt("hip")
        knee     = wpt("knee")
        ankle    = wpt("ankle")
        foot     = wpt("foot")

        # Check any coordinate is None (missing landmark)
        all_pts = [shoulder, hip, knee, ankle, foot]
        if any(None in pt or any(np.isnan(v) for v in pt if v is not None)
               for pt in all_pts):
            angle_rows.append({
                "frame":               frame_idx,
                "knee_flexion_angle":  np.nan,
                "hip_flexion_angle":   np.nan,
                "ankle_angle":         np.nan,
                "pelvic_hike_norm":    np.nan,
            })
            continue

        # ── Knee flexion angle ───────────────────────────────────────────────
        # Angle at knee between thigh segment (hip→knee) and shin segment (ankle→knee)
        # Full extension ≈ 170–180°. Flexion reduces this angle.
        knee_angle = compute_angle(hip, knee, ankle)

        # ── Hip flexion angle ────────────────────────────────────────────────
        # Angle at hip between trunk segment (shoulder→hip) and thigh segment (knee→hip)
        # In supine with leg flat, hip angle ≈ 180°.
        # As thigh lifts off surface (compensation), this angle decreases.
        hip_angle = compute_angle(shoulder, hip, knee)

        # ── Ankle angle ──────────────────────────────────────────────────────
        # Angle at ankle between shin segment (knee→ankle) and foot segment (foot→ankle)
        # Passive in heel slides but tracks heel lift compensation
        ankle_angle = compute_angle(knee, ankle, foot)

        # ── Pelvic hiking (lateral tilt) ─────────────────────────────────────
        # Uses normalised IMAGE y-coordinates (lm_df), not world coords.
        # In image space, y=0 is top of frame, y=1 is bottom.
        # For a supine patient viewed from the side:
        #   exercising_hip_y  = y of the hip on the exercising side
        #   opposite_hip_y    = y of the hip on the other side
        # When the pelvis tilts laterally (hiking), one hip rises relative to the other.
        # Rising means y_image DECREASES (moving toward top of frame).
        #
        # Normalisation by torso length makes this comparable across patients
        # of different heights and different camera distances.
        # Torso length proxy = Euclidean distance between shoulder and hip
        # in normalised image coords.
        #
        # pelvic_hike_norm = (exercising_hip_y - opposite_hip_y) / torso_length
        # Near zero = pelvis level. Large positive = exercising hip lower.
        # Large negative = exercising hip higher = true pelvic hiking compensation.

        ex_hip_y  = row["hip_y"]           # exercising side hip, normalised image y
        opp_hip_y = row.get("opposite_hip_y", np.nan)

        if opp_hip_y is None or np.isnan(float(opp_hip_y if opp_hip_y is not None else np.nan)):
            pelvic_hike_norm = np.nan
        else:
            # Torso length in normalised image coordinates
            sh_x = row["shoulder_x"]
            sh_y = row["shoulder_y"]
            hi_x = row["hip_x"]
            hi_y = row["hip_y"]

            if any(v is None for v in [sh_x, sh_y, hi_x, hi_y]):
                pelvic_hike_norm = np.nan
            else:
                torso_length = np.sqrt((sh_x - hi_x)**2 + (sh_y - hi_y)**2)
                if torso_length < 1e-6:
                    pelvic_hike_norm = np.nan
                else:
                    pelvic_hike_norm = (float(ex_hip_y) - float(opp_hip_y)) / torso_length

        angle_rows.append({
            "frame":               frame_idx,
            "knee_flexion_angle":  knee_angle,
            "hip_flexion_angle":   hip_angle,
            "ankle_angle":         ankle_angle,
            "pelvic_hike_norm":    pelvic_hike_norm,
        })

    angle_df = pd.DataFrame(angle_rows).set_index("frame")
    print(f"[calculate_angles] computed {len(angle_df)} frames, "
          f"{angle_df.isnull().any(axis=1).sum()} frames with NaN")
    return angle_df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — SAVITZKY-GOLAY SMOOTHING
# ─────────────────────────────────────────────────────────────────────────────

def smooth_angles(angle_df, window=SG_WINDOW, order=SG_ORDER):
    """
    Applies Savitzky-Golay filter to each angle column independently.

    Why smooth before computing derivatives
    ───────────────────────────────────────
    Velocity = angle[t] - angle[t-1]. If the raw angle has even small jitter,
    the difference amplifies that noise dramatically. Smoothing first means
    the derivatives reflect genuine movement, not measurement noise.

    NaN handling
    ────────────
    SG filter cannot handle NaN. We interpolate missing values linearly
    before filtering, then restore the original NaN positions afterward
    so downstream code knows which frames were genuinely unreliable.

    Returns a new DataFrame with the same columns, smoothed values.
    """
    angle_cols = ["knee_flexion_angle", "hip_flexion_angle",
                  "ankle_angle", "pelvic_hike_norm"]

    smoothed_df = angle_df.copy()

    for col in angle_cols:
        series = angle_df[col].copy()

        # Track which positions were originally NaN
        nan_mask = series.isna()

        # Interpolate NaNs so filter has a complete signal to work on
        series_filled = series.interpolate(method="linear", limit_direction="both")

        # If the whole column is NaN (e.g. pelvic hike not computable), skip
        if series_filled.isna().all():
            continue

        # Apply Savitzky-Golay filter
        # Ensure window length does not exceed signal length
        effective_window = min(window, len(series_filled))
        # Window must be odd
        if effective_window % 2 == 0:
            effective_window -= 1
        # Window must be greater than polynomial order
        if effective_window <= order:
            effective_window = order + 1
            if effective_window % 2 == 0:
                effective_window += 1

        smoothed_values = savgol_filter(
            series_filled.values,
            window_length=effective_window,
            polyorder=order
        )

        # Restore NaN at originally missing positions
        smoothed_series = pd.Series(smoothed_values, index=series.index)
        smoothed_series[nan_mask] = np.nan

        smoothed_df[col] = smoothed_series

    print(f"[smooth_angles] Savitzky-Golay applied (window={SG_WINDOW}, order={SG_ORDER})")
    return smoothed_df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — REP DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_reps(knee_angles: np.ndarray) -> list[dict]:
    """
    Detects rep boundaries using valley detection on the knee flexion angle.

    One rep = extension peak → flexion valley → extension peak.
    This approach finds valleys (maximum flexion points) first,
    then finds the maximum angle between consecutive valleys as boundaries.
    This is robust to patients with restricted ROM who may not return to
    the same extension angle each rep.

    Returns a list of dicts, one per rep:
        rep_id, start_frame, valley_frame, end_frame
    """
    valleys, _ = find_peaks(
        -knee_angles,
        prominence=REP_MIN_PROMINENCE,
        distance=REP_MIN_DISTANCE,
    )

    if len(valleys) == 0:
        print("  WARNING: no rep valleys found — try lowering REP_MIN_PROMINENCE")
        return []

    # Max angle before first valley = start boundary of rep 1
    boundary_peaks = [int(np.argmax(knee_angles[: valleys[0]]))]

    # Max angle between consecutive valleys = boundary between reps
    for i in range(len(valleys) - 1):
        window    = knee_angles[valleys[i]: valleys[i + 1]]
        local_max = int(np.argmax(window)) + valleys[i]
        boundary_peaks.append(local_max)

    # Max angle after last valley = end boundary of last rep
    last_window = knee_angles[valleys[-1]:]
    boundary_peaks.append(int(np.argmax(last_window)) + valleys[-1])

    reps = []
    for i, valley in enumerate(valleys):
        start     = boundary_peaks[i]
        end       = boundary_peaks[i + 1]
        rep_range = knee_angles[start] - knee_angles[valley]

        if rep_range < REP_MIN_PROMINENCE:
            print(f"  Skipping shallow valley at frame {valley} "
                  f"(range={rep_range:.1f}° < {REP_MIN_PROMINENCE}°)")
            continue

        reps.append({
            "rep_id":       i + 1,
            "start_frame":  int(start),
            "valley_frame": int(valley),
            "end_frame":    int(end),
        })

    print(f"  {len(reps)} rep(s) detected")
    return reps


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — PHASE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_phases(knee_angles_in_rep: np.ndarray, valley_frame_local: int) -> np.ndarray:
    """
    Assigns a phase label to each frame within a single rep.

    Phases
    ──────
    P1 — fully extended   : frames from rep start until significant flexion begins
    P2 — flexing          : angle decreasing toward valley
    P3 — full flexion     : frames around the valley where angle is near minimum
    P4 — extending        : angle increasing back toward full extension

    Parameters
    ──────────
    knee_angles_in_rep  : 1D array of smoothed knee angles for this rep only
    valley_frame_local  : index of the valley within this rep's local frame array
                          (not the global frame number)

    Returns
    ───────
    1D array of phase labels (strings) of same length as knee_angles_in_rep.

    Strategy
    ────────
    Instead of velocity thresholds which are sensitive to noise, we use
    the valley position as a hard anchor and split the rep into two halves:
    - Pre-valley  = flexion half (P1 → P2 → P3)
    - Post-valley = extension half (P3 → P4)

    Within the flexion half, P1 ends when the angle has dropped by more than
    a small threshold from the starting angle (patient has begun moving).
    P3 begins a few frames before the valley and ends a few frames after.
    Everything in between is P2.

    This is more robust than velocity thresholding for patients who move
    slowly or hesitate, because it is anchored to the known valley position.
    """

    n = len(knee_angles_in_rep)
    phases = np.full(n, "P2", dtype=object)   # default everything to flexing

    # How many frames around the valley count as "full flexion" P3
    # 5% of rep length on each side, minimum 2 frames
    p3_half_width = max(2, int(0.05 * n))
    p3_start = max(0, valley_frame_local - p3_half_width)
    p3_end   = min(n - 1, valley_frame_local + p3_half_width)

    # P3 — full flexion (around valley)
    phases[p3_start: p3_end + 1] = "P3"

    # P1 — fully extended (start of rep, before patient begins moving)
    # Ends when angle has dropped by more than 3 degrees from starting angle
    start_angle = knee_angles_in_rep[0]
    p1_end = 0
    for idx in range(valley_frame_local):
        if knee_angles_in_rep[idx] < start_angle - 3.0:
            break
        p1_end = idx
    phases[0: p1_end + 1] = "P1"

    # P2 — flexing: everything between P1 and P3 in the pre-valley half
    # (already set as default, nothing to do)

    # P4 — extending: everything after P3
    phases[p3_end + 1:] = "P4"

    return phases


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — TEMPORAL FEATURES (velocity, acceleration, jerk)
# ─────────────────────────────────────────────────────────────────────────────

def compute_temporal_features(smoothed_angle_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes first, second, and third derivatives of each angle column.

    velocity     = angle[t] - angle[t-1]          (degrees per frame)
    acceleration = velocity[t] - velocity[t-1]    (degrees per frame²)
    jerk         = acceleration[t] - acceleration[t-1]  (degrees per frame³)

    First frame gets 0 for velocity (no previous frame).
    First two frames get 0 for acceleration.
    First three frames get 0 for jerk.

    This is applied to the SMOOTHED angles. Never differentiate raw angles
    because differentiation amplifies noise.

    Returns a DataFrame with additional columns for each derivative.
    """
    angle_cols = ["knee_flexion_angle", "hip_flexion_angle",
                  "ankle_angle", "pelvic_hike_norm"]

    temporal_df = smoothed_angle_df.copy()

    for col in angle_cols:
        series = smoothed_angle_df[col]

        vel   = series.diff().fillna(0.0)          # first difference
        accel = vel.diff().fillna(0.0)              # second difference
        jerk  = accel.diff().fillna(0.0)            # third difference

        temporal_df[f"{col}_velocity"]     = vel
        temporal_df[f"{col}_acceleration"] = accel
        temporal_df[f"{col}_jerk"]         = jerk

    print(f"[compute_temporal_features] velocity, acceleration, jerk computed "
          f"for {len(angle_cols)} angles")
    return temporal_df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — ASSEMBLE FINAL CSV
# ─────────────────────────────────────────────────────────────────────────────

def assemble_features(temporal_df: pd.DataFrame, reps: list[dict]) -> pd.DataFrame:
    """
    Combines all computed features into one row per frame, annotated with
    rep number, phase label, phase duration, and rep duration.

    Frames that do not belong to any detected rep are excluded.
    This is intentional — frames outside rep boundaries (rest between sets,
    calibration frames) should not be in the training data.

    Output columns
    ──────────────
    frame                       : frame index (also used as sort key)
    rep_id                      : which rep this frame belongs to
    rep_duration_frames         : total frame count of this rep
    phase                       : P1, P2, P3, or P4
    phase_duration_frames       : how many frames have elapsed in current phase
                                  at this frame (cumulative count within phase)
    knee_flexion_angle          : smoothed knee angle (degrees)
    hip_flexion_angle           : smoothed hip angle (degrees)
    ankle_angle                 : smoothed ankle angle (degrees)
    pelvic_hike_norm            : normalised lateral pelvic tilt
    [all angle]_velocity        : degrees per frame
    [all angle]_acceleration    : degrees per frame²
    [all angle]_jerk            : degrees per frame³

    Rows are sorted first by rep_id then by frame index within each rep.
    This ensures the CSV preserves temporal order within reps, which is
    essential for LSTM training.
    """

    all_rep_rows = []

    for rep in reps:
        rep_id       = rep["rep_id"]
        start_frame  = rep["start_frame"]
        end_frame    = rep["end_frame"]
        valley_frame = rep["valley_frame"]

        # Extract the slice of temporal_df for this rep
        rep_slice = temporal_df.loc[
            (temporal_df.index >= start_frame) &
            (temporal_df.index <= end_frame)
        ].copy()

        if rep_slice.empty:
            print(f"  WARNING: rep {rep_id} has no data between frames "
                  f"{start_frame} and {end_frame}")
            continue

        rep_duration = len(rep_slice)

        # Valley position in local (0-indexed) coordinates within this rep
        valley_local = valley_frame - start_frame
        valley_local = np.clip(valley_local, 0, rep_duration - 1)

        # Get smoothed knee angles for this rep as a numpy array for phase detection
        knee_col = rep_slice["knee_flexion_angle"].values

        # Handle NaN in knee angle by forward filling for phase detection only
        # (NaN values in the actual feature columns are preserved)
        knee_for_phase = pd.Series(knee_col).interpolate(
            method="linear", limit_direction="both"
        ).values

        # Assign phase labels
        phase_labels = detect_phases(knee_for_phase, valley_local)

        rep_slice = rep_slice.copy()
        rep_slice["rep_id"]             = rep_id
        rep_slice["rep_duration_frames"] = rep_duration
        rep_slice["phase"]              = phase_labels

        # Phase duration: cumulative count of frames in the current phase
        # Reset counter each time phase changes
        phase_duration = np.zeros(rep_duration, dtype=int)
        current_phase  = phase_labels[0]
        count          = 1
        phase_duration[0] = count
        for i in range(1, rep_duration):
            if phase_labels[i] == current_phase:
                count += 1
            else:
                current_phase = phase_labels[i]
                count = 1
            phase_duration[i] = count
        rep_slice["phase_duration_frames"] = phase_duration

        all_rep_rows.append(rep_slice)

    if not all_rep_rows:
        print("  ERROR: no rep data to assemble. Check rep detection parameters.")
        return pd.DataFrame()

    # Concatenate all reps in rep order
    final_df = pd.concat(all_rep_rows)

    # Reorder columns for readability
    id_cols    = ["rep_id", "rep_duration_frames", "phase", "phase_duration_frames"]
    angle_cols = ["knee_flexion_angle", "hip_flexion_angle",
                  "ankle_angle", "pelvic_hike_norm"]
    deriv_cols = [f"{a}_{d}"
                  for a in angle_cols
                  for d in ["velocity", "acceleration", "jerk"]]

    ordered_cols = id_cols + angle_cols + deriv_cols
    # Keep only columns that actually exist in the dataframe
    ordered_cols = [c for c in ordered_cols if c in final_df.columns]
    final_df = final_df[ordered_cols]

    print(f"[assemble_features] {len(final_df)} total frames across "
          f"{len(all_rep_rows)} reps assembled")
    return final_df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(video_path, side, output_csv_path=None):
    """
    Runs the full feature extraction pipeline for one video.

    Returns the final feature DataFrame and saves it to CSV if path provided.
    """

    print("\n=== STEP 1-2: Landmark Extraction ===")
    lm_df, lm_df_world = extract_landmarks(video_path, side)

    print("\n=== STEP 3: Angle Calculation ===")
    angle_df = calculate_angles(lm_df, lm_df_world)

    print("\n=== STEP 4: Smoothing ===")
    smoothed_df = smooth_angles(angle_df)

    print("\n=== STEP 5: Rep Detection ===")
    # Use only frames where knee angle is not NaN for rep detection
    knee_series = smoothed_df["knee_flexion_angle"].interpolate(
        method="linear", limit_direction="both"
    )
    reps = detect_reps(knee_series.values)

    if not reps:
        print("No reps detected. Check video or lower REP_MIN_PROMINENCE.")
        return pd.DataFrame()

    print("\n=== STEP 6-7: Temporal Features ===")
    temporal_df = compute_temporal_features(smoothed_df)

    print("\n=== STEP 8: Assembling Final CSV ===")
    final_df = assemble_features(temporal_df, reps)

    if output_csv_path and not final_df.empty:
        final_df.to_csv(output_csv_path)
        print(f"\nSaved to {output_csv_path}")
        print(f"Shape: {final_df.shape}")
        print(f"\nFirst few rows:")
        print(final_df.head(10).to_string())

    return final_df


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    output_path = "D:\Sayalee\Major_project\Features\Supine_heel_slides_features.csv"
    final_df = run_pipeline(VIDEO_PATH, SIDE, output_csv_path=output_path)