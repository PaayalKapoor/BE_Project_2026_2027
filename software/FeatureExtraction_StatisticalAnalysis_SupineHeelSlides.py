#Feature Extraction Code for Supine Heel Slides: Here we extract important joint coordinates such as hip, knee, shoulder, foot and ankle per frame. These joint coordinates are smoothened using the savitzky-golay filter 
#and then these smoothed coordinates are used to find key joint angles. One of these angles functions as the key tracking angle (the angle with the most movement in degrees). This tracking angle is analyzed to 
#find the peaks and valleys in its data. The peak is the resting position (where the leg is fully extended) and valley is the flexed position of the leg (where the knee angle is minimum). The peaks and valleys in the data 
#are found using python's find_peaks() function. Once the valleys in data are found, the maximum angles before and after the valley are found which then become the rep boundary for that particular rep. Once the reps are detected
#the frames for those reps are isolated and then further divided into states. The angles for each frame of a state are used to calculate important statistics such as mean, min, max, range and standard deviation for that state. These statistics
#in addition to some values like knee ROM, hip compensation and meta data become the final features in the dataset. 
#A visibility threshold of 0.5 is set for each joint coordinate (knee, hip, shoulder, ankle and foot). This gating prevents low visibility frames from being included in calculations and corrupting data with noisy values. However, due to mediapipe's
#low training on supine positions - it by default gives low visibility for certain landmarks such as knee, ankle and foot in the near collinear position (leg fully extended). This low visibility landmark does not signify that the landmark is occluded,
#it indicates that mediapipe is not confident since this trend has been seen across different patient videos and specifically for State 4 and State 1 of the exercise (when knee angle is above 160). To avoid discarding real & valid data where the landmarks are
#valid and jitter free - we use a function called gate_low_visibility() - which basically check if the patient is in the extension phase of the exercise (near collinear position), visibility is low and the coordinate is not jittering a lot. 
#If the above 3 conditions are satisfied - then the frame is saved. The raw coordinate values are kept exactly as mediapipe gave them (unlike in case of rejected frames where the coordinate values along with the visibility parameter are nullified), and only
#the visibility score is overriden to 1.0 so that it successfully bypasses the 0.5 visibility threshold in compute_rep_features() function. This bypassing is valid only for the frames near extension with a low confidence and little to no jitter. 
#Frames that have low confidence in any other phase of the exercise are directly rejected. Their coordinate values and visibility score is set to NaN and then interpolated only so that the rep detection logic and savitzky golay filter works since 
#with gaps the algorithm would break. For computing the final features the coordinates in these frames are not used since they do not pass the visibility threshold. (visibility < 0.5). 

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy.signal import savgol_filter, find_peaks
import numpy as np
import pandas as pd
import urllib.request
import os

MODEL_PATH = "pose_landmarker_heavy.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_heavy/float16/latest/"
    "pose_landmarker_heavy.task"
)

def download_model():
    if os.path.exists(MODEL_PATH):
        print(f"Model already present: {MODEL_PATH}")
        return
    print(f"Downloading model → {MODEL_PATH}")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.")

download_model()

VIDEO_PATH = r"C:\Users\ADMIN\Documents\GitHub\BE_Project_2026_2027\videos\Supine_Heel_Slides\Patient_123_SHS_L.mp4" #Path to the input video that needs to be analyzed
OUTPUT_CSV = r"docs/supine_heel_slides_baseline.csv" #The output file that will be created to store the extracted features
PATIENT_ID = "Patient_123_SHS_L" #Unique patient ID
SIDE = "left"   #The leg that faces the camera

#Smoothing. The angles are smoothed using the savitzky golay filter. This filter basically preserves the peaks and valleys in the data. For example - moving average takes the previous, current and future reading which is averaged.
#The problem with the moving average filter is that it ends up disturbing the overall shape of the signal, eg - 180, 150, 180 here 150 will be averaged to 170. Therefore moving averages tend to flatten peaks and valleys.
#Whereas the savitzky golay filter is a smoothing filter that reduces noise in data while preserving the overall shape, peaks, valleys and trends in the signal.
SMOOTH_WINDOW = 11 #Smoothing window is set to 11, must be odd so that the middle value can be found easily. Points before and after the center value help estimate the smoothed points. For noiser signals, the value should be higher.
#Here, we use 11 neighbouring points to estimate the center value. 5 before, current and 5 after
SMOOTH_POLY = 3 #Fit a cubic polynomial through those 11 points. It uses something called Least Squares Regression. The algorithm tries many possible curves and picks the one with the smallest overall error.
#Basically the center value is smoothed using the polynomial and then the window is shifted by one position - this again changes the center and a new polynomial is used to find the smoothed value of the new center value. This process continues until all values are smoothed.
#For points at the beginning and end of the signal - different processes like mirroring of data, repeating boundary values etc. For eg - 180 176 172 169 165. Extended window: 172 176 | 180 176 172 169 165 | 169 172 (Mirroring data)
# 180 180 | 180 176 172 169 165 | 165 165 (Repeating boundary values)

#Rep detection
REP_MIN_PROMINENCE = 15 #Only bends greater than 15 degrees will be counted as valid reps. 175 degree -> 168 degree. Thats only 7 degrees (very small movement) and hence it wont be counted as a valid rep
REP_MIN_DISTANCE = 20 #min no of frames between detected repetitions to prevent false detections. For example - if the tracking angle is noisy it will lead to false rep detections.

#Used to divide each rep into states. The code determines these states based on the knee angle range of a repetition.
""" How it works - Suppose one rep has max angle = 180 (fully extended)
and min angle = 100 (knee bent)
range = 180 - 100 = 80
boundary = 0.15 x 80 = 12 degrees
Top boundary (S1): Near full extension
180 - 12 = 168
Therefore, 180 - 168 degree is classified as the S1. Therefore instead of hard coding angles which cannot be applied to each patient since the range of motion differs a lot.
Bottom boundary (S3): Near maximum flexion
100 + 12 = 112
Therefore, 100 - 112 degree is classified as S3. Everything between 112 - 168 is classified as the middle state or S2 """
#Therefore, this automatically adapts the state boundaries to each patient's movement range.
STATE_BOUNDARY_FRAC = 0.15

SIDE_LANDMARK_INDICES = {
    "right": dict(shoulder=12, hip=24, knee=26, ankle=28, foot=32, heel=30),
    "left":  dict(shoulder=11, hip=23, knee=25, ankle=27, foot=31, heel=29),
}

print("SIDE_LANDMARKS set up successfully")

def gate_low_visibility(lm_df, vis_threshold=0.5, extension_angle_threshold=160, deviation_threshold=0.04, window=5):
    #This function takes a data frame (lm_df) which stores the landmark coordinates and visibility scores. Each frame is one row of data.
    #extension_angle_threshold indicates the near collinear position where mediapipe gives low visibility scores for landmarks such as knee, ankle and foot. 
    #This low visibility score is not because of real occlusion but mediapipe's less training on supine positions. We have also set a deviation threshold, data points
    #exceeding this limit are rejected and there by not saved. If the landmark moves less than this amount, it is assumed to be stable. For interpolation in case of the rejected 
    #frames to maintain signal continuity, we calculate median using a window of 5 frames: 2 previous, current and 2 future frames. 
    out = lm_df.copy() #Here we create a copy of the data frame so that the original data frame remains intact
    hip, knee, ankle = out[["hip_x","hip_y"]].values, out[["knee_x","knee_y"]].values, out[["ankle_x","ankle_y"]].values #Read the hip, knee and ankle coordinate values so that we can calculate the knee angle using the 3 point formula. 
    provisional_angle = np.array([
        _angle_3pt(h,k,a) if not (np.isnan(h).any() or np.isnan(k).any() or np.isnan(a).any()) else np.nan
        for h,k,a in zip(hip, knee, ankle)
    ]) #Here we calculate the knee angle provided none of the landmark coordinates used in the 3 point formula are null 
    near_extension = provisional_angle >= extension_angle_threshold #If the knee angle is greater than 160 then the value is set to True

    joints = {c.rsplit("_vis",1)[0] for c in lm_df.columns if c.endswith("_vis")} #Here we find all the joints automatically. All landmarks in mediapipe are associated with three properties x & y coordinate and the visibility score. 
    for joint in joints:
        vis_col, x_col, y_col = f"{joint}_vis", f"{joint}_x", f"{joint}_y"
        low_conf = out[vis_col] < vis_threshold #Here we extract the frames that have low visibility scores for each angle 

        #A rolling median is calculated here for comparison with the current frames values for estimation of deviation from its neighboring frames over a short centered window, 
        #computed from RAW positions (no exclusion at all) — this is robust to a single bad frame without needing to guess which frames to trust ahead of time.
        med_x = out[x_col].rolling(window, center=True, min_periods=1).median()
        med_y = out[y_col].rolling(window, center=True, min_periods=1).median()
        deviation = np.sqrt((out[x_col]-med_x)**2 + (out[y_col]-med_y)**2) #Deviation from the median is calculated 
        consistent = deviation <= deviation_threshold #Here we check, if the deviation is less than the given threshold 

        near_ext_mask = near_extension if joint in ("knee","ankle", "foot", "heel") else np.zeros(len(out), dtype=bool) #The three joints that jitter a lot during extension is knee, ankle and foot. 
        rescued = low_conf & near_ext_mask & consistent #The frame will be rescued only if it has a low visibility score, is near extension and stable (not jittery)
        reject = low_conf & ~rescued #else the frame is rejected

        print(f"{joint}: low_conf={low_conf.sum()}, rescued={rescued.sum()}, rejected={reject.sum()}")

        out.loc[reject, x_col] = np.nan #We set the rejected frame landmark coordinates to NaN
        out.loc[reject, y_col] = np.nan
        out[f"{joint}_vis_effective"] = out[vis_col]
        out.loc[rescued, f"{joint}_vis_effective"] = 1.0 #For the rescued frame - we set the visibility score to 1.0 so that it bypasses the visibilty threshold of 0.5

    coord_cols = [c for c in out.columns if c.endswith("_x") or c.endswith("_y")] #Here we interpolate the rejected frames only for signal continuity 
    out[coord_cols] = out[coord_cols].interpolate(method="linear", limit_direction="both")


    return out

def print_video_statistics(angle_df: pd.DataFrame):
    """
    Print how many frames are actually usable for each feature
    after visibility gating.
    """

    total_frames = len(angle_df)

    print("\nVideo statistics")
    print("-" * 30)
    print(f"Total frames: {total_frames}\n")

    feature_names = {
        "knee_angle": "Knee angle",
        "hip_angle": "Hip angle",
        "ankle_angle": "Ankle angle",
        "pelvic_gap": "Pelvic gap",
        "heel_y": "Heel lift"
    }

    for feature, vis_cols in VIS_COLS.items():

        valid = _visible(angle_df, vis_cols)

        usable = int(valid.sum())
        percent = 100 * usable / total_frames

        print(f"{feature_names[feature]:<20}: "
              f"{usable}/{total_frames} ({percent:.1f}%)")

def smooth_landmarks(lm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Smooth raw landmark coordinates before angle calculation.
    This prevents jitter in one landmark from propagating into
    multiple angle calculations that share that landmark.
    
    Apply lighter smoothing than angles since coordinates are
    in normalised units (0-1) and the signal is already small.
    """
    out = lm_df.copy()

    coord_cols = [c for c in lm_df.columns
                  if c.endswith("_x") or c.endswith("_y")]

    for col in coord_cols:
        #Only smooth if enough non-NaN values exist
        values = lm_df[col].values
        if np.sum(~np.isnan(values)) > SMOOTH_WINDOW:
            out[col] = savgol_filter(values, SMOOTH_WINDOW, SMOOTH_POLY)

    return out

def is_valid_leg (lms, lm_indices, side="right"):
    if side=="right":
        if (lms[12].visibility>0.5 and lms[24].visibility>0.5 
        and lms[26].visibility>0.5 and lms[28].visibility>0.5 
        and lms[30].visibility>0.5 and lms[32].visibility>0.5):
            return True
        else:
            return False
        
    if side=="left":
        if (lms[11].visibility>0.5 and lms[23].visibility>0.5 
        and lms[25].visibility>0.5 and lms[27].visibility>0.5 
        and lms[29].visibility>0.5 and lms[31].visibility>0.5):
            return True
        else:
            return False
        
def extract_landmarks(video_path: str, side: str = "right", visualize: bool = False) -> pd.DataFrame:
    """
    Run MediaPipe PoseLandmarker (Tasks API) on every frame.

    Key differences from the old solutions API:
      - Needs a .task model file on disk
      - RunningMode.VIDEO requires a millisecond timestamp per frame
      - Results come back as result.pose_landmarks[pose_idx][landmark_idx]
        instead of result.pose_landmarks.landmark[idx]

    Returns the same DataFrame shape as before so the rest of the
    pipeline is completely unaffected.
    """
    lm_indices = SIDE_LANDMARK_INDICES[side] #Indces for a particular side as defined above are extracted and saved in this variable
    records = [] #Created to store the pose coordinates

    #The pose estimation model is loaded
    #An object basically contains information (data/attributes) and methods or functions that are used to operate on that data
    """ VideoCapture object (cap)
Attributes/state: file path = exercise.mp4, current frame position = 0, fps = 30, codec info, total frames.
Methods: read(), isOpened(), release(), get()"""
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    #Configure the loaded model and set parameters for inference
    options = vision.PoseLandmarkerOptions(
        base_options               = base_options, #The model is attached
        running_mode               = vision.RunningMode.VIDEO,   #Mediapipe Tasks supports multiple modes - image, video and live stream. Therefore, here we define the mode we want to operate in - which is the video mode.
        #During video processing (frame by frame processing), there is temporal tracking so the model keeps track of (or remembers) the previous frame
        num_poses                  = 1, #The main difference between this and solutions is that tasks can detect more than one people in the frame. Here, since we have set num_poses as 1 it will detect the person that gives the best confidence.
        min_pose_detection_confidence = 0.5, #Here, we define the minimum pose detection confidence, only detections with a confidence score equal to or above 0.5 will be considered, others will be ignored
        min_pose_presence_confidence  = 0.5, #This parameter checks that after the detection of a person, how confident is it that the pose landmarks (joint coordinates) are actually present and reliable enough to use
        min_tracking_confidence       = 0.5, #This parameter answers the question I knew where the body landmarks were in the previous frame. How confident am I that I can keep following (tracking) those same landmarks correctly in the current frame?
    ) #If we are processing a video fo 30fps running pose detection on every frame would be expensive so what mediapipe does is - it runs the full pose detection model on the first frame and then for the next frames it only tracks the joint coordinates since it has a context of the previous joint coordinates.
    """ Instead of asking every frame - where the person is. It already knows where the knee was in the last frame, it now checks where did it move now. What is actually happening here - Previous frame + motion estimate -> predict next landmark position.
    The tracker predicts then compares the prediction with the actual image features. If the prediction matches the image then the tracking confidence is high.
    In case the tracking confidence is too low, mediapipe will stop tracking and start detecting again from scratch from the same frame. """
    """ Mediapipe processes frames in the following manner:
    Stage 1 -> Detect a person
    Stage 2 -> Estimate body landmarks
    Stage 3 -> Track landmarks in the next frame

    The above three confidence parameters map to those stages:
    Stage 1 -> min_pose_detection_confidence
    Stage 2 -> min_pose_presence_confidence
    Stage 3 -> min_tracking_confidence
    min_pose_detection_confidence only helps detect a person, not if the joint coordinates can be detected accurately or not. It might be possible that a person might have a detection confidence of 0.8 but since a few joints are partially hidden
    min_pose_presence_confidence might be below the threshold of 0.5 and hence will be rejected. """


    cap = cv2.VideoCapture(video_path) #VideoCapture() creates an object that lets you read a video file frame by frame
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    #FPS is required since the video mode of mediapipe tasks api requires a time stamp which is achieved by converting frame number into time
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0   #fallback if fps not in metadata
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #invalid_count = 0
    paused = False

    GREEN = (0, 255, 0)
    YELLOW = (0, 255, 255)
    RED = (0, 0, 255)
    ORANGE = (0, 165, 255)
    WHITE = (255, 255, 255)

    with vision.PoseLandmarker.create_from_options(options) as detector: #Here we actually load the pose detector model that was downloaded
        frame_idx = 0 #Initial index
        while True: #Loop runs until the video ends
            if not paused:
                ret, frame = cap.read() #Here we read each frame one by one. It returns two things - ret (boolean value) it signifies whether the frame was read successfully or not (true or false) and frame, that is, the actual image - a dictionary that stores all the pixels as RGB values.
                if not ret: #If the frame is not available - it indicates the end of the video and hence we break out of the loop
                   break

                height, width, _ = frame.shape
                #OpenCV stores images in BGR format but mediapipe tasks requires the normal RGB format. Hence conversion is important
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb) #Mediapipe Tasks API cannot directly use NumPy/OpenCV arrays. It expects its own image object, so we convert rgb to image object. Frame is not an object - it is a numpy array which stores the pixel values of an image.
                """Mediapipe tasks api does not directly accept a frame and hence conversion is done to an image object. mp.Image object stores: Data - pixel data, image width, image height, image format (RGB). Methods: image_format(), numpy_view()"""
                #VIDEO mode needs a monotonically-increasing timestamp in ms. Time = frame number/fps and since the time is required in milliseconds - it is multiplied by 1000. Time stamps are important for tracking between frames, without time stamps tracking is impossible.
                timestamp_ms = int(frame_idx * 1000 / fps)
                result = detector.detect_for_video(mp_image, timestamp_ms) #Here, we actually input the image and the time stamp into the mediapipe pose detection model. Internally mediapipe detects person, estimates pose landmarks and tracks coordinates using the previous frame.

                display_frame = cv2.resize(frame, (1200, 600))
                display_height, display_width, _ = display_frame.shape

                row = {"frame": frame_idx, "detected": False} #Stores information about the current frame

                #result.pose_landmarks is a list of poses; [0] = first (only) person. lm is a mediapipe landmark object that contains: x coordinate, y coordinate, z coordinate, visibility and presence.
                """lm_indices = { "shoulder":11, "hip":23, "knee":25, "ankle":27, "foot":29, "heel":31 } .items() gives: ("shoulder",11), ("hip",23), ("knee",25), ("ankle",27), ("foot",31) """
                if result.pose_landmarks and len(result.pose_landmarks) > 0:
                    lms = result.pose_landmarks[0]          #list of NormalizedLandmark
                    valid = is_valid_leg(lms, lm_indices, side)

                    if visualize: 
                        vis_frame = display_frame.copy()

                        right_leg = [(24, 26), (26, 28), (28, 30), (28, 32), (30, 32)]
                        left_leg = [(23, 25), (25, 27), (27, 29), (27, 31), (29, 31)]
                        torso_left = [(11, 23)]
                        torso_right = [(12, 24)]

                        if side=="right":
                            pixel_rl1_s = [int(lms[right_leg[0][0]].x*display_width), int(lms[right_leg[0][0]].y*display_height)]
                            pixel_rl1_e = [int(lms[right_leg[0][1]].x*display_width), int(lms[right_leg[0][1]].y*display_height)]
                            pixel_rl2_s = [int(lms[right_leg[1][0]].x*display_width), int(lms[right_leg[1][0]].y*display_height)]
                            pixel_rl2_e = [int(lms[right_leg[1][1]].x*display_width), int(lms[right_leg[1][1]].y*display_height)]
                            pixel_rl3_s = [int(lms[right_leg[2][0]].x*display_width), int(lms[right_leg[2][0]].y*display_height)]
                            pixel_rl3_e = [int(lms[right_leg[2][1]].x*display_width), int(lms[right_leg[2][1]].y*display_height)]
                            pixel_rl4_s = [int(lms[right_leg[3][0]].x*display_width), int(lms[right_leg[3][0]].y*display_height)]
                            pixel_rl4_e = [int(lms[right_leg[3][1]].x*display_width), int(lms[right_leg[3][1]].y*display_height)]
                            pixel_rl5_s = [int(lms[right_leg[4][0]].x*display_width), int(lms[right_leg[4][0]].y*display_height)]
                            pixel_rl5_e = [int(lms[right_leg[4][1]].x*display_width), int(lms[right_leg[4][1]].y*display_height)]
                            pixel_torso_right_s = [int(lms[torso_right[0][0]].x*display_width), int(lms[torso_right[0][0]].y*display_height)]
                            pixel_torso_right_e = [int(lms[torso_right[0][1]].x*display_width), int(lms[torso_right[0][1]].y*display_height)]
                            cv2.line(vis_frame, pixel_rl1_s, pixel_rl1_e, (150, 150, 150), 2)
                            cv2.line(vis_frame, pixel_rl2_s, pixel_rl2_e, (150, 150, 150), 2)
                            cv2.line(vis_frame, pixel_rl3_s, pixel_rl3_e, (150, 150, 150), 2)
                            cv2.line(vis_frame, pixel_rl4_s, pixel_rl4_e, (150, 150, 150), 2)
                            cv2.line(vis_frame, pixel_rl5_s, pixel_rl5_e, (150, 150, 150), 2)
                            cv2.line(vis_frame, pixel_torso_right_s, pixel_torso_right_e, (150, 150, 150), 2)
                            
                        if side=="left":
                            pixel_ll1_s = [int(lms[left_leg[0][0]].x*display_width), int(lms[left_leg[0][0]].y*display_height)]
                            pixel_ll1_e = [int(lms[left_leg[0][1]].x*display_width), int(lms[left_leg[0][1]].y*display_height)]
                            pixel_ll2_s = [int(lms[left_leg[1][0]].x*display_width), int(lms[left_leg[1][0]].y*display_height)]
                            pixel_ll2_e = [int(lms[left_leg[1][1]].x*display_width), int(lms[left_leg[1][1]].y*display_height)]
                            pixel_ll3_s = [int(lms[left_leg[2][0]].x*display_width), int(lms[left_leg[2][0]].y*display_height)]
                            pixel_ll3_e = [int(lms[left_leg[2][1]].x*display_width), int(lms[left_leg[2][1]].y*display_height)]
                            pixel_ll4_s = [int(lms[left_leg[3][0]].x*display_width), int(lms[left_leg[3][0]].y*display_height)]
                            pixel_ll4_e = [int(lms[left_leg[3][1]].x*display_width), int(lms[left_leg[3][1]].y*display_height)]
                            pixel_ll5_s = [int(lms[left_leg[4][0]].x*display_width), int(lms[left_leg[4][0]].y*display_height)]
                            pixel_ll5_e = [int(lms[left_leg[4][1]].x*display_width), int(lms[left_leg[4][1]].y*display_height)]
                            pixel_torso_left_s = [int(lms[torso_left[0][0]].x*display_width), int(lms[torso_left[0][0]].y*display_height)]
                            pixel_torso_left_e = [int(lms[torso_left[0][1]].x*display_width), int(lms[torso_left[0][1]].y*display_height)]
                            cv2.line(vis_frame, pixel_ll1_s, pixel_ll1_e, (150, 150, 150), 2)
                            cv2.line(vis_frame, pixel_ll2_s, pixel_ll2_e, (150, 150, 150), 2)
                            cv2.line(vis_frame, pixel_ll3_s, pixel_ll3_e, (150, 150, 150), 2)
                            cv2.line(vis_frame, pixel_ll4_s, pixel_ll4_e, (150, 150, 150), 2)
                            cv2.line(vis_frame, pixel_ll5_s, pixel_ll5_e, (150, 150, 150), 2)
                            cv2.line(vis_frame, pixel_torso_left_s, pixel_torso_left_e, (150, 150, 150), 2)

                        joint_labels = {
                            "shoulder" : "SHO",
                            "hip" : "HIP",
                            "knee" : "KNEE",
                            "ankle" : "ANK", 
                            "foot" : "FOOT", 
                            "heel": "HEEL"
                        }
                        for joint, idx in lm_indices.items():
                            lm = lms[idx]
                            px = int(lm.x * display_width)
                            py = int(lm.y * display_height)
                            color = GREEN if valid else RED
                            cv2.circle(vis_frame, (px, py), 8, color, -1)
                            text= f"x:{lm.x:.2f} y:{lm.y:.2f}"
                            cv2.putText(vis_frame, text, (px - 50, py - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                            cv2.putText(vis_frame, joint_labels[joint], (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        if valid:
                            cv2.putText(vis_frame, "Valid Leg", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, GREEN, 2)
                        else:
                            cv2.putText(vis_frame, "Interpolated Later", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2)

                        hip_lm = lms[lm_indices["hip"]]
                        knee_lm = lms[lm_indices["knee"]]
                        ank_lm = lms[lm_indices["ankle"]]
                        sho_lm = lms[lm_indices["shoulder"]]
                        foot_lm = lms[lm_indices["foot"]]
                        heel_lm = lms[lm_indices["heel"]]

                        hip_arr = np.array([hip_lm.x, hip_lm.y])
                        knee_arr = np.array([knee_lm.x, knee_lm.y])
                        ank_arr = np.array([ank_lm.x, ank_lm.y])
                        sho_arr = np.array([sho_lm.x, sho_lm.y])
                        foot_arr = np.array([foot_lm.x, foot_lm.y])
                        
                        knee_angle = _angle_3pt(hip_arr, knee_arr, ank_arr)
                        hip_angle = _angle_3pt(sho_arr, hip_arr, knee_arr)
                        ankle_angle = _angle_3pt(knee_arr, ank_arr, foot_arr)
                        cv2.putText(vis_frame, f"Frame: {frame_idx}", (400, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1)
                        cv2.putText(vis_frame, f"Knee: {knee_angle: .1f}  Hip: {hip_angle: .1f}  Ankle: {ankle_angle: .1f}", (20, display_height-70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2)
                        cv2.putText(vis_frame, f"Visibility: Knee - {knee_lm.visibility: .2f} " f"Hip - {hip_lm.visibility: .2f} " f"Ankle - {ank_lm.visibility: .2f} " f"Shoulder - {sho_lm.visibility: .2f} " f"Foot - {foot_lm.visibility: .2f}" f"Heel - {heel_lm.visibility: .2f}", (20, display_height-40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

                        if paused:
                            cv2.putText(vis_frame, "Paused", (display_width//2 - 60, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2)
                        cv2.imshow("Extraction Visualizer", vis_frame) #cv2.imshow(window_name, image) this command basically shows a pop up window where an image or video is displayed 
                    
                    
                    for joint, idx in lm_indices.items():
                        lm = lms[idx]
                        row[f"{joint}_x"] = lm.x
                        row[f"{joint}_y"] = lm.y
                        row[f"{joint}_vis"] = lm.visibility
                    row["detected"] = True

                else:
                    for joint in lm_indices:
                        row[f"{joint}_x"]   = np.nan
                        row[f"{joint}_y"]   = np.nan
                        row[f"{joint}_vis"] = 0.0

                records.append(row) #Append data to the records array 
                frame_idx += 1 #Increment the frame number

            #Basic commands 
            if visualize:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Visualiser quit early by user")
                    break
                elif key == ord(' '):
                    paused = not paused
                elif key == ord('s'):
                    fname = f"extract_frame_{frame_idx}.jpg"
                    cv2.imwrite(fname, vis_frame)
                    print(f"  Screenshot saved → {fname}")

        if visualize:
            cv2.destroyAllWindows()
    cap.release() #After the video ends, we close the file

    df = pd.DataFrame(records).set_index("frame") #pandas converts list into a table and the frame number is set as the index
    
    detection_rate = df["detected"].mean() * 100 #The detected column of the data frame contains T or F. T = 1, F = 0. Example: [1, 1, 0, 0, 1], we then take the mean of these values and multiply that by 100 to get the detection rate.
    #Change this line in extract_landmarks():
    print(f"  {frame_idx} frames | raw MediaPipe detection rate: {detection_rate:.1f}%")
    if detection_rate < 70:
        print("  WARNING: low detection rate — check video angle / lighting")

    return df

"""Calculate Joint Angles"""

#Currently the dataframe contains raw joint coordinates, we need joint angles therefore we calculate them using trignometry
def _angle_3pt(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float: #Here we calculate the angle formed by three points, the angle is calculated at point B
#In the following step we get two vectors. Formula to get the angle - Cos theta = (ba.bc)/(|ba||bc|)
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8
    cos_val = np.dot(ba, bc) / denom
    return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))

#Trunk lean basically measures how much the torso leans forward
def _pelvic_lift(hip_xy: np.ndarray, shoulder_xy: np.ndarray) -> float:
    """
    Hip-to-shoulder vertical gap, normalized by torso length (the
    shoulder-hip distance in this same frame). Dividing by torso length
    cancels out camera distance — a patient closer to the camera and
    a patient further away will produce comparable values for the same
    physical lift, since both the gap and the torso length scale together.

    This also has a useful side effect: it roughly normalizes for
    differences in patient body size too, not just camera distance.
    """
    gap          = hip_xy[1] - shoulder_xy[1]
    torso_length = np.linalg.norm(hip_xy - shoulder_xy) + 1e-8
    return float(gap / torso_length) #We divide by the torso length so that the value of the pelvic gap that we receive is normalized and is not dependent of the size of the patient in the frame.
    """ The pelvic gap requires normalization since it varies depending on the distance of the camera it is recorded from. If the camera is close a real gap of 2 cm might take up 10 pixels but if the camera is farther,
      the same gap might take up only 3 pixels although the lift is the same. The actual joint positions(in case of calculation of angles) in the real world have not changed. Only the pixel numbers changed.
      Instead of just measuring the lift in pixels, you also measure something else in the same frame that you know doesnt change - like the lenth of the torso.  Because both numbers shrink or grow together by the same amount, dividing one by the
      other cancels out the camera distance completely. This is called normalization and is already being performed while calculating the angles.
      """
#This function actually calculates the angles
def calculate_angles(lm_df: pd.DataFrame) -> pd.DataFrame: #lm_df is a dataframe that contains the landmark coordinates
    rows = [] #Create an empty array that will later store the calculated angles
    for frame_idx, r in lm_df.iterrows(): #It saves the coordinates and loops through each row of data. lm_df is a pandas dataframe and iterrows() returns the index and data in a particular row.
    #frame_idx stores the index of the row and r stores the data in the row
        shoulder = np.array([r["shoulder_x"], r["shoulder_y"]]) #Extract the shoulder coordinates from each row of data
        hip = np.array([r["hip_x"], r["hip_y"]])
        knee = np.array([r["knee_x"], r["knee_y"]])
        ankle = np.array([r["ankle_x"], r["ankle_y"]])
        foot = np.array([r["foot_x"], r["foot_y"]])
        heel = np.array([r["heel_x"], r["heel_y"]])

        torso_length = np.linalg.norm(hip-shoulder) + 1e-8
        
        rows.append({
            "frame": frame_idx, #The functions are called and the data is added to the rows
            "knee_angle": _angle_3pt(hip, knee, ankle),
            "hip_angle": _angle_3pt(shoulder, hip, knee),
            "ankle_angle": _angle_3pt(knee, ankle, foot),
            "pelvic_gap": _pelvic_lift(hip, shoulder),
            "hip_vis": r["hip_vis"],
            "knee_vis": r["knee_vis_effective"],
            "ankle_vis": r["ankle_vis_effective"],
            "shoulder_vis": r["shoulder_vis"],
            "heel_y": heel[1],
            "torso_length": torso_length,
            "heel_vis": r["heel_vis_effective"],
            "foot_vis":r["foot_vis_effective"]
        })

    return pd.DataFrame(rows).set_index("frame") #Set the frame as the index of the row

"""Smooth Angles"""

#This function takes the raw angles that were calculated and smooths them using the savitzky golay filter
def smooth_angles(angle_df: pd.DataFrame) -> pd.DataFrame:
    out = angle_df.copy()
    for col in ["knee_angle", "hip_angle", "ankle_angle", "pelvic_gap", "heel_y"]:
        out[col] = savgol_filter(angle_df[col].values, SMOOTH_WINDOW, SMOOTH_POLY)
    return out

"""Rep Detection Logic"""

def detect_reps(knee_angles: np.ndarray) -> list[dict]: #knee_angles is a numpy array that stores knee angles across video frames
    """
    The major change made here is that instead of looking for peak -> valley -> peak and counting this as one rep we shift to a more robust approach where
    instead we look for the max angle before and after a valley and that is treated as the rep boundary. Because it is not necessary that a patient might have the same ROM as required by the find_peaks() due to which this function
    might miss reps. For a patient 145 -> 110 -> 140 might only be possible because of restricted range of motion - the algorithm should be able to count this as a rep which find_peaks() might miss.
    Old approach: peak -> valley -> peak
    new approach: max -> valley -> max
    """
    #Valleys = minimum knee angle = peak flexion. The find_peaks() can only find maxima, hence we take -knee_angles to find the minima
    valleys, _ = find_peaks(
        -knee_angles,
        prominence = REP_MIN_PROMINENCE, #Rep prominence is the minimum change in angle that will be counted as a rep
        distance   = REP_MIN_DISTANCE,
    )

    if len(valleys) == 0:
        print("  WARNING: no rep valleys found — try lowering REP_MIN_PROMINENCE")
        return []

    #Peaks = maximum knee angle = full extension. Here we find the maxima
    #Find the max angle before the first valley. That is designated as the initial rep boundary
    #np.argmax returns the index of the maxima. Basically we find the all the knee_angles before the valley point and then using np.argmax extract the maxima from those points.
    boundary_peaks = [int(np.argmax(knee_angles[: valleys[0]]))]

    #Find the max angle between consecutive valleys since that is marked as the latter rep boundary, that is, after the valley
    for i in range(len(valleys) - 1): #We are comparings pairs of valleys. Between two consecutive valleys there is one peak which marks the end of rep 1 and the start of rep 2
        window = knee_angles[valleys[i] : valleys[i + 1]] #We extract all the knee angles between the two valleys
        local_max = int(np.argmax(window)) + valleys[i] #We find the maxima from the extracted window using np.argmax. This returns the index of the maxima which is added to the valley
        #before it since the index returned is local to that window and hence it needs to be added to the valley before to get the real frame.
        boundary_peaks.append(local_max)

    #After last valley: max in [last valley → end]. It finds the boundary of the last rep by finding the max angle
    last_window = knee_angles[valleys[-1] :] #valleys[-1] indicates the last element of the array. We consider all frames from the last valley to the end of the video and hence extract all knee angles between those frames.
    boundary_peaks.append(int(np.argmax(last_window)) + valleys[-1]) #Similarly, we find the maxima in the window and extract its index which is added to the index of the last valley to get the real frame number
    #enumerate() works similar to iterrows() in terms that both return the index and value at that location but enumerate() works on data types like arrays, lists, tuples and returns simple values whereas, iterrows() work only on pandas data frames
    #and return the index of the row and all the data of a row, so a dictionary of data unlike a simple value in the case of enumerate().

    reps = [] #Creates an array that later stores rep data
    #Move through each rep/ valleys
    for i, valley in enumerate(valleys): #enumerate returns the index and the value at that index
        start = boundary_peaks[i] #The first peak is the start of the first rep
        end   = boundary_peaks[i + 1] #The end of this rep and start of the next rep.

        rep_range = knee_angles[start] - knee_angles[valley] #ROM is calculated as the difference between the angles at extension and flexion
        if rep_range < REP_MIN_PROMINENCE:
            print(f"  Skipping shallow valley at frame {valley} "
                  f"(range={rep_range:.1f}° < {REP_MIN_PROMINENCE}°)") #Skip if flexion depth is too shallow. For example - start angle = 180, valley = 170. Therefore, difference = 10 which is less than 15. So not detected as a real rep
            continue

        reps.append({
            "rep_id":       i + 1,
            "start_frame":  int(start),
            "valley_frame": int(valley),
            "end_frame":    int(end),
        })

    print(f"  {len(reps)} rep(s) detected")
    return reps

"""State Assignment"""

def assign_states(
    knee_angles: np.ndarray,
    start: int,
    valley: int,
    end: int,
) -> np.ndarray: #Here star, valley & end basically hold the frame number from where that particular rep starts, the point of maximum flexion (which is the valley) and the end of the rep.
#For eg: Frame 100 -> start of the rep, Frame 150 -> maximum flexion (valley), Frame 200 -> end of the rep
    """
    Label each frame in [start, end] with a state 1–4.

    Thresholds are computed from this rep's own min/max so they adapt
    to patients with limited ROM.

    S1 (Extended)   : knee ≥ rep_max − frac × rep_range
    S3 (Peak Flex)  : knee ≤ rep_min + frac × rep_range
    S2 (Descending) : between S1 and S3 thresholds, BEFORE valley
    S4 (Ascending)  : between S1 and S3 thresholds, AFTER  valley

    Returns integer array of length (end − start + 1).
    """
    rep_angles   = knee_angles[start : end + 1] #Return knee angles of that rep
    rep_max      = rep_angles.max() #Compute the max angle - fully extended
    rep_min      = rep_angles.min() #Compute the min angle - maximum flexion
    rep_range    = rep_max - rep_min + 1e-8 #1e-8 is a very small number to avoid division by zero. For example: rep_max = 180, rep_min = 180 then the range of motion would be 0

    S1_EXIT = 10
    S3_EXIT = 10
    thresh_high = rep_max - S1_EXIT
    thresh_low = rep_min + S3_EXIT
    #thresh_high = rep_max - STATE_BOUNDARY_FRAC * rep_range   #S1 boundary - maximum extension
    #thresh_low  = rep_min + STATE_BOUNDARY_FRAC * rep_range   #S3 boundary - maximum flexion
    valley_local = valley - start                              #index within slice. Find the valley within the rep so that we can distinguish whether the patient is moving into flexion or moving back to extension

    n      = len(rep_angles)
    states = np.zeros(n, dtype=int) #Create an array to store the states

    for i, angle in enumerate(rep_angles): #assign states
        if angle >= thresh_high:
            states[i] = 1                            #S1 — Extended
        elif angle <= thresh_low:
            states[i] = 3                            #S3 — Peak Flexion
        elif i <= valley_local:
            states[i] = 2                            #S2 — Descending
        else:
            states[i] = 4                            #S4 — Ascending

    return states

"""Feature Extraction"""

ANGLE_COLS   = ["knee_angle", "hip_angle", "ankle_angle",  "pelvic_gap", "heel_y"] #An array is defined that stores the 4 main angles
ANGLE_LABELS = ["knee", "hip", "ankle", "pelvic", "heel"] #Stores the labels
N_STATES     = 4 #The number of states per repetition
VIS_THRESHOLD = 0.5
VIS_COLS = {
    "knee_angle": ["hip_vis", "knee_vis", "ankle_vis"],
    "hip_angle": ["shoulder_vis", "hip_vis", "knee_vis"],
    "ankle_angle": ["knee_vis", "ankle_vis", "foot_vis"],
    "pelvic_gap":  ["hip_vis", "shoulder_vis"], 
    "heel_y": ["heel_vis"]       
} #A visibility threshold is added to ensure that only frames that have a visibility above the given thresholds for the key joints required would be considered and the frames with lower
#visibility will be rejected.

STAT_FUNCS = {
    "mean":  np.mean,
    "min":   np.min,
    "max":   np.max,
    "range": lambda x: float(np.max(x) - np.min(x)),
    "std":   np.std,
} #The features that need to be calculated per state of a rep

def _visible(df, vis_cols, threshold=VIS_THRESHOLD):
    """
    Row-wise mask: True only where EVERY listed visibility column is
    >= threshold for that row. A feature computed from multiple landmarks
    (e.g. knee_angle needs hip, knee AND ankle) is only kept when all of
    them cleared the threshold, not just the one sharing its name.
    """
    cols = [c for c in vis_cols if c in df.columns]
    if not cols:
        return pd.Series(True, index=df.index)
    return (df[cols] >= threshold).all(axis=1)

def compute_video_baseline(smooth_df: pd.DataFrame, reps: list[dict], states_by_rep: dict) -> float:
    """
    Pool S1 (extended-state) heel_y frames across ALL reps in the video
    to get one stable baseline, rather than recomputing it per rep from
    a small handful of frames each time.
    """
    s1_heel_values = []

    for rep in reps:
        start, end = rep["start_frame"], rep["end_frame"]
        states = states_by_rep[rep["rep_id"]]
        rep_df = smooth_df.loc[start:end].copy()
        rep_df["state"] = states

        valid = _visible(rep_df, ["heel_vis"])
        s1_mask = (rep_df["state"] == 1) & valid
        s1_heel_values.extend(rep_df.loc[s1_mask, "heel_y"].values)

    if len(s1_heel_values) == 0:
        #fallback: no clean S1 frames anywhere in the video
        return np.nan

    return float(np.median(s1_heel_values))

def compute_rep_features(
    angle_df: pd.DataFrame,
    rep: dict,
    states: np.ndarray,
    patient_id: str,
    baseline: float,
) -> dict:
    """
    Compute per-state stats for a single rep.
    Returns one flat dict — one row of the final CSV.

    Column naming: S{state}_{angle}_{stat}
    e.g. S1_knee_min, S2_trunk_mean, S3_ankle_range
    """

    start = rep["start_frame"] #Extract the rep boundaries
    end   = rep["end_frame"]

    #Extract the frames belonging to this rep
    rep_angle_df = angle_df.loc[start:end]
    rep_angle_df = rep_angle_df.copy()
    rep_angle_df["state"] = states

    row = {
        "patient_id":   patient_id,
        "rep_id":       rep["rep_id"],
        "start_frame":  start,
        "end_frame":    end,
        "valley_frame": rep["valley_frame"],
        "rep_duration": end - start + 1,
    }

    for s in range(1, N_STATES + 1):
        mask = (states == s) #for eg: if states == 1, [1, 1, 2, 2, 2, 3, 3, 4, 4] then [True, True, False, False, False, False, False, False, False]
        state_frames = rep_angle_df[mask] #Stores the frames belonging to a particular state

        row[f"S{s}_duration"] = int(mask.sum()) #Stores the summation of those frame. Basically stores how many frames did a particular state last

        for col, label in zip(ANGLE_COLS, ANGLE_LABELS): #zip() pairs values together ("knee_angle", "knee"), ("hip_angle", "hip"), etc.
            vis_mask = _visible(state_frames, VIS_COLS.get(col, []))
            gated_frames = state_frames[vis_mask]
                # Track how many frames were kept vs total
            row[f"S{s}_{label}_vis_frames"] = int(vis_mask.sum())
          

            # Use gated frames for stats, fall back to NaN if all frames dropped
            vals = (
                gated_frames[col].values
                if len(gated_frames) > 0
                else np.array([np.nan])
            )

            for stat_name, stat_fn in STAT_FUNCS.items():
                row[f"S{s}_{label}_{stat_name}"] = (
                    float(stat_fn(vals)) if len(vals) > 0 else np.nan
                )
    for s in range(1, N_STATES + 1):
        mask = (states == s)
        state_frames = rep_angle_df[mask]
        vis_mask = _visible(state_frames, VIS_COLS.get("heel_y", []))
        gated_heel = state_frames.loc[vis_mask, "heel_y"]

        if len(gated_heel) == 0 or np.isnan(baseline):
            row[f"S{s}_heel_range"] = np.nan
        else:
            row[f"S{s}_heel_range"] = float(max(0.0, baseline - gated_heel.min()))
    row["pelvic_lift_max"] = row["S1_pelvic_mean"] - np.nanmin([
        row["S2_pelvic_min"], row["S3_pelvic_min"], row["S4_pelvic_min"]
    ])
    #df.loc[row_indexer, column_indexer]
    valid = _visible(rep_angle_df, ["heel_vis"])
    valid_heel = rep_angle_df.loc[valid, "heel_y"]

    if len(valid_heel) == 0 or np.isnan(baseline):
        row["heel_lift_max"] = np.nan
        row["heel_lift_mean"] = np.nan
    else:
        # No torso-length division: raw pixel-space lift, using the
        # video-level baseline computed once, not per rep.
        heel_lift = (baseline - valid_heel).clip(lower=0)
        row["heel_lift_max"] = float(heel_lift.max())
        row["heel_lift_mean"] = float(heel_lift.mean())

    row["knee_rom"] = row["S1_knee_mean"] - row["S3_knee_min"]
    row["hip_compensation"] = row["S3_hip_mean"] - row["S1_hip_mean"]
    row["S2_S4_speed_ratio"] = row["S2_duration"] / (row["S4_duration"] + 1e-8)

    return row
"""Main Pipeline"""

#Processes everything and calls all the functions defined above
#Basic input arguments are provided here which would be required to process the video and save its data in a pandas dataframe such as the path of the video,
#patient id, the side of the exercise. One video is processed here.
def process_video(
    video_path: str,
    patient_id: str,
    side: str = "right", #The default side is set to right
    output_csv: str = None,
) -> pd.DataFrame:
    """
    Full pipeline: video → one-row-per-rep feature DataFrame.
    """
    print(f"\n{'='*60}")
    print(f"Video     : {os.path.basename(video_path)}")
    print(f"Patient   : {patient_id}   |   Side: {side}")
    print(f"{'='*60}")

    #1. Landmarks. This function uses mediapipe tasks. The input is the video and side. Internally mediapipe detects a person with the highest confidence,
    #extracts joints coordinates (x&y) and stores it in a dataframe - lm_df
    print("\n[1/7] Extracting landmarks")
    lm_df = extract_landmarks(video_path, side, visualize=False)

    #This step is added since when the leg is being flexed - the foot landmark starts jittering so instead of taking noisy landmarks in the savgol filter which would further disrupt calculations
    #we simply reject those frame where visibility is low. This function then interpolates the nosiy values with the neighboring values instead of using noisy data points. 
    print("\n[2/7] Visibility Check")
    lm_df = gate_low_visibility(lm_df) 

    print("[3/7] Smoothing landmarks")         
    lm_df_smooth = smooth_landmarks(lm_df)

    #2. Angles. Here we actually calculate the angles using raw joint coordinates.
    print("[4/7] Calculating joint angles")
    angle_df = calculate_angles(lm_df_smooth)

    print_video_statistics(angle_df)

    #3. Smooth. Here smoothing of the angles is performed using the savitzky golay filter
    print("[5/7] Smoothing")
    smooth_df = smooth_angles(angle_df)

    #4. Reps. Reps are detected based on a tracking angle. In case of supine heel slides, the tracking angle is the knee angle since it has the highest
    #range or movement
    print("[6/7] Detecting reps")
    knee  = smooth_df["knee_angle"].values
    reps  = detect_reps(knee) #The rep detection algorithm is called. Internally the valleys are found and peaks before and after the valley to identify rep boundaries

    if not reps: #If no rep detected, return empty data frame
        return pd.DataFrame()

    #5. States + features
    print("[7/7] Segmenting states & computing features")
    #First pass: compute states for every rep (needed before we can pool S1 frames)
    states_by_rep = {}
    for rep in reps:
        states_by_rep[rep["rep_id"]] = assign_states(knee, rep["start_frame"], rep["valley_frame"], rep["end_frame"])

    #video-level baseline
    video_baseline = compute_video_baseline(smooth_df, reps, states_by_rep)
    print(f"Video-level heel baseline: {video_baseline:.4f} (from pooled S1 frames)")
    all_rows = [] #Stores the final result

    for rep in reps: #Loops through all reps in reps
        start  = rep["start_frame"]  #Extract basic rep features like start frame number, end frame number and valley frame number
        end    = rep["end_frame"]
        valley = rep["valley_frame"]

        states = assign_states(knee, start, valley, end) #Output: states = [1,1,1,2,2,3,3,4,4]. States are assigned each frame of a rep
        row = compute_rep_features(smooth_df, rep, states, patient_id, video_baseline) #This function calculates the statistical values like mean, min, max, range and standard deviation, along with the
        #pelvic lift for each state. Returns one dictionary which is appended to all_rows
        all_rows.append(row)

        #Per-rep console summary
        k_slice = knee[start : end + 1] #Extract only the knee angles for this rep
        print(
            f"  Rep {rep['rep_id']:>2} | frames {start}–{end} " #:>2 right alignment
            f"| knee {k_slice.max():.1f}°→{k_slice.min():.1f}° " #print the rep's max and min for knee angle
            f"| "
            + "  ".join(f"S{s}={int((states == s).sum())}f" for s in range(1, 5)) #Prints how long a state lasts in terms of frames. S2 = 3f
        )

    result_df = pd.DataFrame(all_rows) #Final data that is converted into a pandas data frame and stored in result_df

    if output_csv:
       file_exists = os.path.isfile(output_csv)
       result_df.to_csv(output_csv, mode='a', header=not file_exists, index=False)
       print(f"\n  Saved → {output_csv}  ({'appended' if file_exists else 'created'})")

    print(f"\n  Done: {len(result_df)} rep(s) × {len(result_df.columns)} columns")
    return result_df

 #To check how many reps were detected and the position of the valley
lm_df = extract_landmarks(VIDEO_PATH, SIDE, visualize=True)
lm_df = gate_low_visibility(lm_df) 
angle_df = calculate_angles(lm_df)
smooth = smooth_angles(angle_df)
knee = smooth["knee_angle"].values
reps = detect_reps(knee)

for rep in reps:
    print(f"Rep {rep['rep_id']}: valley at frame {rep['valley_frame']}, "
          f"knee = {knee[rep['valley_frame']]:.1f}°")

"""Diagnostic Plot (To identify if the code is properly marking reps or not, on the basis of the tracking angle)"""

def plot_rep_debug(smooth_df: pd.DataFrame, reps: list[dict], knee: np.ndarray): #Inputs to this function are smoothed angles, detected reps and a numpy array containing the knee angles
    """
    Plot knee angle time series with rep boundaries and states marked.
    Call after process_video() for visual sanity-check.
    Requires matplotlib.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    STATE_COLORS = {1: "steelblue", 2: "orange", 3: "tomato", 4: "mediumseagreen"}
    STATE_NAMES  = {1: "S1 Extended", 2: "S2 Descending",
                    3: "S3 Peak Flex", 4: "S4 Ascending"}

    fig, axes = plt.subplots(4, 1, figsize=(14, 8), sharex=True) #Create a figure with 4 rows and 1 column. axes[0] stores the first graph, axes[1] stores the second graph
    #and axes[3] stores the 3rd graph. size of the figure is 14 inches in width and 8 inches in height. x axis (frame number) is same for all 4 graphs
    frames = smooth_df.index.values

    for ax, col, label in zip(axes,
                               ["knee_angle", "hip_angle", "ankle_angle", "pelvic_gap"],
                               ["Knee (°)", "Hip (°)", "Ankle (°)", "Pelvic Gap (norm)"]):
        ax.plot(frames, smooth_df[col].values, lw=1.5, color="navy") #Here we draw a line graph
        ax.set_ylabel(label, fontsize=9) #y label is the label of the column
        ax.grid(True, alpha=0.3) #Add grid. Adds faint background lines alpha=0.3 means transparency

    #Shade states on the knee subplot
    for rep in reps:
        start  = rep["start_frame"]
        end    = rep["end_frame"]
        valley = rep["valley_frame"]
        states = assign_states(knee, start, valley, end)

        for i, s in enumerate(states):
            f = start + i #Convert local frame to real frame
            axes[0].axvspan(f, f + 1, alpha=0.25,
                             color=STATE_COLORS[s], linewidth=0) #axvspan() draws a vertical colored rectangle

        #Rep valley marker. Draws a vertical colored line marking the valley in the rep
        axes[0].axvline(valley, color="red", lw=1, ls="--", alpha=0.7)
        axes[0].text(valley, knee[valley] - 3, f"R{rep['rep_id']}",
                     ha="center", fontsize=7, color="red")

    patches = [mpatches.Patch(color=c, label=n, alpha=0.5)
               for s, (c, n) in enumerate(
                   zip(STATE_COLORS.values(), STATE_NAMES.values()), 1)] #state colors and state names are paired
    axes[0].legend(handles=patches, fontsize=8, loc="upper right") #Legend is added to the top right corner
    axes[0].set_title("Knee angle with rep states", fontsize=10) #title is provided to the graph
    axes[3].set_xlabel("Frame") #x label is set

    plt.tight_layout() #Prevents the overlapping of labels. Without this the titles may overlap
    plt.savefig("rep_debug.png", dpi=150)
    plt.show()
    print(" Debug plot saved → rep_debug.png")

if __name__ == "__main__":
    df = process_video(
        video_path  = VIDEO_PATH,
        patient_id  = PATIENT_ID,
        side        = SIDE,
        output_csv  = OUTPUT_CSV,
    )

    if not df.empty:
        print("\nColumn list:")
        for c in df.columns: #Loops through columns and prints them
            print(f"  {c}")
        print(f"\nSample row:\n{df.iloc[0].to_string()}")

import matplotlib.pyplot as plt

#To construct a graph of knee angle vs frames we need to extract the landmarks, calculate angles, smooth the angles and then extract the smoothed out knee angle values which are then used in the graph for plotting.
lm_df = extract_landmarks(VIDEO_PATH, SIDE, visualize=False)
lm_df = gate_low_visibility(lm_df)
angle_df = calculate_angles(lm_df)
smooth = smooth_angles(angle_df)
knee = smooth["knee_angle"].values

plt.figure(figsize=(14, 4))
plt.plot(knee)
plt.title("Knee angle over time") #We do this to make sure how many valleys are present. Number of valleys = Number of reps
plt.xlabel("Frame")
plt.ylabel("Degrees")
plt.grid(alpha=0.3)
plt.show()