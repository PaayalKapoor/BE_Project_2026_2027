#This is the feature extraction code for shoulder abduction. This exercise is recorded from the front view and focuses on the range of motion of the shoulder joint.
#The key angles in this exercise include - 1. Shoulder Abduction (arm to trunk angle), 2. Elbow Angle, 3. Trunk Lateral Lean Angle, 4. Shoulder Elevation, 5. Shoulder tilt relative to the hip
#Here we basically extract important coordinates from each frame which are used to calculate the angles formed in that frame using the 3 point formula.
#The tracking angle is shoulder abduction (arm_trunk_angle). Shoulder abduction angle starts at 0 (valley) and can go upto 120-150 degrees (peak)
#For rep detection we use a tracking angle which undergoes the most amount of movement. Peaks or valleys are used to detect reps. In this exercise the valleys are the boundaries (starting and ending) of the reps.
#The peaks (higher angle values) are the midpoint (When the hand is overhead) of the exercise. Once the rep is detected, the frames for a particular rep are isolated. The rep frames are further divided into states. The state classification is done so that errors
#in a particular rep can be specifiec to a particular part of the exercise.
#S1 - resting, S2 - Ascending up, S3 - Hand up, S4 - Descending down. The statistics for a particular state are calculated and taken as a feature. 
#Certain factors like rom and trunk lean are calculated keeping in mind the baseline which is calculated when the body is at rest. This is to avoid giving false errors in cases like kyphosis, where the spine is by default bent forward 
#changing the resting position of the shoulder from that of a normal person. 

#To do: 
#Visualizer code - done 
#Why shoulder angle 180 
#extension? (not starting from 0)
#Hip being occluded (jittering a little)

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy.signal import savgol_filter, find_peaks
import numpy as np
import pandas as pd
import urllib.request
import os
import matplotlib.pyplot as plt

#Here we download the pose landmarker model which performs the following steps: 1. Detect Human, 2. Perform Pose Estimation: Identify 33 body landmarks, 3. Temporal Tracking: Track the landmarks across frames.
#tasks API supports detection of mutliple humans in one frame unlike the older, solutions API
MODEL_PATH = "pose_landmarker_heavy.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_heavy/float16/latest/"
    "pose_landmarker_heavy.task"
)

def download_model():
    if os.path.exists(MODEL_PATH):
        print(f"  Model already present: {MODEL_PATH}")
        return
    print(f"  Downloading model → {MODEL_PATH}")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Downloaded")

download_model()

#Here we provide the video path as well as the csv file where the extracted features should be appeneded. 
VIDEO_PATH  = r"C:\Users\ADMIN\Documents\GitHub\BE_Project_2026_2027\videos\Shoulder_Abduction\Sayalee_SHA_R_1.mp4"
OUTPUT_CSV  = "docs/Datasets/Shoulder_Abduction/shoulder_abduction_data.csv"
PATIENT_ID = "Sayalee_SHA_R_1"
SIDE = "right"   #Here we provide the side that is facing the camera 
PATIENT_HEIGHT_CM = 162
EYE_TO_VERTEX_FRAC = 0.065
VIS_THRESHOLD = 0.5
#The savitzky golay filter does not work on live stream videos since it requires a window of previous, current and future data points to estimate the middle smoothed value. 
#Smoothing. The angles are smoothed using the savitzky golay filter. This filter basically preserves the peaks and valleys in the data. For example - moving average takes the previous, current and future reading which is averaged.
#The problem with the moving average filter is that it ends up disturbing the overall shape of the signal, eg - 180, 150, 180 here 150 will be averaged to 170. Therefore moving averages tend to flatten peaks and valleys.
#Whereas the savitzky golay filter is a smoothing filter that reduces noise in data while preserving the overall shape, peaks, valleys and trends in the signal.
SMOOTH_WINDOW = 11 #Smoothing window is set to 21, must be odd so that the middle value can be found easily. Points before and after the center value help estimate the smoothed points. For noiser signals, the value should be higher.
#Here, we use 21 neighbouring points to estimate the center value. 10 before, current and 10 after
SMOOTH_POLY   = 3 #Fit a cubic polynomial through those 21 points. It uses something called Least Squares Regression. The algorithm tries many possible curves and picks the one with the smallest overall error.
#Basically the center value is smoothed using the polynomial and then the window is shifted by one position - this again changes the center and a new polynomial is used to find the smoothed value of the new center value. This process continues until all values are smoothed.
#For points at the beginning and end of the signal - different processes like mirroring of data, repeating boundary values etc. For eg - 180 176 172 169 165. Extended window: 172 176 | 180 176 172 169 165 | 169 172 (Mirroring data)
# 180 180 | 180 176 172 169 165 | 165 165 (Repeating boundary values)


REP_MIN_PROMINENCE = 15    #Only flexion angles greater than 15 degrees will be counted as a rep. To avoid misdetection of reps
REP_MIN_DISTANCE   = 50    #min no of frames between detected repetitions to prevent false detections. For example - if the tracking angle is noisy it will lead to false rep detections.

#Here, we define landmark indices that we require. Depending on the side landmark indices are called. 
SIDE_LANDMARK_INDICES = dict(shoulder_l=11, shoulder_r=12, elbow_l=13, elbow_r=14, wrist_l=15, wrist_r=16, hip_l=23, hip_r=24, eye_l=2, eye_r=5, heel_l=29, heel_r=30)
JOINTS = ["hip_l", "shoulder_l", "elbow_l", "wrist_l", "eye_l", "heel_l", "hip_r", "shoulder_r", "elbow_r", "wrist_r", "eye_r", "heel_r"]

print("SIDE_LANDMARKS set up successfully")

def compute_angular_velocity(smooth_df: pd.DataFrame, fps: float, angle_col: str, window) -> np.ndarray:
    """
    First derivative of a smoothed angle signal, in degrees/second.
    Same savgol-deriv trick used for knee_velocity in the heel-slides pipeline.
    """
    return savgol_filter(smooth_df[angle_col].values, window, SMOOTH_POLY,
                         deriv=1, delta=1.0/fps)

def is_valid_shoulder(lms, lm_indices):
    for j in ("shoulder_l", "shoulder_r", "elbow_l", "elbow_r", "wrist_l", "wrist_r"):
        if lms[lm_indices[j]].visibility < 0.3:
            return False
    return True

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

def compute_pixels_per_cm(lm_df, height_cm, fps, calib_s=2.0, min_frames=15, video_path=VIDEO_PATH, side=SIDE):
    s = "l" if side == "left" else "r"
    ok = (lm_df[f"eye_{s}_vis"] >= VIS_THRESHOLD) & (lm_df[f"heel_{s}_vis"] >= VIS_THRESHOLD)
    if ok.sum() == 0:
        raise ValueError("No frames with visible eye and heel; cannot calibrate")
    cap = cv2.VideoCapture(video_path) #Here we open the video that needs to be processed 
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #Width of the frame that is extracted from the metadata of the video
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) #Height of the frame that is extracted from the metadata of the video
    print(frame_height)
    print(frame_width)
    
    x_coord = (lm_df[f"eye_{s}_x"] - lm_df[f"heel_{s}_x"])
    y_coord = (lm_df[f"eye_{s}_y"] - lm_df[f"heel_{s}_y"])
    span = np.hypot(x_coord, y_coord)
    print(span)
    whole = float(span[ok].median())
    first = ok & (lm_df.index < int(calib_s * fps))
    if first.sum() >= min_frames:
        init = float(span[first].median())
        if abs(init / whole - 1) > 0.05:
            print(f"WARNING: initial span differs {abs(init/whole-1)*100:.0f}% from " f"whole video; person may have moved toward/away from camera")
        chosen = init
    else:
        chosen = whole
    ref_cm = height_cm * (1 - EYE_TO_VERTEX_FRAC)
    px_per_cm = chosen / ref_cm
    print(f"  px_per_cm = {px_per_cm:.2f}")
    return px_per_cm

def arm_vertical_calculation(elbow, shoulder, side):
    down = np.array([0.0, 1.0])
    arm  = elbow - shoulder
    raw = _signed_angle_between_vectors(down, arm)
    arm_vertical = -raw if side == "left" else raw
    return arm_vertical

def _shoulder_elevation(shoulder_xy, hip_xy, px_per_cm):
    return float(np.linalg.norm(shoulder_xy - hip_xy) / px_per_cm)

def extract_landmarks(video_path: str, side: str = "right", visualize: bool = False) -> pd.DataFrame:
    #This function extracts key landmarks required to calculate angles like shoulder flexion, trunk lean, etc.
    lm_indices = SIDE_LANDMARK_INDICES
    records    = [] #The final feature set is saved here

    #The pose estimation model is loaded
    #An object basically contains information (data/attributes) and methods or functions that are used to operate on that data
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH) 
    #Configure the loaded model and set parameters for inference.
    #The Tasks API is used instead of the Solutions API 
    options = vision.PoseLandmarkerOptions(
        base_options                  = base_options, #The model is attached
        running_mode                  = vision.RunningMode.VIDEO, #Mediapipe Tasks supports multiple modes - image, video and live stream. Therefore, here we define the mode we want to operate in - which is the video mode.
        #During video processing (frame by frame processing), there is temporal tracking so the model keeps track of (or remembers) the previous frame
        num_poses                     = 1, #The main difference between this and solutions is that tasks can detect more than one people in the frame. Here, since we have set num_poses as 1 it will detect the person that gives the best confidence.
        min_pose_detection_confidence = 0.5, #Here, we define the minimum pose detection confidence, only detections with a confidence score equal to or above 0.5 will be considered, others will be ignored.
        min_pose_presence_confidence  = 0.5, #This parameter checks that after the detection of a person, how confident is it that the pose landmarks (joint coordinates) are actually present and reliable enough to use.
        min_tracking_confidence       = 0.5, #This parameter answers the question - I knew where the body landmarks were in the previous frame. How confident am I that I can keep following (tracking) those same landmarks correctly in the current frame?
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

    cap = cv2.VideoCapture(video_path) #Here we open the video that needs to be processed 
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #Width of the frame that is extracted from the metadata of the video
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) #Height of the frame that is extracted from the metadata of the video

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0 #Here we define the frames per second for the video. It is either extracted from the video data or is set to a default value of 30
    

    invalid_count = 0 #Count of frames that are treated as invalid
    paused = False #State of the video

    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    YELLOW = (0, 255, 255)
    WHITE = (255, 255, 255)

    with vision.PoseLandmarker.create_from_options(options) as detector: #Here we use the pose landmarker model that was previously loaded and identify key landmarks that are defined above 
        frame_idx = 0 #Frame counter 
        while True:
            if not paused:
                ret, frame = cap.read() #If the frame is not paused, we read the frame
                if not ret: #If no frame is being returned, we break out of the loop. Video Ended
                    break

                height, width, _ = frame.shape
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #OpenCV stores images in BGR format but mediapipe tasks requires the normal RGB format. Hence conversion is important
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb) #Mediapipe Tasks API cannot directly use NumPy/OpenCV arrays. It expects its own image object, so we convert rgb to image object. Frame is not an object - it is a numpy array which stores the pixel values of an image.
                """Mediapipe tasks api does not directly accept a frame and hence conversion is done to an image object. mp.Image object stores: Data - pixel data, image width, image height, image format (RGB). Methods: image_format(), numpy_view()"""
                #VIDEO mode needs a monotonically-increasing timestamp in ms. Time = frame number/fps and since the time is required in milliseconds - it is multiplied by 1000. Time stamps are important for tracking between frames, without time stamps tracking is impossible.
                ts_ms = int(frame_idx * 1000 / fps)
                result = detector.detect_for_video(mp_image, ts_ms) #Stores the coordinates for a particular frame
                #Here, we actually input the image and the time stamp into the mediapipe pose detection model. Internally mediapipe detects person, estimates pose landmarks and tracks coordinates using the previous frame.
                display_frame = cv2.resize(frame, (420, 720))
                display_height, display_width, _ = display_frame.shape
                row = {"frame": frame_idx, "detected": False}


                #result.pose_landmarks is a list of poses; [0] = first (only) person. lm is a mediapipe landmark object that contains: x coordinate, y coordinate, z coordinate, visibility and presence.
                if result.pose_landmarks and len(result.pose_landmarks) > 0: #This checks if the landmarks have been detected 
                    lms = result.pose_landmarks[0] #
                    valid = is_valid_shoulder(lms, lm_indices) #We check if the frame is valid or not 

                    if visualize: #Display the video with a skeleton overlay 
                        vis_frame = display_frame.copy()

                        #Draw the upper body skeleton. #POSE_CONNECTIONS is a predefined set of landmark pairs provided by MediaPipe. Each pair specifies which two body landmarks should be connected by a line to form the skeleton.
                        left_arm = [[11, 13], [13, 15]]
                        right_arm = [[12, 14], [14, 16]]
                        torso = [[11, 23], [12, 24]]
                        hip = [23, 24]

                        
                        pixel_ra1_s = [int(lms[right_arm[0][0]].x*display_width), int(lms[right_arm[0][0]].y*display_height)]
                        pixel_ra1_e = [int(lms[right_arm[0][1]].x*display_width), int(lms[right_arm[0][1]].y*display_height)]
                        pixel_ra2_s = [int(lms[right_arm[1][0]].x*display_width), int(lms[right_arm[1][0]].y*display_height)]
                        pixel_ra2_e = [int(lms[right_arm[1][1]].x*display_width), int(lms[right_arm[1][1]].y*display_height)]
                        pixel_tr_s = [int(lms[torso[0][0]].x*display_width), int(lms[torso[0][0]].y*display_height)]
                        pixel_tr_e = [int(lms[torso[0][1]].x*display_width), int(lms[torso[0][1]].y*display_height)]
                        pixel_hip_s = [int(lms[hip[0]].x*display_width), int(lms[hip[0]].y*display_height)]
                        pixel_hip_e = [int(lms[hip[1]].x*display_width), int(lms[hip[1]].y*display_height)]
                        pixel_la1_s = [int(lms[left_arm[0][0]].x*display_width), int(lms[left_arm[0][0]].y*display_height)]
                        pixel_la1_e = [int(lms[left_arm[0][1]].x*display_width), int(lms[left_arm[0][1]].y*display_height)]
                        pixel_la2_s = [int(lms[left_arm[1][0]].x*display_width), int(lms[left_arm[1][0]].y*display_height)]
                        pixel_la2_e = [int(lms[left_arm[1][1]].x*display_width), int(lms[left_arm[1][1]].y*display_height)]
                        pixel_tl_s = [int(lms[torso[1][0]].x*display_width), int(lms[torso[1][0]].y*display_height)]
                        pixel_tl_e = [int(lms[torso[1][1]].x*display_width), int(lms[torso[1][1]].y*display_height)]
                        cv2.line(vis_frame, pixel_ra1_s, pixel_ra1_e, (150, 150, 150), 2)
                        cv2.line(vis_frame, pixel_ra1_s, pixel_la1_s, (150, 150, 150), 2)
                        cv2.line(vis_frame, pixel_ra2_s, pixel_ra2_e, (150, 150, 150), 2)
                        cv2.line(vis_frame, pixel_hip_s, pixel_hip_e, (150, 150, 150), 2)
                        cv2.line(vis_frame, pixel_tr_s, pixel_tr_e, (150, 150, 150), 2)
                        cv2.line(vis_frame, pixel_la1_s, pixel_la1_e, (150, 150, 150), 2)
                        cv2.line(vis_frame, pixel_la2_s, pixel_la2_e, (150, 150, 150), 2)
                        cv2.line(vis_frame, pixel_tl_s, pixel_tl_e, (150, 150, 150), 2)

                        #Draw key joints
                        joint_labels = {
                            "hip_l": "HIP_L",
                            "hip_r": "HIP_R",
                            "shoulder_l": "SHO_L",
                            "shoulder_r": "SHO_R",
                            "elbow_l": "ELB_L",
                            "elbow_r": "ELB_R",
                            "wrist_l": "WRI_L",
                            "wrist_r": "WRI_R",
                            "eye_l": "EYE_L", 
                            "eye_r": "EYE_R",
                            "heel_r": "HEL_R",
                            "heel_l": "HEL_L"
                        }
                        #lm_indices is a dictionary that stores the mapping between the joint names and their corresponding MediaPipe landmark indices.
                        #.items() returns the joint and index of the landmark. Joint stores the joint name and idx stores the index of the mediapipe landmark
                        for joint, idx in lm_indices.items():
                            lm = lms[idx] #The landmark object - lm, contains several attributes like .x, .y, .visibility, etc.
                            px = int(lm.x * display_width) #Convert the x coordinate into pixel coordinates by mutliplying it with the width of the frame
                            py = int(lm.y * display_height) #Convert the y coordinate into pixel coordinates by multiplying it with the height of the frame
                            colour = GREEN if valid else RED
                            cv2.circle(vis_frame, (px, py), 8, colour, -1) #Show a circle on the px and py coordinates
                            cv2.putText(vis_frame, joint_labels[joint],  #Display the joint name on screen close to the pixel coordinates of the landmark for a particular joint
                                        (px + 10, py - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, colour, 2)

                        #Here we print if the frame is valid or not 
                        status_text = "VALID" if valid else "REJECTED" #If the validity condition is passed the following text will be displayed on the screen. 
                        cv2.putText(vis_frame, status_text, (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    GREEN if valid else RED, 2)

                        #Live angles. #lms is the landmark object that has attributes like .x, .y and .visibility. lm_indices["hip"] gives us the landmark of the hip, which might be 23. We could have written lms[23] 
                        #directly but writing it this way makes it more readable. hip_lm contains all the information required (such x, y and visibility data) about the hip landmark of the detected side 
                        scale = np.array([frame_width, frame_height])
                        s="l" if side=="left" else "r"
                        hip_arr  = np.array([lms[lm_indices[f"hip_{s}"]].x, lms[lm_indices[f"hip_{s}"]].y]) * scale
                        sho_arr  = np.array([lms[lm_indices[f"shoulder_{s}"]].x, lms[lm_indices[f"shoulder_{s}"]].y]) * scale
                        elb_arr  = np.array([lms[lm_indices[f"elbow_{s}"]].x, lms[lm_indices[f"elbow_{s}"]].y]) * scale
                        wri_arr  = np.array([lms[lm_indices[f"wrist_{s}"]].x, lms[lm_indices[f"wrist_{s}"]].y]) * scale

                        def cal_angle(a, b, c): #Here we calculate the angles to display on the screen
                            ba = a - b; bc = c - b
                            d  = np.linalg.norm(ba)*np.linalg.norm(bc)+1e-8
                            return float(np.degrees(
                                np.arccos(np.clip(np.dot(ba,bc)/d,-1,1))))
                        
                        sho_angle = signed_angle_calculation_3pt(hip_arr, sho_arr, elb_arr)
                        arm_trunk = -sho_angle if side == "left" else sho_angle
                        #print("Shoulder Angle wrt the trunk:", arm_trunk)
                        elb_angle = cal_angle(sho_arr, elb_arr, wri_arr)
                        arm_vertical = arm_vertical_calculation(elb_arr, sho_arr, side)

                        #Display the calculated angles 
                        cv2.putText(vis_frame,
                            f"Sho_tru: {arm_trunk:.1f}  Elbow: {elb_angle:.1f}  Sho_Vert: {arm_vertical:.1f}",
                            (20, display_height - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2)
                        other_s = "r" if s == "l" else "l"

                        #Display the visibility of key angles 
                        sho_vis = lms[lm_indices[f"shoulder_{s}"]].visibility
                        elb_vis = lms[lm_indices[f"elbow_{s}"]].visibility
                        wri_vis = lms[lm_indices[f"wrist_{s}"]].visibility
                        hip_vis = lms[lm_indices[f"hip_{s}"]].visibility
                        hip_other_vis = lms[lm_indices[f"hip_{other_s}"]].visibility
                        shoulder_other_vis = lms[lm_indices[f"shoulder_{other_s}"]].visibility
                        cv2.putText(vis_frame,f"Vis - Sho:{sho_vis:.2f}  " f"Elb:{elb_vis:.2f}  Wri:{wri_vis:.2f} Hip:{hip_vis:.2f}", (20, display_height - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
                        
                        #Frame Counter 
                        cv2.putText(vis_frame, f"Frame: {frame_idx}  Invalid: {invalid_count}", (display_width - 300, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)

                        if paused:
                            cv2.putText(vis_frame, "PAUSED", (display_width//2 - 60, 40),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2)

                        cv2.imshow("Shoulder Abduction Extraction", vis_frame)

                    if valid:
                        #lm_indices is a dictionary that stores the mapping between the joint names and their corresponding MediaPipe landmark indices.
                        #.items() returns the joint and index of the landmark. Joint stores the joint name and idx stores the index of the mediapipe landmark
                        for joint, idx in lm_indices.items():
                            lm = lms[idx]
                            row[f"{joint}_x"]   = lm.x #Each landmark object - lm, contains several attributes such as .x, .y, .visibility, etc.
                            row[f"{joint}_y"]   = lm.y
                            row[f"{joint}_vis"] = lm.visibility
                        row["detected"] = True
                    else:
                        for joint in lm_indices:
                            row[f"{joint}_x"]   = np.nan
                            row[f"{joint}_y"]   = np.nan
                            row[f"{joint}_vis"] = 0.0
                        invalid_count += 1
                else: #This condition runs when no human is detected in the frame and all the landmarks are 0
                    for joint in lm_indices:
                        row[f"{joint}_x"]   = np.nan
                        row[f"{joint}_y"]   = np.nan
                        row[f"{joint}_vis"] = 0.0

                records.append(row) #Here we append data that is extracted
                frame_idx += 1 #Frame number is incremented

                #Basic commands
            if visualize:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("  Visualiser quit early by user")
                    break
                elif key == ord(' '):
                    paused = not paused
                elif key == ord('s'):
                    fname = f"extract_frame_{frame_idx}.jpg"
                    cv2.imwrite(fname, vis_frame)
                    print(f"  Screenshot saved → {fname}")

        if visualize:
            cv2.destroyAllWindows() #After the video ends, we close the file

    cap.release()

    df = pd.DataFrame(records).set_index("frame") #pandas converts list into a table and the frame number is set as the index
    #Fill short detection gaps (≤ 5 consecutive frames). Here we select columns ending in _x and _y and fill small gaps by either forward filling or backward filling.
    #In forward filling the values are filled with a known value before it. This is done for a maximum of 5 consecutive frames.
    #In backward filling the values are filled with a known value after it. Forward fill handles missing middle values and backward fill handles missing beginning values.
    df.attrs["frame_width"]  = frame_width
    df.attrs["frame_height"] = frame_height
    df.attrs["fps"]          = fps

    coord_cols = [c for c in df.columns if c.endswith("_x") or c.endswith("_y")]
    df[coord_cols] = df[coord_cols].interpolate(method="linear", limit=5, limit_direction="both")
    if df[coord_cols].isna().any().any():
        print("  WARNING: gaps > 5 frames remain; filling so smoothing does not break")
        df[coord_cols] = df[coord_cols].ffill().bfill()

    detection_rate = df["detected"].mean() * 100 #The detected column of the data frame contains T or F. T = 1, F = 0. Example: [1, 1, 0, 0, 1], we then take the mean of these values and multiply that by 100 to get the detection rate.
    print(f"{frame_idx} frames | detection rate: {detection_rate:.1f}%")
    print(f"Spatial validation: {invalid_count} frames rejected "
          f"({invalid_count/frame_idx*100:.1f}% of detected frames)")
    if detection_rate < 70:
        print("  WARNING: low detection rate — check video angle / lighting")

    return df

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

#Currently the dataframe contains raw joint coordinates, we need joint angles therefore we calculate them using trignometry
def _angle_3pt(a, b, c): #Here we calculate the angle formed by three points, the angle is calculated at point B
#In the following step we get two vectors. Formula to get the angle - Cos theta = (ba.bc)/(|ba||bc|)
    """Angle at point b formed by vectors b→a and b→c."""
    ba = a - b; bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc)/denom, -1, 1))))

def signed_angle_calculation_3pt(a, b, c): #We do this so that we can calculate the negative angle when the patient while descending down moves into extension (past the neutral or 0 degree mark). 
    #This is specifically only for the shoulder angle.
    ba = a-b
    bc = c-b
    cross = ba[0]*bc[1]-ba[1]*bc[0]
    dot = np.dot(ba, bc)
    return float(np.degrees(np.arctan2(cross, dot)))

#Trunk lean basically measures how much the torso leans forward
def _signed_angle_between_vectors(v1, v2):
    """
    Signed angle from v1 to v2, in degrees. Positive = v1 is rotated
    counter-clockwise from v2 (in image coordinates); negative = clockwise.
    Same cross/dot + arctan2 approach as signed_angle_calculation_3pt,
    just applied to two vectors directly instead of three points.
    """
    cross = v1[0]*v2[1] - v1[1]*v2[0]
    dot = np.dot(v1, v2)
    return float(np.degrees(np.arctan2(cross, dot)))

def _trunk_lean(sho_l, sho_r, hip_l, hip_r, side: str):
    """
    Signed angle between the spine vector (hip→shoulder) and vertical.
    For a perfectly upright patient this is ~0°.
    Sign convention: positive = backward lean (common compensation in
    shoulder flexion), negative = forward lean.
    Sign is flipped for the right side to keep the convention consistent
    regardless of which way the patient faces the camera — same reasoning
    as the shoulder_angle sign flip.
    """
    mid_hip = (hip_l + hip_r) / 2
    mid_sho = (sho_l + sho_r) / 2
    dx = mid_sho[0] - mid_hip[0]
    dy = mid_sho[1] - mid_hip[1]            # image y points down
    lean = np.degrees(np.arctan2(dx, -dy))  # + = top of trunk leans toward image right
    lean_away = -lean if side == "left" else lean   # + = leaning away from working arm
    return lean_away

def shoulder_tilt_raw(sho_l, sho_r, hip_l, hip_r, side):
    """Shoulder-line tilt minus hip-line tilt, in degrees. + = working shoulder higher."""
    sho_tilt = np.degrees(np.arctan2(sho_r[1] - sho_l[1], sho_r[0] - sho_l[0]))
    hip_tilt = np.degrees(np.arctan2(hip_r[1] - hip_l[1], hip_r[0] - hip_l[0]))
    tilt = sho_tilt - hip_tilt
    return -tilt if side == "left" else tilt   # verify the sign on a known frame

def shoulder_tilt_calculation(sho_l, sho_r, hip_l, hip_r, baseline, side):
    dx, dy = sho_r - sho_l      #Here we calculate the how much the working shoulder is raised relative to the resting shoulder, while compensating for the tilt of the hips. 
    #dx measures the horizontal difference between the right and the left shoulders and dy measures the vertical difference. np.arctan2 calculates the angle of the line joining the left
    #and right shoulders. np.degrees converts radians into degrees                 
    sho_tilt = np.degrees(np.arctan2(dy, dx))
    dx, dy = hip_r - hip_l
    hip_tilt = np.degrees(np.arctan2(dy, dx))
    tilt = sho_tilt - hip_tilt  #subtract the hip tilt from the shoulder tilt. We need to distinguish the entire body tilting from only the shoulder tilting. The shoulder will tilt if the hip/pelvis tilts.
    #we need to find the tilt of the shoulder relative to the hip/pelvis.
    tilt_corrected = tilt - baseline    
    if side=="left":
        tilt_corrected = -tilt_corrected
    return tilt_corrected 
        
def compute_video_baseline(smooth_df, reps, states_by_rep, col="shoulder_height",
                           vis_cols=("shoulder_vis", "hip_vis")):
    vals = []
    for rep in reps:
        rep_df = smooth_df.loc[rep["start_frame"]:rep["end_frame"]].copy()
        rep_df["state"] = states_by_rep[rep["rep_id"]]
        mask = (rep_df["state"] == 1) & _visible(rep_df, list(vis_cols))
        vals.extend(rep_df.loc[mask, col].values)
    return float(np.median(vals)) if vals else np.nan

def calculate_angles(lm_df: pd.DataFrame, side: str, px_per_cm) -> pd.DataFrame:
    """
    Calculates five angles for shoulder flexion:
    1. shoulder_angle - hip-shoulder-elbow. Primary tracking angle. Starts near 180° (arm at side), decreases as arm raises.
    2. elbow_angle - shoulder-elbow-wrist. Should remain near 180° throughout. Deviation = elbow bending compensation.
    3. trunk_angle - angle of spine vector relative to vertical. Should remain near 0° throughout. Increase = backward trunk lean compensation.
    4. shoulder_height - normalised vertical position of shoulder relative to hip. Increase during movement = shoulder shrug compensation.
    5. wrist_elbow_gap - normalised vertical offset of wrist above elbow. Positive = wrist correctly above elbow (overhead). Negative = wrist dropped below elbow (resting position - arm at the side).
    """
    s="l" if side=="left" else "r"
    rows = [] #Create an empty array that will later store the calculated angles
    for frame_idx, r in lm_df.iterrows(): #It saves the coordinates and loops through each row of data. lm_df is a pandas dataframe and iterrows() returns the index and data in a particular row.
    #frame_idx stores the index of the row and r stores the data in the row
        hip      = np.array([r[f"hip_{s}_x"], r[f"hip_{s}_y"]]) #Extract the shoulder coordinates from each row of data
        shoulder = np.array([r[f"shoulder_{s}_x"], r[f"shoulder_{s}_y"]])
        sho_l = np.array([r[f"shoulder_l_x"], r[f"shoulder_l_y"]])
        sho_r = np.array([r[f"shoulder_r_x"], r[f"shoulder_r_y"]])
        hip_r = np.array([r[f"hip_r_x"], r[f"hip_r_y"]])
        hip_l = np.array([r[f"hip_l_x"], r[f"hip_l_y"]])
        elbow    = np.array([r[f"elbow_{s}_x"], r[f"elbow_{s}_y"]])
        wrist    = np.array([r[f"wrist_{s}_x"], r[f"wrist_{s}_y"]])

        sho_angle = signed_angle_calculation_3pt(hip, shoulder, elbow) #This is basically the angle between the upper arm and the trunk. In image coordinates, the image right (left side for the video) the sign convention is negative.
        arm_trunk_angle = -sho_angle if side == "left" else sho_angle
        arm_vertical_angle = arm_vertical_calculation(elbow, shoulder, side)
        tilt_angle = shoulder_tilt_raw(sho_l, sho_r, hip_l, hip_r, side)
        o = "r" if side == "left" else "l"
        rows.append({
            "frame": frame_idx, #The functions are called and the data is added to the rows
            "arm_trunk_angle": arm_trunk_angle,
            "arm_vertical_angle": arm_vertical_angle,
            "shoulder_tilt_angle": tilt_angle,
            "elbow_angle": _angle_3pt(shoulder, elbow, wrist),
            "trunk_angle": _trunk_lean(sho_l, sho_r, hip_l, hip_r, side),
            "shoulder_height": _shoulder_elevation(shoulder, hip, px_per_cm),
            "shoulder_vis": r[f"shoulder_{s}_vis"],
            "elbow_vis": r[f"elbow_{s}_vis"],
            "wrist_vis": r[f"wrist_{s}_vis"],
            "hip_vis": r[f"hip_{s}_vis"],
            "hip_other_vis": r[f"hip_{o}_vis"],
            "shoulder_other_vis": r[f"shoulder_{o}_vis"]
        })

    out = pd.DataFrame(rows).set_index("frame")
    out["arm_trunk_angle"] = np.degrees(np.unwrap(np.radians(out["arm_trunk_angle"].values))) #The unwrap function is used to prevent sudden jumps even though the motion is smooth (example: +180 to -180)
    #We convert to radians since the unwrap function works only on radians
    out["arm_vertical_angle"] = np.degrees(np.unwrap(np.radians(out["arm_vertical_angle"].values)))
    return out

#This function takes the raw angles that were calculated and smooths them using the savitzky golay filter
def smooth_angles(angle_df: pd.DataFrame) -> pd.DataFrame:
    out = angle_df.copy()
    for col in ["arm_trunk_angle", "arm_vertical_angle", "shoulder_tilt_angle", "elbow_angle", "trunk_angle",
                "shoulder_height"]:
        out[col] = savgol_filter(angle_df[col].values, SMOOTH_WINDOW, SMOOTH_POLY)
    return out


def detect_reps(arm_trunk_angles: np.ndarray) -> list[dict]:
    """
    similar to shoulder flexion, we find the peaks in the data. The peaks signify 
    the mid point of the repetition. The repetition starts at near 0 degrees. And when the arm is raised overhead
    the angle increases and reaches a peak value, following which it descends back down. We use argmin to find the local minimas
    near a particular peak so that we can identify a repetition's boundaries. 
    """
    peaks, _ = find_peaks(
        arm_trunk_angles,
        prominence = REP_MIN_PROMINENCE, #Rep prominence is the minimum change in angle that will be counted as a rep
        distance   = REP_MIN_DISTANCE,
    )

    if len(peaks) == 0:
        print("No rep peaks found")
        return []
    
    boundary_rests = [int(np.argmin(arm_trunk_angles[:peaks[0]]))] #This is done so that we can accurately detect peaks of the initial and final reps. find peaks tends to miss these two reps since it does not have any data before it to compare with. 
    #Peaks = Shoulder Flexion Angle = Resting Position. Here we find the maxima
    #Find the max angle before the first valley. That is designated as the initial rep boundary
    #np.argmax returns the index of the maxima. Basically we find the all the shoulder_angles before the valley point and then using np.argmax extract the maxima from those points.

    #Move through each rep/ valleys
    for i in range(len(peaks)-1): #enumerate returns the index and the value at that index
        window = arm_trunk_angles[peaks[i]:peaks[i+1]] #We are comparings pairs of valleys. Between two consecutive valleys there is one peak which marks the end of rep 1 and the start of rep 2
        #We extract all the shoulder angles between the two valleys
        local_min = int(np.argmin(window)) + peaks[i]
        boundary_rests.append(local_min)
        
    last_window = arm_trunk_angles[peaks[-1]:]
    boundary_rests.append(int(np.argmin(last_window)) + peaks[-1])

    reps=[]
    for i, peak in enumerate(peaks):
        start = boundary_rests[i]
        end = boundary_rests[i+1]

        rep_range = arm_trunk_angles[peak] - arm_trunk_angles[start] #Calculate the rep range 

        if rep_range < REP_MIN_PROMINENCE:
            print(f"Skipping shallow rep at frame {start}"
                  f"(range={rep_range:.1f}° < {REP_MIN_PROMINENCE}°)")
            continue

        #Basic frame data is added 
        reps.append({
            "rep_id":       i + 1,
            "start_frame":  int(start),
            "peak_frame": peak,
            "end_frame":    int(end),
        })

    print(f"{len(reps)} rep(s) detected")
    return reps


def assign_states(arm_trunk_angles, start, peak, end):
    """
    Labels each frame in [start, end] with a state 1-4.

    S1 - Resting: arm at side, shoulder near minimum angle
    S2 - Lifting: arm rising, shoulder angle increasing (after valley)
    S3 - Peak: arm at maximum height, shoulder angle at maximum
    S4 - Lowering: arm returning to side, shoulder angle decreasing 
    """
    rep_angles = arm_trunk_angles[start:end+1] #Shoulder angles of the rep are returned 
    rep_max = rep_angles.max() #Compute the max angle
    rep_min = rep_angles.min() #Compute the min angle - maximum flexion

    S1_EXIT = 10   #degrees below rep_max to exit S1
    S3_EXIT = 10   #degrees above rep_min to exit S3

    thresh_low  = rep_min + S1_EXIT   #S1: within 10° of resting position
    thresh_high   = rep_max - S3_EXIT   #S3: within 10° of peak elevation
    #Since in this exercise the peaks are the overhead positions (maximum flexion we find the peaks instead of valleys in case of heel slides, SLR)
    peak_local = peak - start

    n = len(rep_angles)
    states = np.zeros(n, dtype=int)

    for i, angle in enumerate(rep_angles):
        if angle <= thresh_low:
            states[i] = 1            #S1 - Resting (arm at side)
        elif angle >= thresh_high:
            states[i] = 3            #S3 - Peak elevation
        elif i <= peak_local:
            states[i] = 2            #S2 - Lifting upward
        else:
            states[i] = 4            #S4 - Lowering back down

    return states


ANGLE_COLS = ["arm_trunk_angle", "arm_vertical_angle", "shoulder_tilt_corrected", "elbow_angle", "trunk_angle","shoulder_height", "arm_trunk_velocity", "arm_trunk_acceleration"] #An array is defined that stores the 5 main angles
ANGLE_LABELS = ["arm_trunk", "arm_vertical", "shoulder_tilt", "elbow", "trunk", "sho_height", "velocity", "acceleration"] #Stores the labels
N_STATES = 4 #States of the exercise 
VIS_COLS = {
    "arm_trunk_angle": ["hip_vis", "shoulder_vis", "elbow_vis"],
    "arm_vertical_angle": ["shoulder_vis", "elbow_vis"],
    "shoulder_tilt_corrected": ["shoulder_vis", "hip_vis", "shoulder_other_vis", "hip_other_vis"],
    "elbow_angle": ["shoulder_vis", "elbow_vis", "wrist_vis"],
    "trunk_angle": ["hip_vis", "shoulder_vis", "shoulder_other_vis", "hip_other_vis"], #Trunk lean and shoulder tilt depend on both shoulders and hips
    "shoulder_height": ["hip_vis", "shoulder_vis"],
    "arm_trunk_velocity": ["hip_vis", "shoulder_vis", "elbow_vis"],
    "arm_trunk_acceleration": ["hip_vis", "shoulder_vis", "elbow_vis"],
} #A visibility threshold is added to ensure that only frames that have a visibility above the given thresholds for the key joints required would be considered and the frames with lower
#visibility will be rejected.

#Statistics that need to be calculated
STAT_FUNCS = {
    #"mean":  np.mean,
    "min":   np.min,
    "max":   np.max,
    "range": lambda x: float(np.max(x) - np.min(x)),
    #"std":   np.std,
}


def compute_rep_features(angle_df, rep, states, patient_id, baseline: float, fps):
    """
    Computes per-state statistics for a single rep.
    Also computes derived features relative to S1 baseline to
    account for inter-patient anatomical variation (kyphosis,
    scapular position, natural trunk lean at rest).
    """
    start = rep["start_frame"] #Extract the rep boundaries 
    end = rep["end_frame"]

    #Extract the frames belonging to this rep
    rep_angle_df = angle_df.loc[start:end]
    #Basic metadata is added 
    row = {
        "patient_id":   patient_id,
        "rep_id":       rep["rep_id"],
        "start_frame":  start,
        "end_frame":    end,
        "peak_frame": rep["peak_frame"],
        "rep_duration": end - start + 1,
    }

    for s in range(1, N_STATES + 1):
        mask = (states == s) #for eg: if states == 1, [1, 1, 2, 2, 2, 3, 3, 4, 4] then [True, True, False, False, False, False, False, False, False]
        state_frames = rep_angle_df[mask] #Stores the frames belonging to a particular state

        row[f"S{s}_duration"] = int(mask.sum()) #Stores the summation of those frame. Basically stores how many frames did a particular state last

        for col, label in zip(ANGLE_COLS, ANGLE_LABELS): #zip() pairs values together ("knee_angle", "knee"), ("hip_angle", "hip"), etc.
            vis_cols = VIS_COLS.get(col, [])
            existing_cols = [c for c in vis_cols if c in state_frames.columns]
            if existing_cols:
                    vis_mask = (state_frames[existing_cols] >= VIS_THRESHOLD).all(axis=1)
                    gated_frames = state_frames[vis_mask]
                    row[f"S{s}_{label}_vis_frames"] = int(vis_mask.sum())
            else:
                    gated_frames = state_frames
                    row[f"S{s}_{label}_vis_frames"] = len(state_frames)

            vals = (
                gated_frames[col].values
                if len(gated_frames) > 0
                else np.array([np.nan])
            )

            for stat_name, stat_fn in STAT_FUNCS.items():
                row[f"S{s}_{label}_{stat_name}"] = (
                    float(stat_fn(vals)) if len(vals) > 0 else np.nan
                )

    #Derived features relative to S1 baseline
    #These are the features that matter for standing exercises where postural variation (kyphosis etc.) shifts the absolute baseline.
    #By expressing as change from S1 we cancel out anatomical variation and keep only the movement quality signal.

    #How far the arm actually raised from its own starting position
    # his is the ROM achieved, expressed in degrees of shoulder movement
    row["arm_trunk_rom"] = row["S3_arm_trunk_max"] - row["S1_arm_trunk_min"]#Changed the formula for shoulder rom from S3_max - S1_min. since most of the patients move into extension while lowering their hand back down, giving a negative min angle
    #for state 1. Therefore, the min angle instead of being subtracted from the max angle of s3, it gets added to it. Thereby, giving inflated angles. To avoid this we consider 0 degrees as the neutral point.

    #How much trunk lean developed beyond the patient's own resting posture
    #A kyphotic patient starts with higher trunk_angle - expressing change from S1 means we measure ADDITIONAL lean caused by compensation, not the lean that was already there at rest
    row["trunk_compensation_s2"] = (
        row["S2_trunk_max"] - row["S1_trunk_max"]
    )
    row["trunk_compensation_s3"] = (
        row["S3_trunk_max"] - row["S1_trunk_max"] #S{state}_{label}_{stat}
    )
    row["range_from_lean_compensation"] = np.nanmax([row["S1_arm_vertical_max"], row["S2_arm_vertical_max"], row["S3_arm_vertical_max"], row["S4_arm_vertical_max"]])
    - np.nanmax([row["S1_arm_trunk_max"], row["S2_arm_trunk_max"], row["S3_arm_trunk_max"], row["S4_arm_trunk_max"]])

    #How much the elbow bent beyond its starting position. S1 elbow may already be slightly less than 180° due to natural arm position
    #We measure deviation from that patient's own baseline
    row["elbow_deviation_s2"] = abs(
        row["S2_elbow_max"] - row["S1_elbow_max"]
    )
    row["elbow_deviation_s3"] = abs(
        row["S3_elbow_max"] - row["S1_elbow_max"]
    )

    #How much the shoulder shrugged from its own resting height. shoulder_height increases as shoulder rises toward the ear
    row["shoulder_shrug_s2"] = (
        row["S2_sho_height_max"] - baseline
    )
    row["shoulder_shrug_s3"] = (
        row["S3_sho_height_max"] - baseline
    )

    #Speed symmetry - is the patient lowering with control?
    #Ratio > 1 means ascent took longer than descent (uncontrolled drop)
    #Ratio < 1 means ascent faster than descent (momentum on the way up)
    row["s2_s4_speed_ratio"] = (
        row["S2_duration"] / (row["S4_duration"] + 1e-8)
    )
    return row

#Processes everything and calls all the functions defined above
#Basic input arguments are provided here which would be required to process the video and save its data in a pandas dataframe such as the path of the video,
#patient id, the side of the exercise. One video is processed here.
def process_video(video_path, patient_id, side="right", output_csv=None, visualize=False, patient_height_cm=PATIENT_HEIGHT_CM):
    print(f"\n{'='*60}\nVideo : {os.path.basename(video_path)}\nPatient : {patient_id} | Side: {side}\n{'='*60}")

    print("\n[1/6] Extracting landmarks")
    lm_df = extract_landmarks(video_path, side, visualize=visualize)
    fps = lm_df.attrs["fps"]
    W, H = lm_df.attrs["frame_width"], lm_df.attrs["frame_height"]

    print("[2/6] Smoothing landmarks, converting to pixels")
    lm_df = smooth_landmarks(lm_df)
    for j in JOINTS:
        lm_df[f"{j}_x"] *= W
        lm_df[f"{j}_y"] *= H
    px_per_cm = compute_pixels_per_cm(lm_df, patient_height_cm, fps)

    print("[3/6] Calculating angles")
    angle_df  = calculate_angles(lm_df, side, px_per_cm)

    print("[4/6] Smoothing")
    smooth_df = smooth_angles(angle_df)
    dw = int(0.7 * fps); dw += (dw % 2 == 0)      # odd window, ~0.7 s
    smooth_df["arm_trunk_velocity"] = compute_angular_velocity(smooth_df, fps, "arm_trunk_angle", dw)
    smooth_df["arm_trunk_acceleration"] = savgol_filter(
    smooth_df["arm_trunk_angle"].values, dw, SMOOTH_POLY, deriv=2, delta=1.0/fps)

    print("[5/6] Detecting reps")
    arm_trunk_angle = smooth_df["arm_trunk_angle"].values
    reps = detect_reps(arm_trunk_angle)
    if not reps:
        return pd.DataFrame()

    print("[6/6] Segmenting states & computing features")
    states_by_rep = {r["rep_id"]: assign_states(arm_trunk_angle, r["start_frame"], r["peak_frame"], r["end_frame"])
                     for r in reps}
    baseline = compute_video_baseline(smooth_df, reps, states_by_rep)
    tilt_baseline = compute_video_baseline(
    smooth_df, reps, states_by_rep,
    col="shoulder_tilt_angle",
    vis_cols=("shoulder_vis", "hip_vis", "shoulder_other_vis", "hip_other_vis"),
)
    smooth_df["shoulder_tilt_corrected"] = smooth_df["shoulder_tilt_angle"] - tilt_baseline

    all_rows = []
    for rep in reps:
        states = states_by_rep[rep["rep_id"]]
        all_rows.append(compute_rep_features(smooth_df, rep, states, patient_id, baseline, fps=30.0))
        s = arm_trunk_angle[rep["start_frame"]:rep["end_frame"]+1]
        print(f"Rep {rep['rep_id']:>2} | frames {rep['start_frame']}–{rep['end_frame']} "
              f"| shoulder {s.min():.1f}°→{s.max():.1f}° | "
              + "  ".join(f"S{k}={int((states==k).sum())}f" for k in range(1, 5)))

    result_df = pd.DataFrame(all_rows)

    if output_csv:
        if os.path.isfile(output_csv):
            existing = pd.read_csv(output_csv, usecols=["patient_id"])
            if patient_id in set(existing["patient_id"]):
                print(f"  {patient_id} already in CSV; not appending")
                return result_df
        result_df.to_csv(output_csv, mode="a", header=not os.path.isfile(output_csv), index=False)
    return result_df

def plot_rep_debug(smooth_df, reps, shoulder):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    STATE_COLORS = {1: "steelblue", 2: "orange", 3: "tomato",    4: "mediumseagreen"}
    STATE_NAMES = {1: "S1 Resting",   2: "S2 Lifting", 3: "S3 Peak",      4: "S4 Lowering"}

    fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True)
    frames = smooth_df.index.values

    for ax, col, label in zip(axes,
        ["shoulder_angle", "elbow_angle", "trunk_angle", "shoulder_height"],
        ["Shoulder (°)", "Elbow (°)", "Trunk lean (°)", "Shoulder height (norm)"]):
        ax.plot(frames, smooth_df[col].values, lw=1.5, color="navy")
        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, alpha=0.3)

    for rep in reps:
        start = rep["start_frame"]
        end = rep["end_frame"]
        peak = rep["peak_frame"]
        states = assign_states(shoulder, start, peak, end)

        for i, s in enumerate(states):
            f = start + i
            axes[0].axvspan(f, f+1, alpha=0.25, color=STATE_COLORS[s], linewidth=0)

        axes[0].axvline(peak, color="red", lw=1, ls="--", alpha=0.7)
        axes[0].text(peak, shoulder[peak] - 3, f"R{rep['rep_id']}", ha="center", fontsize=7, color="red")

    patches = [mpatches.Patch(color=c, label=n, alpha=0.5) for s, (c, n) in enumerate(zip(STATE_COLORS.values(), STATE_NAMES.values()), 1)]
    axes[0].legend(handles=patches, fontsize=8, loc="upper right")
    axes[0].set_title("Shoulder angle with rep states", fontsize=10)
    axes[4].set_xlabel("Frame")

    plt.tight_layout()
    plt.savefig("shoulder_flexion_debug.png", dpi=150)
    plt.show()
    print("Debug plot saved → shoulder_flexion_debug.png")

if __name__ == "__main__":
    df = process_video(
        video_path = VIDEO_PATH,
        patient_id = PATIENT_ID,
        side = SIDE,
        output_csv = OUTPUT_CSV,
        visualize = True
    )



