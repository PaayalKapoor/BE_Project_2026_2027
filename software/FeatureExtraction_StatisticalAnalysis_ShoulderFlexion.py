#This is the feature extraction code for shoulder flexion. This exercise is recorded from the side view and focuses on the range of motion of the shoulder joint.
#The key angles in this exercise include - 1. Shoulder Flexion, 2. Elbow Angle, 3. Trunk Lean Angle, 4. Wrist position relative to elbow, 5. Shoulder Elevation
#Here we basically extract important coordinates from each frame which are used to calculate the angles formed in that frame using the 3 point formula.
#The tracking angle is shoulder flexion. Shoulder flexion angle starts at 180 (peak) and can go upto 60 degrees (valley)
#For rep detection we use a tracking angle which undergoes the most amount of movement. Peaks or valleys are used to detect reps. In this exercise the peaks are the boundaries (starting and ending) of the reps.
#The valleys (lower angle values) are the midpoint of the exercise. Once the rep is detected, the frames for a particular rep are isolated. The rep frames are further divided into states. The state classification is done so that errors
#in a particular rep can be specifiec to a particular part of the exercise.
#S1 - resting, S2 - Ascending up, S3 - Hand up, S4 - Descending down. The statistics for a particular state are calculated and taken as a feature. 
#Certain factors like rom and trunk lean are calculated keeping in mind the baseline which is calculated when the body is at rest. This is to avoid giving false errors in cases like kyphosis, where the spine is by default bent forward 
#changing the resting position of the shoulder from that of a normal person. 

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy.signal import savgol_filter, find_peaks
import numpy as np
import pandas as pd
import urllib.request
import os

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
VIDEO_PATH  = "videos/Shoulder_Flexion/Patient_127_R.mp4"
OUTPUT_CSV  = "docs/shoulder_flexion_data.csv"
PATIENT_ID  = "Patient_127_R"
SIDE        = "right"   #Here we provide the side that is facing the camera 

#The savitzky golay filter does not work on live stream videos since it requires a window of previous, current and future data points to estimate the middle smoothed value. 
#Smoothing. The angles are smoothed using the savitzky golay filter. This filter basically preserves the peaks and valleys in the data. For example - moving average takes the previous, current and future reading which is averaged.
#The problem with the moving average filter is that it ends up disturbing the overall shape of the signal, eg - 180, 150, 180 here 150 will be averaged to 170. Therefore moving averages tend to flatten peaks and valleys.
#Whereas the savitzky golay filter is a smoothing filter that reduces noise in data while preserving the overall shape, peaks, valleys and trends in the signal.
SMOOTH_WINDOW = 21 #Smoothing window is set to 21, must be odd so that the middle value can be found easily. Points before and after the center value help estimate the smoothed points. For noiser signals, the value should be higher.
#Here, we use 21 neighbouring points to estimate the center value. 10 before, current and 10 after
SMOOTH_POLY   = 3 #Fit a cubic polynomial through those 21 points. It uses something called Least Squares Regression. The algorithm tries many possible curves and picks the one with the smallest overall error.
#Basically the center value is smoothed using the polynomial and then the window is shifted by one position - this again changes the center and a new polynomial is used to find the smoothed value of the new center value. This process continues until all values are smoothed.
#For points at the beginning and end of the signal - different processes like mirroring of data, repeating boundary values etc. For eg - 180 176 172 169 165. Extended window: 172 176 | 180 176 172 169 165 | 169 172 (Mirroring data)
# 180 180 | 180 176 172 169 165 | 165 165 (Repeating boundary values)


REP_MIN_PROMINENCE = 15    #Only flexion angles greater than 15 degrees will be counted as a rep. To avoid misdetection of reps
REP_MIN_DISTANCE   = 50    #min no of frames between detected repetitions to prevent false detections. For example - if the tracking angle is noisy it will lead to false rep detections.

#Here, we define landmark indices that we require. Depending on the side landmark indices are called. 
SIDE_LANDMARK_INDICES = {
    "right": dict(hip=24, shoulder=12, elbow=14, wrist=16),
    "left":  dict(hip=23, shoulder=11, elbow=13, wrist=15),
}

print("SIDE_LANDMARKS set up successfully")

def is_valid_shoulder(lms, lm_indices):
    #This function is used to check whether the correct hand is being tracked. (This was a major issue in straight leg raises and although it is not very important here since only one arm would be 
    #visible at all times in the side view frame, however it has been added as an additional security.)
    shoulder_x = lms[lm_indices["shoulder"]].x
    shoulder_y = lms[lm_indices["shoulder"]].y
    elbow_x    = lms[lm_indices["elbow"]].x
    elbow_y    = lms[lm_indices["elbow"]].y
    wrist_x    = lms[lm_indices["wrist"]].x
    wrist_y    = lms[lm_indices["wrist"]].y

    #For shoulder flexion the elbow must stay near straight (170-180°) throughout the movement. If the elbow angle drops significantly it means either MediaPipe grabbed the wrong landmark or the patient
    #is bending their elbow as a compensation pattern.
    ba = np.array([shoulder_x - elbow_x, shoulder_y - elbow_y])
    bc = np.array([wrist_x    - elbow_x, wrist_y    - elbow_y])
    denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8
    cos_val = np.dot(ba, bc) / denom
    elbow_angle = float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))

    if elbow_angle < 90:
        return False
    
    #Here, we check the visibility of the wrist
    if lms[lm_indices["wrist"]].visibility < 0.3:
        return False

    return True


def extract_landmarks(video_path: str, side: str = "right", visualize: bool = False) -> pd.DataFrame:
    #This function extracts key landmarks required to calculate angles like shoulder flexion, trunk lean, etc.
    lm_indices = SIDE_LANDMARK_INDICES[side]
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

    #Here we define the connections we want to display on the screen. 
    UPPER_BODY_CONNECTIONS = [
        (11, 12),                   #shoulders
        (11, 13), (13, 15),         #left arm
        (12, 14), (14, 16),         #right arm
        (11, 23), (12, 24),         #torso sides
        (23, 24),                   #hips
    ]

    cap = cv2.VideoCapture(video_path) #Here we open the video that needs to be processed 
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0 #Here we define the frames per second for the video. It is either extracted from the video data or is set to a default value of 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #Width of the frame that is extracted from the metadata of the video
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) #Height of the frame that is extracted from the metadata of the video

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

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #OpenCV stores images in BGR format but mediapipe tasks requires the normal RGB format. Hence conversion is important
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb) #Mediapipe Tasks API cannot directly use NumPy/OpenCV arrays. It expects its own image object, so we convert rgb to image object. Frame is not an object - it is a numpy array which stores the pixel values of an image.
                """Mediapipe tasks api does not directly accept a frame and hence conversion is done to an image object. mp.Image object stores: Data - pixel data, image width, image height, image format (RGB). Methods: image_format(), numpy_view()"""
                #VIDEO mode needs a monotonically-increasing timestamp in ms. Time = frame number/fps and since the time is required in milliseconds - it is multiplied by 1000. Time stamps are important for tracking between frames, without time stamps tracking is impossible.
                ts_ms = int(frame_idx * 1000 / fps)
                result = detector.detect_for_video(mp_image, ts_ms) #Stores the coordinates for a particular frame
                #Here, we actually input the image and the time stamp into the mediapipe pose detection model. Internally mediapipe detects person, estimates pose landmarks and tracks coordinates using the previous frame.

                row = {"frame": frame_idx, "detected": False}


                #result.pose_landmarks is a list of poses; [0] = first (only) person. lm is a mediapipe landmark object that contains: x coordinate, y coordinate, z coordinate, visibility and presence.
                if result.pose_landmarks and len(result.pose_landmarks) > 0: #This checks if the landmarks have been detected 
                    lms = result.pose_landmarks[0] #
                    valid = is_valid_shoulder(lms, lm_indices) #We check if the frame is valid or not 

                    if visualize: #Display the video with a skeleton overlay 
                        vis_frame = frame.copy()

                        #Draw the upper body skeleton. #POSE_CONNECTIONS is a predefined set of landmark pairs provided by MediaPipe. Each pair specifies which two body landmarks should be connected by a line to form the skeleton.
                        for s_idx, e_idx in UPPER_BODY_CONNECTIONS: #Here s_idx and e_idx represents the start and end index between which the connections need to be made respectively.
                            if (s_idx < len(lms) and e_idx < len(lms) and  #This condition checks whether the landmarks are valid or not
                                lms[s_idx].visibility > 0.3 and #Visibility of the start and end index is checked
                                lms[e_idx].visibility > 0.3):
                                pixel_s = (int(lms[s_idx].x * width),
                                      int(lms[s_idx].y * height)) #Here we calculate the pixel coordinates of the starting landmark so that accordingly the coordinate can be displayed on the screen.
                                              #x coordinate is multiplied with the width of the frame and y coordinate is multiplied with the height of the frame
                                pixel_e = (int(lms[e_idx].x * width),
                                      int(lms[e_idx].y * height))
                                cv2.line(vis_frame, pixel_s, pixel_e, (150, 150, 150), 1) #Here we draw a line between the starting landmark and the ending landmark 
                                #cv2.line(image, start_point, end_point, color, thickness)
                        #Draw key joints
                        joint_labels = {
                            "hip":      "HIP",
                            "shoulder": "SHO",
                            "elbow":    "ELB",
                            "wrist":    "WRI",
                        }
                        #lm_indices is a dictionary that stores the mapping between the joint names and their corresponding MediaPipe landmark indices.
                        #.items() returns the joint and index of the landmark. Joint stores the joint name and idx stores the index of the mediapipe landmark
                        for joint, idx in lm_indices.items():
                            lm = lms[idx] #The landmark object - lm, contains several attributes like .x, .y, .visibility, etc.
                            px = int(lm.x * width) #Convert the x coordinate into pixel coordinates by mutliplying it with the width of the frame
                            py = int(lm.y * height) #Convert the y coordinate into pixel coordinates by multiplying it with the height of the frame
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
                        hip_arr  = np.array([lms[lm_indices["hip"]].x,
                                             lms[lm_indices["hip"]].y])
                        sho_arr  = np.array([lms[lm_indices["shoulder"]].x,
                                             lms[lm_indices["shoulder"]].y])
                        elb_arr  = np.array([lms[lm_indices["elbow"]].x,
                                             lms[lm_indices["elbow"]].y])
                        wri_arr  = np.array([lms[lm_indices["wrist"]].x,
                                             lms[lm_indices["wrist"]].y])

                        def cal_angle(a, b, c): #Here we calculate the angles to display on the screen
                            ba = a - b; bc = c - b
                            d  = np.linalg.norm(ba)*np.linalg.norm(bc)+1e-8
                            return float(np.degrees(
                                np.arccos(np.clip(np.dot(ba,bc)/d,-1,1))))

                        sho_angle = cal_angle(hip_arr, sho_arr, elb_arr)
                        elb_angle = cal_angle(sho_arr, elb_arr, wri_arr)

                        #Display the calculated angles 
                        cv2.putText(vis_frame,
                            f"Shoulder: {sho_angle:.1f}  Elbow: {elb_angle:.1f}",
                            (20, height - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2)

                        #Display the visibility of key angles 
                        sho_vis = lms[lm_indices["shoulder"]].visibility
                        elb_vis = lms[lm_indices["elbow"]].visibility
                        wri_vis = lms[lm_indices["wrist"]].visibility
                        cv2.putText(vis_frame,
                            f"Vis — Sho:{sho_vis:.2f}  "
                            f"Elb:{elb_vis:.2f}  Wri:{wri_vis:.2f}",
                            (20, height - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (200, 200, 200), 1)
                        
                        #Frame Counter 
                        cv2.putText(vis_frame,
                            f"Frame: {frame_idx}  Invalid: {invalid_count}",
                            (width - 300, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (200, 200, 200), 2)

                        if paused:
                            cv2.putText(vis_frame, "PAUSED",
                                        (width//2 - 60, 40),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        WHITE, 2)

                        cv2.imshow("Shoulder Flexion Extraction", vis_frame)

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
    coord_cols = [c for c in df.columns if c.endswith("_x") or c.endswith("_y")]
    df[coord_cols] = df[coord_cols].ffill(limit=5).bfill(limit=5) 

    detection_rate = df["detected"].mean() * 100 #The detected column of the data frame contains T or F. T = 1, F = 0. Example: [1, 1, 0, 0, 1], we then take the mean of these values and multiply that by 100 to get the detection rate.
    print(f"{frame_idx} frames | detection rate: {detection_rate:.1f}%")
    print(f"Spatial validation: {invalid_count} frames rejected "
          f"({invalid_count/frame_idx*100:.1f}% of detected frames)")
    if detection_rate < 70:
        print("  WARNING: low detection rate — check video angle / lighting")

    return df

#Currently the dataframe contains raw joint coordinates, we need joint angles therefore we calculate them using trignometry
def _angle_3pt(a, b, c): #Here we calculate the angle formed by three points, the angle is calculated at point B
#In the following step we get two vectors. Formula to get the angle - Cos theta = (ba.bc)/(|ba||bc|)
    """Angle at point b formed by vectors b→a and b→c."""
    ba = a - b; bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc)/denom, -1, 1))))

#Trunk lean basically measures how much the torso leans forward
def _trunk_lean(shoulder_xy, hip_xy):
    """
    Angle between the spine vector (hip→shoulder) and vertical.
    For a perfectly upright patient this is 0°.
    As the patient leans backward (common compensation in shoulder flexion)
    this angle increases.
    In image coordinates y increases downward so vertical is (0, -1).
    """
    spine    = shoulder_xy - hip_xy          #vector from hip to shoulder
    vertical = np.array([0.0, -1.0])         #upward direction in image coords
    denom    = np.linalg.norm(spine) + 1e-8
    cos_val  = np.dot(spine, vertical) / denom
    return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))

def _shoulder_elevation(shoulder_xy, hip_xy):
    """
    Normalised vertical height of shoulder relative to hip.
    Computed as (hip_y - shoulder_y) / torso_length.
    In image coordinates shoulder rising = shoulder_y decreasing
    so this value increases as the shoulder rises (shrugging).
    Normalised by torso length to remove camera distance dependency.
    """
    torso_length = np.linalg.norm(shoulder_xy - hip_xy) + 1e-8
    return float((hip_xy[1] - shoulder_xy[1]) / torso_length)

def _wrist_above_elbow(wrist_xy, elbow_xy, torso_length):
    """
    Normalised vertical offset of wrist relative to elbow.
    Positive = wrist is higher than elbow in the frame (correct position).
    Negative = wrist has dropped below elbow (compensation/fatigue).
    Normalised by torso length for camera distance independence.
    """
    #In image coords, higher = smaller y value since it follows an opposite axis
    #So wrist above elbow means wrist_y < elbow_y
    return float((elbow_xy[1] - wrist_xy[1]) / torso_length)

def calculate_angles(lm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates five angles for shoulder flexion:
    1. shoulder_angle - hip-shoulder-elbow. Primary tracking angle. Starts near 180° (arm at side), decreases as arm raises.
    2. elbow_angle - shoulder-elbow-wrist. Should remain near 180° throughout. Deviation = elbow bending compensation.
    3. trunk_angle - angle of spine vector relative to vertical. Should remain near 0° throughout. Increase = backward trunk lean compensation.
    4. shoulder_height - normalised vertical position of shoulder relative to hip. Increase during movement = shoulder shrug compensation.
    5. wrist_elbow_gap - normalised vertical offset of wrist above elbow. Positive = wrist correctly above elbow (overhead). Negative = wrist dropped below elbow (resting position - arm at the side).
    """
    rows = [] #Create an empty array that will later store the calculated angles
    for frame_idx, r in lm_df.iterrows(): #It saves the coordinates and loops through each row of data. lm_df is a pandas dataframe and iterrows() returns the index and data in a particular row.
    #frame_idx stores the index of the row and r stores the data in the row
        hip      = np.array([r["hip_x"], r["hip_y"]]) #Extract the shoulder coordinates from each row of data
        shoulder = np.array([r["shoulder_x"], r["shoulder_y"]])
        elbow    = np.array([r["elbow_x"], r["elbow_y"]])
        wrist    = np.array([r["wrist_x"], r["wrist_y"]])

        torso_length = np.linalg.norm(shoulder - hip) + 1e-8 #This is calculated so that normalization can be performed 

        rows.append({
            "frame": frame_idx, #The functions are called and the data is added to the rows
            "shoulder_angle": _angle_3pt(hip, shoulder, elbow),
            "elbow_angle": _angle_3pt(shoulder, elbow, wrist),
            "trunk_angle": _trunk_lean(shoulder, hip),
            "shoulder_height": _shoulder_elevation(shoulder, hip),
            "wrist_elbow_gap": _wrist_above_elbow(wrist, elbow, torso_length),
            "shoulder_vis": r["shoulder_vis"],
            "elbow_vis": r["elbow_vis"],
            "wrist_vis": r["wrist_vis"],
        })

    return pd.DataFrame(rows).set_index("frame") #Set the frame as the index of the row

#This function takes the raw angles that were calculated and smooths them using the savitzky golay filter
def smooth_angles(angle_df: pd.DataFrame) -> pd.DataFrame:
    out = angle_df.copy()
    for col in ["shoulder_angle", "elbow_angle", "trunk_angle",
                "shoulder_height", "wrist_elbow_gap"]:
        out[col] = savgol_filter(angle_df[col].values, SMOOTH_WINDOW, SMOOTH_POLY)
    return out


def detect_reps(shoulder_angles: np.ndarray) -> list[dict]:
    """
    Detects reps using shoulder angle.
    Arm at rest (side) = high shoulder angle (~175-180°) = PEAK
    Arm raised = low shoulder angle (~60-90°) = VALLEY
    Valley-to-valley approach: finds valleys (peak elevation moments)
    then finds the maximum angle (resting position) before and after
    each valley to define rep boundaries.
    """
    valleys, _ = find_peaks(
        -shoulder_angles,
        prominence = REP_MIN_PROMINENCE, #Rep prominence is the minimum change in angle that will be counted as a rep
        distance   = REP_MIN_DISTANCE,
    )

    if len(valleys) == 0:
        print("  WARNING: no rep valleys found — try lowering REP_MIN_PROMINENCE")
        return []

    #Peaks = Shoulder Flexion Angle = Resting Position. Here we find the maxima
    #Find the max angle before the first valley. That is designated as the initial rep boundary
    #np.argmax returns the index of the maxima. Basically we find the all the shoulder_angles before the valley point and then using np.argmax extract the maxima from those points.
    boundary_peaks = [int(np.argmax(shoulder_angles[:valleys[0]]))]

    #Find the max angle between consecutive valleys since that is marked as the latter rep boundary, that is, after the valley
    for i in range(len(valleys) - 1): 
        window = shoulder_angles[valleys[i]:valleys[i+1]] #We are comparings pairs of valleys. Between two consecutive valleys there is one peak which marks the end of rep 1 and the start of rep 2
        #We extract all the shoulder angles between the two valleys
        local_max = int(np.argmax(window)) + valleys[i] #We find the maxima from the extracted window using np.argmax. This returns the index of the maxima which is added to the valley
        #before it since the index returned is local to that window and hence it needs to be added to the valley before to get the real frame.
        boundary_peaks.append(local_max)
    #After last valley: max in [last valley → end]. It finds the boundary of the last rep by finding the max angle
    last_window = shoulder_angles[valleys[-1]:] #valleys[-1] indicates the last element of the array. We consider all frames from the last valley to the end of the video and hence extract all knee angles between those frames.
    boundary_peaks.append(int(np.argmax(last_window)) + valleys[-1]) #Similarly, we find the maxima in the window and extract its index which is added to the index of the last valley to get the real frame number
    #enumerate() works similar to iterrows() in terms that both return the index and value at that location but enumerate() works on data types like arrays, lists, tuples and returns simple values whereas, iterrows() work only on pandas data frames
    #and return the index of the row and all the data of a row, so a dictionary of data unlike a simple value in the case of enumerate().


    reps = [] #Creates an array that later stores rep data
    #Move through each rep/ valleys
    for i, valley in enumerate(valleys): #enumerate returns the index and the value at that index
        start = boundary_peaks[i] #The first peak is the start of the first rep
        end = boundary_peaks[i+1] #The end of this rep and start of the next rep.
        rep_range = shoulder_angles[start] - shoulder_angles[valley] #ROM is calculated as the difference between the angles at rest and flexion

        if rep_range < REP_MIN_PROMINENCE:
            print(f"Skipping shallow valley at frame {valley}"
                  f"(range={rep_range:.1f}° < {REP_MIN_PROMINENCE}°)")
            continue

        #Basic frame data is added 
        reps.append({
            "rep_id":       i + 1,
            "start_frame":  int(start),
            "valley_frame": int(valley),
            "end_frame":    int(end),
        })

    print(f"{len(reps)} rep(s) detected")
    return reps


def assign_states(shoulder_angles, start, valley, end):
    """
    Labels each frame in [start, end] with a state 1-4.

    S1 - Resting: arm at side, shoulder near maximum angle
    S2 - Lifting: arm rising, shoulder angle decreasing (before valley)
    S3 - Peak: arm at maximum height, shoulder angle at minimum
    S4 - Lowering: arm returning to side, shoulder angle increasing
    """
    rep_angles = shoulder_angles[start:end+1] #Shoulder angles of the rep are returned 
    rep_max = rep_angles.max() #Compute the max angle
    rep_min = rep_angles.min() #Compute the min angle - maximum flexion

    S1_EXIT = 10   #degrees below rep_max to exit S1
    S3_EXIT = 10   #degrees above rep_min to exit S3

    thresh_high  = rep_max - S1_EXIT   #S1: within 10° of resting position
    thresh_low   = rep_min + S3_EXIT   #S3: within 10° of peak elevation
    valley_local = valley - start

    n = len(rep_angles)
    states = np.zeros(n, dtype=int)

    for i, angle in enumerate(rep_angles):
        if angle >= thresh_high:
            states[i] = 1            #S1 - Resting (arm at side)
        elif angle <= thresh_low:
            states[i] = 3            #S3 - Peak elevation
        elif i <= valley_local:
            states[i] = 2            #S2 - Lifting upward
        else:
            states[i] = 4            #S4 - Lowering back down

    return states


ANGLE_COLS = ["shoulder_angle", "elbow_angle", "trunk_angle",
                "shoulder_height", "wrist_elbow_gap"] #An array is defined that stores the 5 main angles
ANGLE_LABELS = ["shoulder",       "elbow",       "trunk",
                "sho_height",     "wrist_gap"] #Stores the labels
N_STATES = 4 #States of the exercise 
VIS_THRESHOLD = 0.5

VIS_COLS = {
    "shoulder_angle": "shoulder_vis",
    "elbow_angle": "elbow_vis",
    "trunk_angle": None,  
    "shoulder_height": None,
    "wrist_elbow_gap": "wrist_vis",
} #A visibility threshold is added to ensure that only frames that have a visibility above the given thresholds for the key joints required would be considered and the frames with lower
#visibility will be rejected.

#Statistics that need to be calculated
STAT_FUNCS = {
    "mean":  np.mean,
    "min":   np.min,
    "max":   np.max,
    "range": lambda x: float(np.max(x) - np.min(x)),
    "std":   np.std,
}

def compute_rep_features(angle_df, rep, states, patient_id):
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
        "valley_frame": rep["valley_frame"],
        "rep_duration": end - start + 1,
    }

    for s in range(1, N_STATES + 1):
        mask = (states == s) #for eg: if states == 1, [1, 1, 2, 2, 2, 3, 3, 4, 4] then [True, True, False, False, False, False, False, False, False]
        state_frames = rep_angle_df[mask] #Stores the frames belonging to a particular state

        row[f"S{s}_duration"] = int(mask.sum()) #Stores the summation of those frame. Basically stores how many frames did a particular state last

        for col, label in zip(ANGLE_COLS, ANGLE_LABELS): #zip() pairs values together ("knee_angle", "knee"), ("hip_angle", "hip"), etc.
            vis_col = VIS_COLS.get(col)
            if vis_col is not None and vis_col in state_frames.columns:
                vis_mask = state_frames[vis_col] >= VIS_THRESHOLD
                gated_frames = state_frames[vis_mask]
                row[f"S{s}_{label}_vis_frames"] = int(vis_mask.sum())
            else:
                gated_frames = state_frames

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
    row["shoulder_rom"] = (
        row["S1_shoulder_mean"] - row["S3_shoulder_min"]
    )

    #How much trunk lean developed beyond the patient's own resting posture
    #A kyphotic patient starts with higher trunk_angle - expressing change from S1 means we measure ADDITIONAL lean caused by compensation, not the lean that was already there at rest
    row["trunk_compensation_s2"] = (
        row["S2_trunk_mean"] - row["S1_trunk_mean"]
    )
    row["trunk_compensation_s3"] = (
        row["S3_trunk_mean"] - row["S1_trunk_mean"]
    )

    #How much the elbow bent beyond its starting position. S1 elbow may already be slightly less than 180° due to natural arm position
    #We measure deviation from that patient's own baseline
    row["elbow_deviation_s2"] = (
        row["S1_elbow_mean"] - row["S2_elbow_min"]
    )
    row["elbow_deviation_s3"] = (
        row["S1_elbow_mean"] - row["S3_elbow_min"]
    )

    #How much the shoulder shrugged from its own resting height. shoulder_height increases as shoulder rises toward the ear
    row["shoulder_shrug_s2"] = (
        row["S2_sho_height_mean"] - row["S1_sho_height_mean"]
    )
    row["shoulder_shrug_s3"] = (
        row["S3_sho_height_mean"] - row["S1_sho_height_mean"]
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
def process_video(video_path, patient_id, side="right", output_csv=None):
    print(f"\n{'='*60}")
    print(f"Video : {os.path.basename(video_path)}")
    print(f"Patient : {patient_id} | Side: {side}")
    print(f"{'='*60}")
    #Here we call all the functions defined above 
    print("\n[1/5] Extracting landmarks")
    lm_df = extract_landmarks(video_path, side, visualize=False)

    print("[2/5] Calculating angles")
    angle_df = calculate_angles(lm_df)

    print("[3/5] Smoothing")
    smooth_df = smooth_angles(angle_df)

    print("[4/5] Detecting reps")
    shoulder = smooth_df["shoulder_angle"].values
    reps     = detect_reps(shoulder)

    if not reps:
        return pd.DataFrame()

    print("[5/5] Segmenting states & computing features")
    all_rows = []

    for rep in reps: #We loop through all the detected reps
        start = rep["start_frame"]
        end = rep["end_frame"]
        valley = rep["valley_frame"]

        states = assign_states(shoulder, start, valley, end)
        row = compute_rep_features(smooth_df, rep, states, patient_id)
        all_rows.append(row)

        s_slice = shoulder[start:end+1] #Separate the frames of a particular rep
        print(f"Rep {rep['rep_id']:>2} | frames {start}–{end} "
            f"| shoulder {s_slice.max():.1f}°→{s_slice.min():.1f}° "
            f"| "
            + "  ".join(f"S{s}={int((states==s).sum())}f" for s in range(1, 5))) #Prints how long a state lasts in terms of frames. S2 = 3f

    result_df = pd.DataFrame(all_rows) #Final data that is converted into a pandas data frame and stored in result_df

    if output_csv:
        file_exists = os.path.isfile(output_csv)
        result_df.to_csv(output_csv, mode='a', header=not file_exists, index=False)
        print(f"\n  Saved → {output_csv} "
              f"({'appended' if file_exists else 'created'})")

    print(f"\n  Done: {len(result_df)} rep(s) × {len(result_df.columns)} columns")
    return result_df


lm_df = extract_landmarks(VIDEO_PATH, SIDE, visualize=True)
angle_df = calculate_angles(lm_df)
smooth = smooth_angles(angle_df)
shoulder = smooth["shoulder_angle"].values
reps = detect_reps(shoulder)

for rep in reps:
    print(f"Rep {rep['rep_id']}: valley at frame {rep['valley_frame']}, "
          f"shoulder = {shoulder[rep['valley_frame']]:.1f}°")


def plot_rep_debug(smooth_df, reps, shoulder):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    STATE_COLORS = {1: "steelblue", 2: "orange", 3: "tomato",    4: "mediumseagreen"}
    STATE_NAMES = {1: "S1 Resting",   2: "S2 Lifting", 3: "S3 Peak",      4: "S4 Lowering"}

    fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True)
    frames = smooth_df.index.values

    for ax, col, label in zip(axes,
        ["shoulder_angle", "elbow_angle", "trunk_angle", "shoulder_height", "wrist_elbow_gap"],
        ["Shoulder (°)", "Elbow (°)", "Trunk lean (°)", "Shoulder height (norm)", "Wrist-elbow gap (norm)"]):
        ax.plot(frames, smooth_df[col].values, lw=1.5, color="navy")
        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, alpha=0.3)

    for rep in reps:
        start = rep["start_frame"]
        end = rep["end_frame"]
        valley = rep["valley_frame"]
        states = assign_states(shoulder, start, valley, end)

        for i, s in enumerate(states):
            f = start + i
            axes[0].axvspan(f, f+1, alpha=0.25, color=STATE_COLORS[s], linewidth=0)

        axes[0].axvline(valley, color="red", lw=1, ls="--", alpha=0.7)
        axes[0].text(valley, shoulder[valley] - 3, f"R{rep['rep_id']}", ha="center", fontsize=7, color="red")

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
    )

    if not df.empty:
        print("\nColumn list:")
        for c in df.columns:
            print(f"  {c}")
        print(f"\nSample row:\n{df.iloc[0].to_string()}")

import matplotlib.pyplot as plt

lm_df    = extract_landmarks(VIDEO_PATH, SIDE, visualize=False)
angle_df = calculate_angles(lm_df)
smooth   = smooth_angles(angle_df)
shoulder = smooth["shoulder_angle"].values

plt.figure(figsize=(14, 4))
plt.plot(shoulder)
plt.title("Shoulder angle over time")
plt.xlabel("Frame")
plt.ylabel("Degrees")
plt.grid(alpha=0.3)
plt.show()