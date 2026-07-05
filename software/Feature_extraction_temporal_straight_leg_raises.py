#Code to extract features from the videos and save to csv

import os
import mediapipe as mp
from mediapipe.tasks import python
#tasks is designed to support multiple programming languages
from mediapipe.tasks.python import vision
#tasks are grouped by category - vision, audio, genai, text
#vision related tasks are in mediapipe.tasks.python.vision 
#this imports the package that contains all computer vision related tasks
#to import 'vision' module that are implemented in python to perform vision related tasks
#vision is a toolbox that has pose, hand and face landmarkers
#it is a module holding classes like PoseLandmarker

import cv2
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

VIDEO_PATH="D:\Sayalee\Major_project\Dataset_Videos\Patient120_slr.mp4"
OUTPUT_PATH="D:\Sayalee\Major_project\Features\straight_leg_raises.csv"
MODEL_PATH="D:\Sayalee\Major_project\models\pose_landmarker_heavy.task"
SIDE="right"

PATIENT_ID=1

SLIDING_WINDOW=11
POLY_ORDER=3    
#tune the order of the polynomial since less order will be too stiff and more would start following the noise than signal

VELOCITY_THRESHOLD=2

LANDMARK_INDICES= {
  "right": dict(shoulder=12, hip=24, knee=26, ankle=28, foot=32),
  "left":  dict(shoulder=11, hip=23, knee=25, ankle=27, foot=31),
}

def is_valid_leg(lms, lm_indices, side="right"):
    hip_x   = lms[lm_indices["hip"]].x
    hip_y   = lms[lm_indices["hip"]].y
    knee_x  = lms[lm_indices["knee"]].x
    knee_y  = lms[lm_indices["knee"]].y
    ankle_x = lms[lm_indices["ankle"]].x
    ankle_y = lms[lm_indices["ankle"]].y

    #Rule 1: This rule basically checks what is the angle of the moving leg. By default the angle of the moving leg should roughly remain unaltered (around 170-180) throughout the movement. 
    #Whereas the bent knee would be having a flexion angle of around 90-100. This is the largest differentiating factor between the resting leg and the moving leg. 
    #As a result we check whether the knee is below a certain threshold, if yes then that particular frame (where the knee coordinate jumps to the resting leg) is regarded as invalid and discarded from the rest of the pipeline to avoid errenous calculations of joint angles.
    #Knee angle is calculated using the the 3 point formula which in this case uses three coordinates - hip, knee and ankle
    ba = np.array([hip_x   - knee_x, hip_y   - knee_y])
    bc = np.array([ankle_x - knee_x, ankle_y - knee_y])
    denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8
    cos_val = np.dot(ba, bc) / denom
    knee_angle = float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))

    if knee_angle < 120: #If this condition satisfies then we return false indicating that the particular frame is invalid (invalid leg has been detected)
        return False
    #The ankle is the landmark most likely to be occluded in SLR. During the resting state the ankle lies flat on the bed and can be partially hidden by the bed surface edge, the patient's clothing, or the resting bent leg's foot. 
    #When the ankle is not clearly visible, MediaPipe guesses its position and a guessed ankle position produces an unreliable knee angle and hip angle calculation.
    #This rule just checks the visibility of the ankle since it plays a vital role in calculation of the knee and hip angle (if visibility is less due to occlusion the coordinate is an estimate and thereby might result in errenous calculations of joint angles)
    if lms[lm_indices["ankle"]].visibility < 0.3:
        return False
    #if any of the following rules are not satisfied the frame is marked as invalid and discarded
    return True

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
    lm_indices = LANDMARK_INDICES[side] #Indces for a particular side as defined above are extracted and saved in this variable
    records    = [] #Created to store the pose coordinates

    #The pose estimation model is loaded
    #An object basically contains information (data/attributes) and methods or functions that are used to operate on that data

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

    POSE_CONNECTIONS = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 23), (12, 24), (23, 24),
        (23, 25), (25, 27), (27, 29), (27, 31),
        (24, 26), (26, 28), (28, 30), (28, 32),
        (29, 31), (30, 32),
    ] #This is a dictionary that basically contains pairs of connections between landmarks

    cap = cv2.VideoCapture(video_path) #VideoCapture() creates an object that lets you read a video file frame by frame
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    #FPS is required since the video mode of mediapipe tasks api requires a time stamp which is achieved by converting frame number into time
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0   #fallback if fps not in metadata
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    invalid_count = 0
    paused = False

    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    YELLOW = (0, 255, 255)
    CYAN = (255, 255, 0)
    WHITE = (255, 255, 255)

    with vision.PoseLandmarker.create_from_options(options) as detector: #Here we actually load the pose detector model that was downloaded
        frame_idx = 0 #Initial index
        while True: #Loop runs until the video ends
            if not paused: 
                ret, frame = cap.read() #Here we read each frame one by one. It returns two things - ret (boolean value) it signifies whether the frame was read successfully or not (true or false) and frame, that is, the actual image - a dictionary that stores all the pixels as RGB values.
                if not ret: #If the frame is not available - it indicates the end of the video and hence we break out of the loop
                    break

                #OpenCV stores images in BGR format but mediapipe tasks requires the normal RGB format. Hence conversion is important
                rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb) #Mediapipe Tasks API cannot directly use NumPy/OpenCV arrays. It expects its own image object, so we convert rgb to image object. Frame is not an object - it is a numpy array which stores the pixel values of an image.
                """Mediapipe tasks api does not directly accept a frame and hence conversion is done to an image object. mp.Image object stores: Data - pixel data, image width, image height, image format (RGB). Methods: image_format(), numpy_view()"""
                #VIDEO mode needs a monotonically-increasing timestamp in ms. Time = frame number/fps and since the time is required in milliseconds - it is multiplied by 1000. Time stamps are important for tracking between frames, without time stamps tracking is impossible.
                timestamp_ms = int(frame_idx * 1000 / fps)
                result = detector.detect_for_video(mp_image, timestamp_ms) #Here, we actually input the image and the time stamp into the mediapipe pose detection model. Internally mediapipe detects person, estimates pose landmarks and tracks coordinates using the previous frame.

                row = {"frame": frame_idx, "detected": False} #Stores information about the current frame

                #result.pose_landmarks is a list of poses; [0] = first (only) person. lm is a mediapipe landmark object that contains: x coordinate, y coordinate, z coordinate, visibility and presence.
                """lm_indices = { "shoulder":11, "hip":23, "knee":25, "ankle":27, "foot":3 } .items() gives: ("shoulder",11), ("hip",23), ("knee",25), ("ankle",27), ("foot",31) """
                if result.pose_landmarks and len(result.pose_landmarks) > 0:
                    lms = result.pose_landmarks[0]          #list of NormalizedLandmark

                    valid = is_valid_leg(lms, lm_indices, side)

                    if visualize:
                        vis_frame = frame.copy()
                        
                        #POSE_CONNECTIONS is a predefined set of landmark pairs provided by MediaPipe. Each pair specifies which two body landmarks should be connected by a line to form the skeleton.
                        for s_idx, e_idx in POSE_CONNECTIONS: #Here s_idx and e_idx represents the start and end index between which the connections need to be made respectively. 
                            if (s_idx < len(lms) and e_idx < len(lms) and  #This condition checks whether the landmarks are valid or not
                                lms[s_idx].visibility > 0.3 and #Visibility of the start and end index is checked
                                lms[e_idx].visibility > 0.3):
                                pixel_of_s = (int(lms[s_idx].x * width), #Here we calculate the pixel coordinates of the starting landmark so that accordingly the coordinate can be displayed on the screen.
                                              #x coordinate is multiplied with the width of the frame and y coordinate is multiplied with the height of the frame
                                      int(lms[s_idx].y * height))
                                pixel_of_e = (int(lms[e_idx].x * width),
                                      int(lms[e_idx].y * height))
                                cv2.line(vis_frame, pixel_of_s, pixel_of_e, (150, 150, 150), 1) #Here we draw a line between the starting landmark and the ending landmark 
                                #cv2.line(image, start_point, end_point, color, thickness)
                        joint_labels = {
                            "shoulder": "SHO",
                            "hip":      "HIP",
                            "knee":     "KNEE",
                            "ankle":    "ANK",
                            "foot":     "FOOT",
                        }
                        #lm_indices is a dictionary that stores the mapping between the joint names and their corresponding MediaPipe landmark indices.
                        #.items() returns the joint and index of the landmark. Joint stores the joint name and idx stores the index of the mediapipe landmark
                        for joint, idx in lm_indices.items():
                            lm = lms[idx] #The landmark object - lm, contains several attributes like .x, .y, .visibility, etc.
                            px = int(lm.x * width) #Convert the x coordinate into pixel coordinates by mutliplying it with the width of the frame
                            py = int(lm.y * height) #Convert the y coordinate into pixel coordinates by multiplying it with the height of the frame
                            colour = GREEN if valid else RED
                            cv2.circle(vis_frame, (px, py), 8, colour, -1) #Show a circle on the px and py coordinates 
                            cv2.putText(vis_frame, joint_labels[joint], #Display the joint name on screen close to the pixel coordinates of the landmark for a particular joint
                                        (px + 10, py - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, colour, 2)
                            
                        if valid: #If the validity condition is passed the following text will be displayed on the screen. 
                            cv2.putText(vis_frame, "VALID LEG",
                                        (20, 40),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        GREEN, 2)
                        else: #If the above condition is not satisfied then the following text is displayed on the screen 
                            cv2.putText(vis_frame, "REJECTED — BENT LEG",
                                        (20, 40),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        RED, 2)
                        hip_lm  = lms[lm_indices["hip"]] #lms is the landmark object that has attributes like .x, .y and .visibility. lm_indices["hip"] gives us the landmark of the hip, which might be 23. We could have written lms[23] 
                        #directly but writing it this way makes it more readable. hip_lm contains all the information required (such x, y and visibility data) about the hip landmark of the detected side 
                        knee_lm = lms[lm_indices["knee"]]
                        ank_lm  = lms[lm_indices["ankle"]]
                        sho_lm  = lms[lm_indices["shoulder"]]

                        hip_arr  = np.array([hip_lm.x,  hip_lm.y]) #We then create a np array of the x and y coordinates of the landmarks so that mathematical calculations like angle calculations can be performed. 
                        knee_arr = np.array([knee_lm.x, knee_lm.y])
                        ank_arr  = np.array([ank_lm.x,  ank_lm.y])
                        sho_arr  = np.array([sho_lm.x,  sho_lm.y])

                        def cal_angle(a, b, c): #Here we calculate the angles to display on the screen  
                            ba = a - b; bc = c - b
                            d  = np.linalg.norm(ba)*np.linalg.norm(bc)+1e-8
                            return float(np.degrees(
                                np.arccos(np.clip(np.dot(ba,bc)/d,-1,1))))

                        hip_angle  = cal_angle(sho_arr, hip_arr, knee_arr)
                        knee_angle = cal_angle(hip_arr, knee_arr, ank_arr)

                        #Display the calculated knee and hip angles 
                        cv2.putText(vis_frame,
                            f"Hip: {hip_angle:.1f}  Knee: {knee_angle:.1f}",
                            (20, height - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2)
                        #Display the visibility of key angles 
                        cv2.putText(vis_frame,
                            f"Vis — Hip:{hip_lm.visibility:.2f}  "
                            f"Knee:{knee_lm.visibility:.2f}  "
                            f"Ankle:{ank_lm.visibility:.2f}",
                            (20, height - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (200, 200, 200), 1)

                        #Here we display the frame number and also the number of frames that have been rejected 
                        cv2.putText(vis_frame,
                            f"Frame: {frame_idx}  "
                            f"Invalid: {invalid_count}",
                            (width - 300, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (200, 200, 200), 2)
                        
                        if paused:
                            cv2.putText(vis_frame, "PAUSED",
                                        (width//2 - 60, 40),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        WHITE, 2)

                        cv2.imshow("Extraction Visualiser", vis_frame)

                    if valid: 
                        for joint, idx in lm_indices.items():
                            #lm_indices is a dictionary that stores the mapping between the joint names and their corresponding MediaPipe landmark indices.
                            #.items() returns the joint and index of the landmark. Joint stores the joint name and idx stores the index of the mediapipe landmark
                            lm=lms[idx] #Each landmark object - lm, contains several attributes such as .x, .y, .visibility, etc.
                            row[f"{joint}_x"] = lm.x
                            row[f"{joint}_y"] = lm.y
                            row[f"{joint}_vis"] = lm.visibility
                        row["detected"] = True

                    else:
                        for joint in lm_indices:
                            row[f"{joint}_x"] = np.nan
                            row[f"{joint}_y"] = np.nan
                            row[f"{joint}_vis"] = 0.0
                        invalid_count +=1 
                else: #This condition runs when no human is detected in the frame and all the landmarks are 0
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
                    print("  Visualiser quit early by user")
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

    #Fill short detection gaps (≤ 5 consecutive frames). Here we select columns ending in _x and _y and fill small gaps by either forward filling or backward filling.
    #In forward filling the values are filled with a known value before it. This is done for a maximum of 5 consecutive frames.
    #In backward filling the values are filled with a known value after it. Forward fill handles missing middle values and backward fill handles missing beginning values.
    coord_cols = [c for c in df.columns if c.endswith("_x") or c.endswith("_y")]
    df[coord_cols] = df[coord_cols].ffill(limit=5).bfill(limit=5)

    detection_rate = df["detected"].mean() * 100 #The detected column of the data frame contains T or F. T = 1, F = 0. Example: [1, 1, 0, 0, 1], we then take the mean of these values and multiply that by 100 to get the detection rate.
    valid_rate = 100 - (invalid_count/frame_idx*100)
    print(f"  {frame_idx} frames | detection rate: {detection_rate:.1f}%")
    print(f"  Spatial validation: {invalid_count} frames rejected "
          f"({100-valid_rate:.1f}% of detected frames)")
    if detection_rate < 70:
        print("  WARNING: low detection rate — check video angle / lighting")

    return df, fps

#STEP 3: CALCULATE ANGLE AND MEASUREMENTS
#to calculate angle at a point B, we need 3 points- A, B, C. the angle between vectors BA and BC is the angle at B
#theta= cos_inv(BA.BC/|BA||BC|) cos theta is dot product of the two vectors divided by the product of their magnitudes
# a b c are three anatomical points forming an angle, which change as per the angle required. a is point above joint, b is joint vertex and c is point below vertex
#we measure how much the vector BA rotates relative to BC
def compute_angle(a,b,c):
    a= np.array(a) #np.array converts the input into ndarray object and returns them
    b= np.array(b) 
    c= np.array(c)
    #in the dataframe, each point (joint) is saved as a x,y,z which are in float and should be converted to ndarray object before they are used to form vectors and compute angles 
    #python objects (tuple, list, dict) are bad at vector math
    #make vectors
    ba= a-b
    bc= c-b
    cosine= (np.dot(ba,bc))/(np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-6) #+1e-6 is added for stability to prevent edge case failures in floating point computatioj
    cosine= np.clip(cosine, -1.0, 1.0) #limit the value between -1 and 1 becuase sometimes float computations may give 1.00000003 and arccos strictly needs value between -1 and 1
    angle= np.arccos(cosine) #this is given in radians so it should be first converted to degrees 
    return np.degrees(angle) 

def compute_pelvic_lift(hip, shoulder):
    hip= np.array(hip)
    shoulder= np.array(shoulder)

    gap= hip[1]-shoulder[1]
    torso_length= np.linalg.norm(hip-shoulder) + 1e-8
    return(gap/torso_length)

#create a function that calculates the angles that are required
#while passing world landmarks make, this flag true: calculate_angles(dfw, world=True) 
def calculate_measurements(lm_df : pd.DataFrame , world=False):
    all_measurements=[] #empty list to store all measurements and later convert to dataframe. it stores everything measured for all frames over the video
    #when looping over the dataframe, both can be obtained- the serial number, (which was the frame number) as well as the entire row
    for frame_idx, row in lm_df.iterrows():
        frame_measurement={
            "frame": frame_idx
        }#dictionary to store data of one frame, which is later appended to the list of all frames. this stores everything measured for one frame
       
        #"row" represents each row in the dataframe, each column can be extracted by row["column_name"]
        #since to represent one joint, its x,y,z coords would be required
        if world:
            #for world landmarks, 3d angle is calculated so z coordinate is also required
            shoulder= row[["shoulder_x","shoulder_y", "shoulder_z"]].values
            hip= row[["hip_x", "hip_y", "hip_z"]].values
            knee= row[["knee_x", "knee_y", "knee_z"]].values
            ankle= row[["ankle_x", "ankle_y", "ankle_z"]].values
            foot= row[["foot_x", "foot_y", "foot_z"]].values

        else:
            shoulder= row[["shoulder_x","shoulder_y"]].values
            hip= row[["hip_x", "hip_y"]].values
            knee= row[["knee_x", "knee_y"]].values
            ankle= row[["ankle_x", "ankle_y"]].values
            foot= row[["foot_x", "foot_y"]].values

        frame_measurement["Knee_flexion"]= compute_angle(hip, knee, ankle)
        frame_measurement["Hip_flexion"]= compute_angle(shoulder, hip, knee)
        frame_measurement["Ankle_flexion"]= compute_angle(knee, ankle, foot)
        frame_measurement["Pelvic_lift"]= compute_pelvic_lift(hip, shoulder)
        #append the angles of one frame that is stored as dictionary, to the list that contains angles of all frames
        all_measurements.append(frame_measurement)
    
    #convert the lists to a dataframe
    measurement_df= pd.DataFrame(all_measurements).set_index("frame")
    print(measurement_df.head())
   
    return measurement_df

#STEP 4: SMOOTHEN ANGLES AND MEASUREMENTS USING SAVITZKY-GOLAY FILTER
def smooth_measurments(angle_df : pd.DataFrame):
    smooth_df= pd.DataFrame(index=angle_df.index) #create an empty dataframe to save the smoothed values
    for col in angle_df.columns:
        signal= angle_df[col].values #.values strips all the pandas metadata and gives the numbers as numpy array
        smooth_df[f"{col}_smooth"]= savgol_filter(
            signal, 
            window_length= SLIDING_WINDOW,
            polyorder= POLY_ORDER,
            deriv=0,
            mode='interp' #this is for edge handling. it uses polynomial extrapolation at the edges of the signal 
        )
    return smooth_df #return smoothed angles

#STEP 5: CALCULATE THE TEMPORAL FEATURES OF EACH ANGLE
#this function computes the velocitu, accln and jerk
#it receives the dataframe of smoothed angles as derivates of the clean signal are required 
def compute_temporal_features(smooth_df : pd.DataFrame, fps):
    dt=1/fps #dt is the time between frames in seconds
    #for deriv=1, SG computes the rate of change per sample that is, per frame. but velocity is needed rate of change of angle per second
    #vel= deg per frame/dt = deg per sec. if this is not done, velocity would change depending on fps of the video. deg/sec is more consistent

    temporal_df= pd.DataFrame(index=smooth_df.index) #create an empty dataframe to store the temporal features
    for col in smooth_df.columns:
        base_name= col.replace("_smooth", "") #remove the suffix smooth from each column name to later add vel, accln and jerk
        signal= smooth_df[col].values #get angles and numpy array without the metadata

        temporal_df[f"{base_name}_velocity"]= savgol_filter(
            signal,
            window_length= SLIDING_WINDOW,
            polyorder= POLY_ORDER,
            deriv=1,
            delta=dt,
            mode='interp'
        )

        temporal_df[f"{base_name}_accln"]= savgol_filter(
            signal,
            window_length= SLIDING_WINDOW,
            polyorder= POLY_ORDER,
            deriv=2,
            delta=dt,
            mode='interp'
        )

        temporal_df[f"{base_name}_jerk"]= savgol_filter(
            signal,
            window_length= SLIDING_WINDOW,
            polyorder=4,
            deriv=3,
            delta=dt,
            mode='interp'
        )
    return temporal_df #return the temporal features

#STEP 6: REP DETECTION
#detect rep boundaries and return a list of dictonaries that describe each rep
def detect_rep(smooth_df):
    signal= smooth_df["Hip_flexion_smooth"].values #the tracking angle for straight leg raises is hip flexion angle
    frames= smooth_df.index.values
    #find all candidate valleys (local minima)- each valley corresponds to maximum knee flexion
    
    valley_indices, _= find_peaks(-signal,
                                  prominence=0.25*(signal.max()-signal.min()), #setting an adjustable prominence
                                  distance=50) 
    #negating the original signal to find valleys. the highest point in -signal would correspond to the lowest point in the original signal (valley)
    #the find peaks function gives two values- first is index of the peak. second- properties is a dictionary containing extra information (prominence, width, height)
    #find_peaks defines a peak at a point that is higher than its immediate neighbours
    #prominence is a measure of how much does the peak stand out from the surrounding signal. measures the vertical distance between peak and lowest contour line that connects it to a higher peak (how prominent is that peak).
    #how much peak rises in the surrounding valley. higher prominence means tiny fluctuations wont be considered as peaks. it means how deep the valley is compared to its surrounding hills
    #distance is after how many samples would next peak be allowed. in case samples fluctuate a lot, this would not flag it as a peak again and again. so after how many minimum frames (samples) would a true peak be detected

    if len(valley_indices)==0:
        print("No reps detected")
        return [] #if no reps are detected return empty array
    
    rep_boundaries=[] #empty list for reps 
    #enumerate iterates over both- the index as well as the value at the same time
    #this defines where exactly to search for the peaks 
    for i, valley in enumerate(valley_indices):

        #define the search boundaries for the left and right maxima
        if i==0: #in the first iteration, left start is 0
            left_start=0
        else:
            left_start=valley_indices[i-1] #from the next iteration it is the previous valley
        #left maxima is searched for between left start and end, and right maxima- right start and end
        #while evaluating for 118 in [50 118 201 286], ls-50 le-118 rs-18 re-201. left maxima searched bw 50 and 118 frames and right maxima between 118 and 201 frames
        left_end= valley
        right_start= valley 
        
        if i== len(valley_indices)-1: #in the last iteration the right end is the end of the signal
            right_end= len(signal)
        else:
            right_end= valley_indices[i+1] #otherwise it is the next valley
        
        #indices of the left and right peak around a valley
        left_max_idx= left_start+ np.argmax(signal[left_start:left_end])
        right_max_idx= right_start+ np.argmax(signal[right_start:right_end])

        rep_boundaries.append({
            "rep_id": i+1,
            "rep_start": frames[left_max_idx],
            "rep_end": frames[right_max_idx],
            "valley_frame": frames[valley]
        })
    print(f"{len(rep_boundaries)} reps detected")
    return rep_boundaries

#STEP 7: PHASE DETECTION
#detect four phases: P1- max extension. P2- descending angle towards flexion. P3- max flexion. P4- ascending angle towards extension
#for this pass a dataframe that contains the smoothed tracking angle, and velocity of it, with frame as index
#also requires the rep boundaries
def detect_phase(feature_df, rep_boundaries):
    phase_df= feature_df.copy()
    phase_df["Phase"]= None #create a new column in the phase dataframe and initialize all its rows to none

    #iterate over each rep and extract the start end and valley frame of each rep
    for rep in rep_boundaries:
        start= rep["rep_start"]
        end= rep["rep_end"]
        valley= rep["valley_frame"]
    
        #.loc is used to select rows and columns by their labels. this is selecting all rows from "start" to "end" of the column "Knee_flexion_velocity" 
        #extract only the frames belonging to that rep 
        rep_data= feature_df.loc[start:end]
        #extract the knee flexion velocities of the frames belonging to the rep
        rep_velocity= rep_data["Hip_flexion_velocity"] 
        


        #p1_end is the last frame in phase 1, which is full extension. start is around peak extension
        #later last frame is shifted to the last stationary frame
        p1_end= start
        #include all near stationary frames in phase 1 and break phase 1 only when frame velocity goes beyond threshold
        for frame, velocity in rep_velocity.items():
            if abs(velocity)< VELOCITY_THRESHOLD:
                p1_end= frame
            else:
                break
    
        #to define phase 3 frames. both p3_start and p3_end are initialised to be the valley frame. the valley frame lies inside the hold phase
        #p3_start frame is shifted to the first frame to where the velocity of frame is less than threshold to find where the hold began
        #then the p3_end frame is shifted to the last frame when the velocity of that frame is still less than threshold velocity to find where the hold ends
        #the moment velocity of the frame exceeds threshold, it breaks out of phase 3
        p3_start= valley
        p3_end= valley

        #shift p3_start
        #.items() returns an iterator. iterator is an object that just gives the next item when asked for it- doesnt build the whole list
        #for reversed, the whole list is needed
        #select only the portion upto the valley and then reverse it
        for frame, velocity in reversed(list(rep_velocity.loc[start:valley].items())):
            if abs(velocity)< VELOCITY_THRESHOLD:
                p3_start=frame
            else:
                break #only include near stationary frames and break when velocity exceeds threshold
        
        #shift p3_end
        for frame, velocity in rep_velocity.loc[valley:end].items():
            if abs(velocity)<VELOCITY_THRESHOLD:
                p3_end= frame
            else:
                break 
        
        #phase boundaries are defined. now define the phases
        phase_df.loc[start:p1_end, "Phase"]= "P1" #full extension and hold
        phase_df.loc[p1_end+1 : p3_start-1, "Phase"]= "P2" #descending from extension to flexion
        phase_df.loc[p3_start:p3_end, "Phase"]= "P3" #full flexion and hold
        phase_df.loc[p3_end+1: end-1, "Phase"]= "P4" #ascending from flexion to extension
    
    return phase_df

#STEP 8: EXTRACT FEATURES INTO SINGLE DATAFRAME
def extract_features(smooth_ang_df, temporal_df, rep_boundaries, phase_df, patient_id):
    feature_df= pd.concat([smooth_ang_df, temporal_df, phase_df[["Phase"]]], axis=1) #axis=1 means stack column wise
    feature_df["Patient_Id"]= patient_id

    feature_df["Rep_Start"]= 0
    feature_df["Rep_Id"]= 0

    for rep in rep_boundaries:
        start= rep["rep_start"]
        end= rep["rep_end"]
        rep_id= rep["rep_id"]
        feature_df.loc[start:end-1, "Rep_Id"]= rep_id
        feature_df.loc[start, "Rep_Start"]=1

    return feature_df

#STEP 9: SAVE TO CSV
def save_to_csv(feature_df, output_path):
    if os.path.exists(output_path):
        feature_df.to_csv(
            output_path, 
            mode='a', #append
            header=True, #dont write column names again 
            index=True #save dataframe index
        )
    else:
        feature_df.to_csv(
            output_path,
            mode='w', #write
            header=True,
            index=True
        )

#STEP 10: PLOT GRAPHS 
#add all graphs
def plot_graphs(raw_angle_df, smooth_angle_df, temporal_df):
    plt.figure(figsize=(12,5))

    plt.plot(
        smooth_angle_df.index,
        smooth_angle_df["Hip_flexion_smooth"],
        label="Knee Flexion Angle"
    )
    

    plt.plot(
        # raw_angle_df.index,
        # raw_angle_df["Hip_flexion"],
        temporal_df["Hip_flexion_velocity"],
        label="Knee Flexion Velocity",
        #alpha=0.8 #controls transparency. this signal is drawn lighter
    )

    # plt.plot(
    #     # smooth_angle_df.index,
    #     # smooth_angle_df["Hip_flexion_smooth"],
    #     temporal_df["Knee_flexion_accln"],
    #     label="Knee Flexion Acceleration"
    # )

    plt.xlabel("Frame")
    plt.ylabel("Knee Flexion (degrees)")
    plt.title("Knee Flexion vs Frame")
    plt.grid(True)
    plt.legend()

    plt.show()



def process_video(video_path, output_path, side, patient_id):

    #extract the landmarks into two dataframes- one with 2d normalized and another with world coords
    df,fps= extract_landmarks(video_path, side, visualize=True)

    #compute the angles and measurements with the extracted landmarks
    angle_df= calculate_measurements(df)

    #smooth the measurements using SG filter
    smooth_df= smooth_measurments(angle_df)

    #calculate the velocity, accln, and jerk
    temp_df= compute_temporal_features(smooth_df, fps)
    print(temp_df.head())

    #calculate the rep boundaries (reps)
    rep_boundaries= detect_rep(smooth_df)
    print(rep_boundaries)

    #make a combines dataframe for what phase detection function needs (instead of passing two dataframes)
    #and calculate phase boundaries
    fdf= pd.concat([smooth_df, temp_df], axis=1)
    phase_df= detect_phase(fdf,rep_boundaries)

    #extract all features into one common dataframe
    feature_df= extract_features(smooth_df,temp_df,rep_boundaries,phase_df,patient_id)
    print(feature_df.head())
   
    #convert this dataframe into csv and save it
    save_to_csv(feature_df,output_path)

    #plot the graphs
    plot_graphs(angle_df,smooth_df,temp_df)

def main():
    process_video(VIDEO_PATH, OUTPUT_PATH,SIDE,PATIENT_ID)
    

if __name__=="__main__":
    main()