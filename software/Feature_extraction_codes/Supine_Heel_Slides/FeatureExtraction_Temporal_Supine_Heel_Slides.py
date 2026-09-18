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
import math
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

VIDEO_PATH=r"D:\Sayalee\Projects\Major_project\Dataset_Videos\Supine_Heel_Slides\Gauri_SHS_L.mp4"
OUTPUT_PATH=r"D:\Sayalee\Projects\Major_project\Features\shs_per_video_temp.csv"
SIDE="right"
PATIENT_ID="Yash"
VIDEO_ID= "Yash_SHS_R"

SLIDING_WINDOW=11
POLY_ORDER=3    
#tune the order of the polynomial since less order will be too stiff and more would start following the noise than signal
REWIND_FRAMES=300        

#velocity threshold is used in order to get the phase 
#near peak extension and peak flexion, the patient holds the position so instantaneous velocities of those frames becomes nearly 0
#even though the patient is essentially stationary, fluctuations (noise) in angles- amplify noise in velocity. these fluctuations around zero are the noise floor
#thus we need to know- if the patient is stationary, how far does the calculated velocity normally wander away from zero 
#the velocity signal itself has to be used to estimate the noise floor around zero
VELOCITY_THRESHOLD=6
SEGMENTATION_THRESHOLD=0.5
MAD_MULTIPLIER=3
KERNAL_RADIUS= 2

LANDMARK_INDICES= {
  "right": dict(shoulder=12, hip=24, knee=26, ankle=28, heel=30, foot=32),
  "left":  dict(shoulder=11, hip=23, knee=25, ankle=27, heel=29, foot=31),
}

def extract_landmarks(video_path, side):
    print("DO NOT REWIND WHILE GENERATING THE DATASET")
    base_options= python.BaseOptions(
        model_asset_path= r"D:\Sayalee\Major_project\models\pose_landmarker_heavy.task"
    ) #also try with landmarker_heavy
    #BaseOptions is a configuration object that tells mediapipe how to load the model
    #It provides common model loading settings for all mediapipe tasks. All mediapipe tasks have some settings in common-
    #Like which model to be used, should it run on CPU or GPU. Instead of repeating them in every task, a common class BaseOptions is created for it
    #pose_landmarker.task contains the trained neural network for pose detection
    #this, gives the path to where the model file is located

    options= vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        #setting the running mode as Video or Livestream, automatically enables the internal smooth_landmarks filter
        #it is a post processing step applied to the raw output before it is returned. It is a filter applied after the model produces landmarks for each frame, using the landmarks from previous frame as context
        #it is a velocity scaled filter
        #first velocity is estimated based on how much the current raw estimation point moved wrt previous frame 
        #if velocity is low, assume that any frame to frame difference is noise and apply heavy smoothning, so new output is close to previous output
        #if velocity is high, point has actually moved and the difference is not noise now, but real motion, so there is light to no smoothning and raw landmarks are given as output
        num_poses=1,
        min_pose_detection_confidence=0.6,
        min_tracking_confidence=0.5,
        output_segmentation_masks=True
    )
    #an Options object (BaseOptions, PoseLandmarkerOptions) is a configurations object that stores settings
    #PoseLandmarkerOptions configures other things which are unique to PoseLandmarker
    #it can configure things like - 
    #1. running mode: what kind of input is being processed
    #2. num_poses: how many people to detect
    #3. min pose detection confidence
    #4. min pose presence confidence: how sure this pose actually exists
    #5. min tracking confidence
    #6. output segmentation masks

    cap= cv2.VideoCapture(video_path) #VideoCapture() creates an object that lets you read a video file frame by frame
    fps= cap.get(cv2.CAP_PROP_FPS) or 30

    lms_indices=LANDMARK_INDICES[side] #this saves the indices landmarks depending on which side is facing the camera
    all_frames_data=[] #this saves the whole list of landmarks that are later to be used to calculate angles. it stores the data of all frames in the video in a list
    all_frames_data_world=[] #same thing for world landmarks

    segmentation_masks=[] #list to save the segmentation masks

    #the actual pose detector
    #this is equivalent to this. but using 'with' helps to automatically clean the resource after its use is done
    #it automatically destroys it when you leave the block. 
    #with statement ensures that the allocated resources (Model memory, Buffers, Internal tracking state, GPU resources,Threads) are automatically released when block ends even if the code crashes
    #detector= vision.PoseLandmarker.create_from_options(options) ... detector.close()
    #create_from_options builds the actual pose detector with the given settings
    with vision.PoseLandmarker.create_from_options(options) as detector:
        frame_number=0 #this should be out of the loop, if not at every iteration it would reset 
        frame_idx=0
        while True:
            #STEP 1: EXTRACT EACH FRAME
            ret, frame=cap.read() #reads each frame from the video
            #cap.read() returns two values (success, image) {ret=success, frame=image}
            #if the frame is successfully captured, ret becomes true
            #if frame is no longer captured (no more frames- video ends), ret becomes false and breaks from the while loop
            #this is the only way to break out of the loop, otherwise this is an infinite loop
            if not ret:
                break
           
            #get the height and width of the frame, to actually draw the normalized coords
            height, width, _= frame.shape

            #STEP 2: EXTRACT LANDMARKS
            rgb_frame= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
            #since opencv has frames in bgr format but mediapipe needs frames in rgb, convert

            #the tasks api requires a mediapipe image. rgb_frame is a numpy array, but mediapipe tasks needs an mp.image object
            #mp.Image is a container. Instead of using a plain image object, mediapipe creates its own image class
            #mediapipe requires mp.Image() which explicitly describes the image. It wants image format information, metadata, etc. 
            mp_image= mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data= rgb_frame
            )
            #mediapipe tracks motion across frames. to track motion smoothly it needs to know which frame occured at what timestamp
            #VIDEO mode needs monotonically increasing timestamp in milliseconds
            timestamp_ms= int(frame_number*1000/fps)
            #for a livestream, to get the timestamp in ms, we can use the time library
            #with this, there are two options, time.time()- this gives the time in ms passed from jan 1 1970
            #time.perf_counter() measures elapsed time. unlike time() which may jump backward if system time resets, perf_counter() does not jump backward 
            #this tells time elapsed where python chooses some reference point internally (it is some arbitrary point chosen by the system)
            #mediapipe cares that frame1 happened before frame2, basically, the order of the frames 
            #timestamp_ms= int(time.perf_counter()*1000)
            
            result= detector.detect_for_video(
                mp_image, #image to analyze
                timestamp_ms #timestamp of when this image occurred 
            )
            #detector is the PoseLandmarker object that knows how to find poses
            #detect_for_video detects landmarks using the tracking logic. it runs inference on first frame and then tracks till tracking confidence is above the threshold
            #if tracking confidence goes below the threshold, it will again run inference on the next frame
            #detector.detect() treats each frame independently and runs the inference model on each frame (no tracking)
            #result contains the 33 landmarks

            #result.pose_landmarkers is of the type list[list[NormalizedLandmark]]. result is a PoseLandmarkerResult type object
            #[[person_0 33 landmarks in a list],[person_1 33 landmarks list]]
            #each landmark contains .x, .y, .z, .visibility, .presence (this gives the 3D coords, the visibility score- if the camera actually sees the landmark, presence score- if the model really belives that the landmark exists)

            #normalized coordinates are those that shouldnt depend on what size the image is. like the landmark should be same for 640x480px and 1280x960px
            #if not normalized, for different sizes it would give different pixel coordinates for the same physical location (knee for ex)
            #so to get normalized coords the obtained x pixel coordinate of the landmark is divided by the width of the image and y is divided by image height
            #pose_landmarks gives these normalized coordinates. but normalization only removes dependence on image resolution not on how close the person is to the camera (that is- perspective shifts)
            #the x and y coords are normalized- they are between 0 and 1. the top left corner of the image is the origin
            #if the person moves closer, then the normalized coordinates still change as the image itself changes 
            #they are percentages of image height and width
            #also the depth (z) given by this is a relative depth. it just tells which landmark is closer to the camera

            #pose_world_landmarks gives depth as an estimate in the physical coordinate system. 
            #in this method, the model is trying to estimate the 3d skeleton that could be produced by the image
            #so the coordinates given by this are not image (2d coords) but coords in a 3d coordinate system that attached to our body
            #the origin of this coordinate frame is generally centered around the hip/pelvis centre. the units are in metres. but they are not precise measurements
            #this gives the coordinates of joints as what the model thinks  
            #moving farther or closer to the camera reconstructs the entire 3d skeleton. 

            #this is a dictionary that stores everything about a frame. currently it only stores the frame number which also acts as the sr no
            #initially assume that no pose is found, so lms are not detected
            #later it is appended with the landmarks of the required joints when a pose is actually detected
            #each dictionary describes one frame. later dictionary of each frame is appended to the list "all_frame_data" in a sequential order
            #it is a temporary container to store data about one frame, which when combined with all frames would be turned into a DataFrame
            frame_data= { 
                "frame": frame_number,
                "detected":False
                #the detected flag tells if the video is usable or not. it tells in how many frames the pose was detected, as it is set to True only if pose detected
                #for frames in which pose is not detected, NaN value is stored for landmarks
                #if detection is too low, discard the video
            }

            #same thing for world landmarks
            frame_data_world= {
                "frame": frame_number,
                "detected": False
            }

            #proceed with the internal block only if a pose is detected. result.pose_landmarks is a list. in case no pose is detected, this list is empty and if the further code (trying to access the landmarks) is run on this empty list, it would crash
            if result.pose_landmarks:
                landmarks= result.pose_landmarks[0] #since only one person is to be detected, directly save whatever landmarks are in there, to a variable "landmarks"
                #if there were multiple, we wouldve done landmarks= resukt.pose_landmarks, not  with [0], [0] gives lms of person 1
                world_landmarks= result.pose_world_landmarks[0] #also save world lms

                #to save the segmentation masks for spatial check and append it to the list
                #it gives out an image the same size as the input frame
                #instead of rgb values every pixel stores the probability that it belongs to the person where 1.0 = definitely body
                #each returned mask is an mpImage object
                mask= result.segmentation_masks[0]
                segmentation_masks.append(mask)

                pixel_lms= [] #create an empty list to store the landmarks in pixel values in order to draw them on the skeleton overlay (opencv needs pixel values)
                #to get lms to draw on overlay
                for lm in landmarks: #for each landmark, convert it to pixel and append to the empty pixel_lms list                
                    x_px= int(lm.x*width)
                    y_px= int(lm.y*height)
                    pixel_lms.append((x_px,y_px)) #the pixelated landmarks are in the same order as in landmarks
                    #pixel_lms[n] contains same landmark as landmarks[n]

                #to draw the skeleton overlay    
                cv2.line(frame, pixel_lms[lms_indices["hip"]], pixel_lms[lms_indices["knee"]], (255, 0, 0), 2) #hip to knee
                cv2.line(frame, pixel_lms[lms_indices["knee"]], pixel_lms[lms_indices["ankle"]], (255, 0, 0), 2) #knee to ankle
                cv2.line(frame, pixel_lms[lms_indices["ankle"]], pixel_lms[lms_indices["foot"]], (255, 0, 0), 2) #ankle to foot
                cv2.line(frame, pixel_lms[lms_indices["shoulder"]], pixel_lms[lms_indices["hip"]], (255, 0, 0), 2) #shoulder to hip

                frame_data["detected"]=True #change to True since pose was successfully found
                frame_data_world["detected"]=True
                #store the coordinates and visibilty score of the required landmarks in frame_data
                #.items returns both key and value together. in lm_indices, for both sides, the joint name and index are stored as a dictionary 
                #so joint_name is for the key and joint_index is for the value. joint name is required to get the column name
                #every iteration gives one lm_index. since this dict contains 5 landmarks, 5 iterations are needed
                for joint_name, joint_index in lms_indices.items():
                    lm= landmarks[joint_index]
                    lmw= world_landmarks[joint_index]

                    #to view the 5 coords on the skeleton overlay   
                    x_px = int(lm.x * width)
                    y_px = int(lm.y * height)

                    #f"" stands for formatted string. whatever is inside the {}, if its a variable, its value will be subsitutes
                    #suppose in some iteration, joint_name=knee. so this would become knee_x
                    #"" gives only normal string- whatever is inside the quotes is printed/used as it is
                    frame_data[f"{joint_name}_x"]= lm.x
                    frame_data[f"{joint_name}_y"]= lm.y   
                    frame_data[f"{joint_name}_z"]= lm.z
                    frame_data[f"{joint_name}_visibility"]= lm.visibility

                    #world landmarks do not have visibility
                    frame_data_world[f"{joint_name}_x"]= lmw.x
                    frame_data_world[f"{joint_name}_y"]= lmw.y   
                    frame_data_world[f"{joint_name}_z"]= lmw.z              

                    cv2.circle(frame, (x_px, y_px), 5, (0,255,0), -1)
                    #to view the exact coordinates of the concerned joints
                    text= f"{joint_name} X:{lm.x:.2f} Y:{lm.y:.2f} Z:{lm.z:.2f} Vis:{lm.visibility:.2f}"
                    cv2.putText(frame, text, (x_px + 10, y_px), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
                    cv2.putText(frame, str(frame_idx), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,0), 2)

            #in case mediapipe fails on a certain frame, that is, pose is not detected, there would be no landmarks obtained from "result"
            #in such a case, in the columns, instead of leaving them blank which would cause inconsistency, we enter None
            else:
                for joint_name in lms_indices.keys():
                    frame_data[f"{joint_name}_x"]= None
                    frame_data[f"{joint_name}_y"]= None   
                    frame_data[f"{joint_name}_z"]= None

                    frame_data_world[f"{joint_name}_x"]= None
                    frame_data_world[f"{joint_name}_y"]= None  
                    frame_data_world[f"{joint_name}_z"]= None     
            
            #append each dictionary that describes one frame to the list that stores frame by frame landmarks of all frames
            all_frames_data.append(frame_data)
            all_frames_data_world.append(frame_data_world)

            display_frame= cv2.resize(frame,(1280,720))

            cv2.imshow('Pose_Detection', frame)
            #to listen for key inputs every 25 ms
            key= cv2.waitKey(25) & 0xFF
            #optional way to break out of loop, in order to exit before video ends 
            if key == ord('q'):
                break
            #to pause the video
            elif key == ord('p'):
                print("Video Paused. Press Any Key to Resume.")
                cv2.waitKey(0)

            #to rewind the video by 10 seconds
            elif key == ord('r'):
                #get the frame that opencv is currently on
                current_frame= cap.get(cv2.CAP_PROP_POS_FRAMES)
                #taget frame is either 0 or 300 frames before the current frame 
                target_frame= max(0, current_frame- REWIND_FRAMES)
                frame_idx=target_frame
            
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                print("Rewound by 10 seconds")

            frame_number +=1 #increment the frame number
            frame_idx +=1 #separate the frame id to be printed
            #after this the detector object is automatically destroyed it is no longer required

    cap.release()
    cv2.destroyAllWindows()

    #convert the lists into a dataframe. dataframe is kind of like a table. the list is converted into a tabular form- which is easier to handle
    df= pd.DataFrame(all_frames_data).set_index("frame")
    print(df.head()) #shows the first 5 rows
    #to fill missing values
    coord_cols=[] #create an empty list to collect the values of the detected column
    for col in df.columns: #df.columns gives the names of the columns in the dataframe
        if col != "detected" and col != "visibility":
            coord_cols.append(col) #append all the columns except "detected" and visibility
    #interpolation for frames not detected 
    df[coord_cols]=df[coord_cols].interpolate(method='linear').ffill().bfill()
    #interpolate handles the nans (gaps) in between. ffill does it for the last rows and bfill for the first few rows

    dfw= pd.DataFrame(all_frames_data_world).set_index("frame")
    print(dfw.head())
    coord_cols_w=[]
    for col in dfw.columns:
        if col != "detected":
            coord_cols_w.append(col)
    dfw[coord_cols_w]=dfw[coord_cols_w].interpolate(method='linear').ffill().bfill() 

    print("EXTRACTION COMPLETE")

    return df, dfw, fps, segmentation_masks

def get_pixel_landmarks(row, width, height):
    landmarks= {
        "shoulder": (
            int(row["shoulder_x"]*width),
            int(row["shoulder_y"]*height)
        ),
        "hip": (
            int(row["hip_x"]*width),
            int(row["hip_y"]*height)
        ),
        "knee": (
            int(row["knee_x"]*width),
            int(row["knee_y"]*height)
        ),
        "ankle": (
            int(row["ankle_x"]*width),
            int(row["ankle_y"]*height)
        ),
        "heel":(
            int(row["heel_x"]*width),
            int(row["heel_y"]*height)
        ),
        "foot": (
            int(row["foot_x"]*width),
            int(row["foot_y"]*height)
        )
    }
    return landmarks

#STEP 3: SMOOTH THE RAW LANDMARKS
#function 1 to smooth the landmarks- if the detected landmark is outside the body, checked using segmentation mask
#this is then interpolated based on- either parent reconstruction or temporal interpolation
#for parent reconstruction- if the parent landmark is trusted, recursive chain reconstruction, hermite spline with real velocity boundaries. 
#recheck the frames that are interpolated using the hermite spline interpolation
#if all fail- then flag rep/video as unreliable
#this now detects three errors- if the landmark is outside the body using segmentation mask. single spike detection using hampel filter. several frames being mistracked using 
def detect_spatial_errors(lm_df, segmentation_masks, video_path):
    clean_df= lm_df.copy()
    cap= cv2.VideoCapture(video_path)
    width= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    #create columns that would flag the frame if the particular joint shows spatial error
    clean_df["shoulder_spatial_flag"]= False
    clean_df["hip_spatial_flag"]= False
    clean_df["knee_spatial_flag"]= False
    clean_df["ankle_spatial_flag"]= False
    clean_df["foot_spatial_flag"]= False

    #create columns to store the bone lengths
    clean_df["foot_length"]= np.nan
    clean_df["tibia_length"]= np.nan
    clean_df["femur_length"]= np.nan

    #iterate through every row of the dataframe. pass 1 to just get all the bone lengths
    for frame_idx, row in clean_df.iterrows():
        
        #extract the landmark coordinates and convert each normalized landmark into pixeled and store them in a dictionary
        landmarks= get_pixel_landmarks(row, width, height)

        #compute the bone lengths by computing the eulidean distance
        foot_bone= math.dist(landmarks["foot"],landmarks["ankle"]) #connects foot and ankle
        tibia= math.dist(landmarks["ankle"],landmarks["knee"]) #connects ankle and knee
        femur= math.dist(landmarks["knee"],landmarks["hip"]) #connects knee and hip 

        clean_df.loc[frame_idx, "femur_length"]= femur
        clean_df.loc[frame_idx, "tibia_length"]= tibia
        clean_df.loc[frame_idx, "foot_length"]= foot_bone

    #set the reference values around which the tolerance band will be 
    #it is taken as the centre value because unlike mean it is resistant to outliers
    reference_foot= clean_df["foot_length"].median()
    reference_tibia= clean_df["tibia_length"].median()
    reference_femur= clean_df["femur_length"].median()
    print(f"Foot ref: {reference_foot}")
    print(f"Tibia ref: {reference_tibia}")
    print(f"Femur ref: {reference_femur}")

    #then take the absolute deviations from the median and take the median of the absolute deviation list.
    #that is the median absolute deviation (MAD). value of MAD (for femur length, as an example) tells that a typical femur length differs from the reference femur length by about 1 pixel
    #median is the expected bone length and MAD is how much mediapipe normally jitter around that expected length
    #in this way it computes deviations much larger than usual mediapipe jitter
    #first get the absolute deviations from the median
    clean_df["foot_length_deviation"]= (clean_df["foot_length"]- reference_foot).abs()
    clean_df["tibia_length_deviation"]= (clean_df["tibia_length"]- reference_tibia).abs()
    clean_df["femur_length_deviation"]= (clean_df["femur_length"]- reference_femur).abs()

    #now take the median of the deviation to compute MAD for each bone length 
    foot_mad= clean_df["foot_length_deviation"].median()
    tibia_mad= clean_df["tibia_length_deviation"].median()
    femur_mad= clean_df["femur_length_deviation"].median()
    print(f"Foot MAD: {foot_mad}")
    print(f"Tibia MAD: {tibia_mad}")
    print(f"Femur MAD: {femur_mad}")

    #compute acceptable range
    foot_min= reference_foot - (MAD_MULTIPLIER*foot_mad)
    foot_max= reference_foot + (MAD_MULTIPLIER*foot_mad)

    tibia_min= reference_tibia - (MAD_MULTIPLIER*tibia_mad)
    tibia_max= reference_tibia + (MAD_MULTIPLIER*tibia_mad)

    femur_min= reference_femur - (MAD_MULTIPLIER*femur_mad)
    femur_max= reference_femur + (MAD_MULTIPLIER*femur_mad)

    #second pass 
    for frame_idx, row in clean_df.iterrows():
        foot_bone= row["foot_length"]
        tibia= row["tibia_length"]
        femur= row["femur_length"]

        #booleans to check if the current bone length lies between the required range
        foot_ok= foot_min <= foot_bone <= foot_max
        tibia_ok= tibia_min <= tibia <= tibia_max
        femur_ok= femur_min <= femur <= femur_max

        #extract the seg mask for the current frame and convert it into a np array. get the height and width
        mask= segmentation_masks[frame_idx]
        mask_np= mask.numpy_view()

        landmarks= get_pixel_landmarks(row, width, height)
        #loop over each landmark for the particular frame
        #in a numpy array row corresponds to the vertical position (y) and column to the horizontal position (x)
        #compare every landmark to see if they are on the body or not
        #store both key and its value
        for lm_name, (x,y) in landmarks.items():

            x=np.clip(x,0,width-1)
            y=np.clip(y,0,height-1)
            #to make the kernal to check 
            y_min= max(0, y- KERNAL_RADIUS)
            y_max= max(mask_np.shape[0], y+KERNAL_RADIUS+1)
            x_min= max(0, x-KERNAL_RADIUS)
            x_max= max(mask_np.shape[1], x+KERNAL_RADIUS-1)
            #instead of checking the probability of a single pixel, take a 5x5 kernal around the pixel and take the max prob out of it. if it is greater than threshold, it is a part of the body
            patch= mask_np[y_min:y_max, x_min:x_max]
            probability= patch.max()

            #probability= mask_np[y,x] #store the probability of the pixel of predicted landmark being on body
            if probability< SEGMENTATION_THRESHOLD:
               clean_df.loc[frame_idx, f"{lm_name}_spatial_flag"]= True
         

        if (not femur_ok) and foot_ok and tibia_ok:
            clean_df.loc[frame_idx, "hip_spatial_flag"]= True

        if (not femur_ok) and (not tibia_ok) and foot_ok:
            clean_df.loc[frame_idx, "knee_spatial_flag"]= True

        if femur_ok and (not tibia_ok) and (not foot_ok):
            clean_df.loc[frame_idx, "ankle_spatial_flag"]= True

        if femur_ok and tibia_ok and (not foot_ok):
            clean_df.loc[frame_idx, "foot_spatial_flag"]= True


    return clean_df

def smooth_coords(coord_df):
    smooth_coord_df= pd.DataFrame(index=coord_df.index)
    for col in coord_df.columns:
        signal= coord_df[col].values
        smooth_coord_df[f"{col}"]= savgol_filter(
            signal,
            window_length=SLIDING_WINDOW,
            polyorder=POLY_ORDER,
            deriv=0,
            mode='interp'
        )
    return smooth_coord_df


#STEP 4: CALCULATE ANGLE AND MEASUREMENTS
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

def compute_heel_displacement(baseline,heel):
    heel= np.array(heel)
    heel_disp= baseline-heel
  
    return heel_disp
    


#create a function that calculates the angles that are required
#while passing world landmarks make, this flag true: calculate_angles(dfw, world=True) 
def calculate_measurements(lm_df : pd.DataFrame , world=False):
    all_measurements=[] #empty list to store all measurements and later convert to dataframe. it stores everything measured for all frames over the video
    #when looping over the dataframe, both can be obtained- the serial number, (which was the frame number) as well as the entire row

    #calculate the baseline for heel displacement
    baseline_heel= lm_df["heel_y"].iloc[:15].median()
    print(f"Baseline for heel: {baseline_heel}")

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
            heel= row[["heel_y"]].values
            foot= row[["foot_x", "foot_y"]].values

        frame_measurement["Knee_flexion"]= compute_angle(hip, knee, ankle)
        frame_measurement["Hip_flexion"]= compute_angle(shoulder, hip, knee)
        frame_measurement["Ankle_flexion"]= compute_angle(knee, ankle, foot)
        frame_measurement["Heel_displacement"]=  compute_heel_displacement(baseline_heel, heel)
        frame_measurement["Pelvic_lift"]= compute_pelvic_lift(hip, shoulder)
        #append the angles of one frame that is stored as dictionary, to the list that contains angles of all frames
        all_measurements.append(frame_measurement)
    
    #convert the lists to a dataframe
    measurement_df= pd.DataFrame(all_measurements).set_index("frame")
    print(measurement_df.head())
   
    return measurement_df

#STEP 5: SMOOTHEN ANGLES AND MEASUREMENTS USING SAVITZKY-GOLAY FILTER
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
    signal= smooth_df["Knee_flexion_smooth"].values #the tracking angle for supine heel slides is knee flexion. the 
    frames= smooth_df.index.values
    #find all candidate valleys (local minima)- each valley corresponds to maximum knee flexion
    
    valley_indices, _= find_peaks(-signal,
                                  prominence=0.15*(signal.max()-signal.min()), #setting an adjustable prominence
                                  distance=30) 
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
        rep_velocity= rep_data["Knee_flexion_velocity"] 

        #in order to calculate the noise floor around stationary values, first take the stationary velocity list in each rep
        #the start of the rep is peak extension around which the frames are supposed to be stationary
        #stationary_velocity= rep_velocity.loc[start:]

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
def extract_features(df, smooth_ang_df, temporal_df, rep_boundaries, phase_df, patient_id, clean_df,video_id):
    feature_df= pd.concat([smooth_ang_df, temporal_df, df[["heel_y"]], phase_df[["Phase"]]], axis=1) #axis=1 means stack column wise
    feature_df["Patient_Id"]= patient_id
    feature_df["Video_Id"]= video_id

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
def plot_graphs(raw_angle_df, smooth_angle_df, temporal_df, rep_boundaries):
    plt.figure(figsize=(12,5))  

    plt.plot(
        smooth_angle_df.index,
        smooth_angle_df["Knee_flexion_smooth"],
        label="Knee Flexion Angle"
    )
    #to visualize the rep boundaries
    for rep in rep_boundaries:
        start= rep["rep_start"]
        #to plot a vertical line where the rep starts        
        plt.axvline(x=start, color="purple", linestyle="--")    

    plt.plot(
        # raw_angle_df.index,
        # raw_angle_df["Hip_flexion"],
        temporal_df["Knee_flexion_velocity"],
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

def show_processed_overlay(video_path,lm_df,angle_df,raw_lm_df,fps,side):
    cap= cv2.VideoCapture(video_path)
    lms_indices=LANDMARK_INDICES[side]
    frame_idx=0
    delay= int(1000/fps)
    while True:
        ret, frame= cap.read()
        if not ret:
            break

        #iloc returns the row of the position frame_idx from the dataframe
        row= lm_df.iloc[frame_idx]
        raw_row= raw_lm_df.iloc[frame_idx]
        height, width, _= frame.shape
        landmarks= get_pixel_landmarks(row, width, height)
        raw_landmarks= get_pixel_landmarks(raw_row, width, height)

        cv2.line(frame, landmarks["hip"], landmarks["knee"], (255, 0, 0), 2) #hip to knee
        cv2.line(frame, landmarks["knee"], landmarks["ankle"], (255, 0, 0), 2) #knee to ankle
        cv2.line(frame, landmarks["ankle"], landmarks["foot"], (255, 0, 0), 2) #ankle to foot
        cv2.line(frame, landmarks["shoulder"], landmarks["hip"], (255, 0, 0), 2) #shoulder to hip
        #draw landmarks for every joint on the frame
        for lm_name, (x,y) in landmarks.items():
            cv2.circle(frame, (x, y), 5, (0,255,0), -1)
            cv2.putText(frame, str(lm_name), (x+10,y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
        for lm_name, (x,y) in raw_landmarks.items():
            cv2.circle(frame, (x, y), 5, (0,0,255), -1)

        display_frame= cv2.resize(frame,(1280,720))

        cv2.imshow('Pose_Detection', display_frame)
            #to listen for key inputs every 25 ms
        key= cv2.waitKey(delay) & 0xFF
        #optional way to break out of loop, in order to exit before video ends 
        if key == ord('q'):
            break
        #to pause the video
        elif key == ord('p'):
            print("Video Paused. Press Any Key to Resume.")
            cv2.waitKey(0)
        #to rewind the video by 10 seconds
        elif key == ord('r'):
            #get the frame that opencv is currently on
            current_frame= cap.get(cv2.CAP_PROP_POS_FRAMES)
            #taget frame is either 0 or 300 frames before the current frame 
            target_frame= max(0, current_frame- REWIND_FRAMES)

            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            print("Rewound by 10 seconds")

        frame_idx=frame_idx+1
            

def process_video(video_path, output_path, side, patient_id, video_id):

    #extract the landmarks into two dataframes- one with 2d normalized and another with world coords
    df,dfw,fps, segmentation_masks= extract_landmarks(video_path, side)

    #clean_df= detect_spatial_errors(df, segmentation_masks, VIDEO_PATH)
    smooth_coord_df= smooth_coords(df)


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
    feature_df= extract_features(df, smooth_df,temp_df,rep_boundaries,phase_df,patient_id, smooth_coord_df,video_id)
    print(feature_df.head())

    #show_processed_overlay(VIDEO_PATH, smooth_coord_df, smooth_df, df, fps, SIDE)
   
    #convert this dataframe into csv and save it
    save_to_csv(feature_df,output_path)

    #plot the graphs
    plot_graphs(angle_df,smooth_df,temp_df,rep_boundaries)

def main():
    process_video(VIDEO_PATH, OUTPUT_PATH,SIDE,PATIENT_ID,VIDEO_ID)
    

if __name__=="__main__":
    main()