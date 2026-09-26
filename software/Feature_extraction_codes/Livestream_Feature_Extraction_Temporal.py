import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
from dataclasses import dataclass
import time
import queue

#the vision thread would contain two main components- the camera that captures and reads each frame of the livestream
#and the pose estimator that runs pose inference on each frame keeping in context that the frames are not individual pictures but part of a livestream
#both these components have their initialization function that runs once and a runtime behavior function that is called in a loop (repeatedly)    

#now, this function directly opens the webcam, but when interfacing a depth camera, this function would initialize the rgb and depth stream
#a depth camera has multiple sensors (RGB camera, Depth camera ), so when it is initialised, multiple streams are opened and have to be enabled and synchronized, and at every instance two image frames are produced by the camera (rgb frame and the depth frame)
#rgb camera and depth camera have to be synchronized so that frames from both correspond to the same instant (not 12:00:00 and 12:00:05) 
#so a depth camera interfacing manages multiple synchronized sensors rather than a single RGB stream 
#VideoCapture() is designed for one video stream. to initialize it, the manufactures sdk would be used (pyrealsense for intel). it knows how to synchronize the sensors, calibrate, align depth to rgb, configure sensor specific settings
#the initialize camera function would no longer return a single VideoCapture() object but a dataclass with its attributes being the outputs of the multiple sensors with synchronized timestampss

#this function is used to initialize and setup/configure the camera so that it is ready to stream frames. it runs once and its main job is to create a camera object 
#configure as well establish connection
def initialize_camera():
    #VideoCapture() communicates with the camera through the OS. It finds the camera, opens a connection to it, allocates internal buffers, starts preparing a video stream
    #VideoCapture() creates an object that lets you read the video frame by frame, which remembers everything that is needed to keep talking to the camera- like which file is open, which frame is currently on, how many frames are there
    #the returned VideoCapture() object maintains the connection to the camera and stores everything required to continuously acquire frames from it
    cap= cv2.VideoCapture(0)
    #check if the camera is actually opened and connected to
    if not cap.isOpened():
        raise RuntimeError("Failed to Load Camera")  

    #later can configure settings like width, height, fps, exposure, autofocus, buffer size depending on what the system needs     
    return cap

#initialize what the capture_frame function will return- a dataclass is a class whose main purpose is to store data
#it doesnt compute or control anything. it is a structured container that carries data
#it is an optimized format to transfer data 
#when depth camera is added, two frames are returned- rgb frame and depth frame
@dataclass
class CameraFrame: #the capture_frame function return these three things
    rgb_frame: np.ndarray
    depth_frame: np.ndarray | None
    timestamp_ms: int
    frame_number: int

#defining the pose estimator output. this has the raw sensor data + pose estimation output
@dataclass
class PoseFrame:
    landmarks: list
    rgb_frame: np.ndarray
    depth_frame: np.ndarray | None
    frame_number: int
    timestamp_ms: int

#the landmarks after tracking/kalman estimation to get the filtered landmarks. this has the filtered landmarks
@dataclass
class FilteredPoseFrame:
    landmarks: list
    rgb_frame: np.ndarray
    depth_frame: np.ndarray | None
    frame_number: int
    timestamp_ms: int

#this is the runtime funtion of the camera component and runs again and again while the application is operating
#a runtime function is executed repeatedly till the system is running (runs continuously)
#this function reads each frame, checks if it has been successfully read, generates a timestamp, updates the frame number and return a CameraFrame object
def capture_frame(cap : cv2.VideoCapture, frame_number : int):
    #extract and acquire every valid frame. cap.read() returns two values- (success, image) {ret=success frame=image}
    #if the latest frame is successfully captured, success becomes True. the captured image is stored as a numpy array
    #each time read() is called, the internal frame pointer moves forward
    ret, frame= cap.read()
    #when a depth camera would be used, .read() cannot be used since it is only for a VideoCapture object. first we need to wait for synchronized capture
    #unlike this, we would receive a frame set. extract rgb frame and depth frame. since the sdk's frame objects are not numpy arrays, convert the received rgb and depth frames into nparrays
    
    #in case the frame is not captured due to reasons like camera disconnection, stops responding or driver error
    if not ret:
        raise RuntimeError("Failed to capture frame")
    
    #generate timestamp for mediapipe. since mediapipe tracks motion across frames, it needs to know the order in which the frames occurred
    #thus it requires monotonically increasing timestamp in ms
    #monotonic_ns returns an integer representing the elapsed time from an arbitrary starting point in nanoseconds
    #to get in millisecs, divide by 10^6. // gives the dividend only in int (no remainder). this avoids any float point calculation that would arise in time.monotonic
    timestamp_ms= time.monotonic_ns() // 1_000_000
    #the realsense depth camera provides timestamps for the captured frames which reflect when the image was actually captured by the sensor

    #instead of making this a global variable, the vision thread would be given its ownership. it would be initialised in the vision thread
    #frame_number=frame_number+1 #increment in vision thread, outside this function
    
    return CameraFrame(
        rgb_frame=frame,
        depth_frame=None, #add when actual depth camera is interfaced
        timestamp_ms= timestamp_ms,
        frame_number=frame_number
    )

#initialization of the pose estimator
#it loads the pose model, configures the model on desired settings, creates a PoseLandmarker object and returns the detector
def initialize_pose_estimator(pose_result_callback):

    base_options= python.BaseOptions(
        model_asset_path=r"D:\Sayalee\Projects\Major_project\models\pose_landmarker_heavy.task"
    )
    #BaseOptions is a configuration object that tells mediapipe how to load the model
    #It provides common model loading settings for all mediapipe tasks. All mediapipe tasks have some settings in common-
    #Like which model to be used, should it run on CPU or GPU. Instead of repeating them in every task, a common class BaseOptions is created for it
    #pose_landmarker.task contains the trained neural network for pose detection
    #this, gives the path to where the model file is located

    #an Options object (BaseOptions, PoseLandmarkerOptions) is a configurations object that stores settings
    #PoseLandmarkerOptions configures other things which are unique to PoseLandmarker
    #it can configure things like -
    options=vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM, #running mode: what kind of input is being processed
        #setting the running mode as Video or Livestream, automatically enables the internal smooth_landmarks filter
        #it is a post processing step applied to the raw output before it is returned. It is a filter applied after the model produces landmarks for each frame, using the landmarks from previous frame as context
        #it is a velocity scaled filter
        #first velocity is estimated based on how much the current raw estimation point moved wrt previous frame 
        #if velocity is low, assume that any frame to frame difference is noise and apply heavy smoothning, so new output is close to previous output
        #if velocity is high, point has actually moved and the difference is not noise now, but real motion, so there is light to no smoothning and raw landmarks are given as output
        num_poses=1, #num_poses: how many people to detect
        min_pose_detection_confidence=0.6,
        min_pose_presence_confidence=0.6, #min confidence that the pose is actually present
        min_tracking_confidence=0.5, #how confident that tracker is that it can continue tracking the pose from previous frames without running full detection
        output_segmentation_masks=False, #with segmentation masks mediapipe returns a grayscale mask where 0=background and 255=definetly person
        #it is useful for removing, blurring and replacing background, isolating human
        result_callback=pose_result_callback #the callback function that mediapipe automatically calls to deliver its result after inference is finished
        #pose_result_callback() is wrong. () would call function immediately
    )
    #when mediapipe is used in live stream mode, it becomes asynchronous- code does not wait for pose estimation to finish
    #with a video- frame is given, mp processes it and returns landmarks. further code is blocked till inference is done
    #for a livestream- a frame is given to it, detect_async() return immediately, code continues and later when mp is done with inference, it calls the callback function
    #so instead of returning the result directly, mediapipe calls the callback function when its done
    #so by the time mediapipe infers one frame, camera doesnt stop and keeps getting next frames
    #callback function is where mediapipe delivers its inference result in a livestream. it just takes the result and stores or passes it to the next stage
    
    #it is a PoseLandmarker object that internally contains the neural network, model weights
    #detector is an object that already knows how to perform pose detection. with this, we dont need to load model, allocate tensors, initialize tracker every frame
    detector= vision.PoseLandmarker.create_from_options(options)

    return detector


#the runtime function for pose estimate component
#this function receives the raw sensor data as well as the detector object that knows how to perform pose detection
#it then makes an mpimage and gives it for pose inference
def estimate_pose(detector, camera_frame : CameraFrame):

    #since opencv reads a frame in bgr format, and mediapipe requires an rgb frame, convert
    #this wont be required when a depth camera is used
    rgb_frame= cv2.cvtColor(camera_frame.rgb_frame, cv2.COLOR_BGR2RGB)

    #the tasks api requires a mediapipe image. rgb_frame is a numpy array, but mediapipe tasks needs an mp.image object
    #mp.Image is a container. Instead of using a plain image object, mediapipe creates its own image class
    #mediapipe requires mp.Image() which explicitly describes the image. It wants image format information, metadata, etc. 
    mp_image= mp.Image(
        image_format=mp.ImageFormat.SRGB, #tells the format the current passed image is in 
        data=rgb_frame
    )

    detector.detect_async(
        mp_image,
        camera_frame.timestamp_ms
    )

#create the 2D Kalman Filter class
#a 2D kalman filter would help to estimate the true position and velocity of a moving object in a 2D space by combining noisy sensor measurements with a physical motion model
#this means that what we are tracking has a 2 dimensional position 
#Kalman filter is an object that has a state and must persist from frame to frame. a class gives a convinient place to store the persistent state
#the class defines what a kalman filter is and how it behaves, but each object created from it has its own state
#for every landmark, that is required, this filter can be created
#the kalman filter maintains an estimate of the system's state, along with uncertainty about that estimate and repeatedly updates that estimate as new measurements arrive
class KalmanFilter2D:

    def __init__ (self):

        #the state of the landmark would consist of 4 elements
        #x- horizontal position, y- vertical position
        #vx- horizontal velocity vy- vertical velocity
        #state is the kalman filter's current estimate of the thing we are tracking
        self.state= np.zeros(4, dtype=float)
        #this creates [x,y,vx,vy] as [0,0,0,0]
        #later when the first mediapipe measurement arrives, we will initialize the actual heel position

        #this is initially False and becomes True when it receives the first measurement
        self.initialized= False

        #covariance tells how uncertain the filter is about the estimate. lower the covariance, higher the confidence
        #uncertainty of the belief. represents the uncertainty associated with the state
        #the covariance matrix would be a 4x4 matrix where the diagonal elements would be the uncertainty for that state component
        #the non diagonal terms describe the relationships/correlation between the uncertainties
        #.eye creates an identity matrix and initialize the covariance matrix
        self.covariance= np.eye(4, dtype=float)

    def predict(self, dt):

        #next add the prediction model that actually estimates the next landmarks in the next frame if it knows the previous landmark position and velocity
        #initially, trying with the constant velocity model
        #for that define the state transition matrix. this matrix describes how the current state changes over a time interval dt
        #dt is the time elapsed between two frames (time between current state and next state that is being predicted)
        #since dt would change, it should not be a fixed value defined in __init__
        #but the state transition matrix would need dt when making a prediction

        #if the object needs to remember the variable after the method finishes, it should be a self. variable
        #dt is not persistent, it is an input that changes for every prediction
        #F doesnt need to be remembered, so it is made a local variable

        #the motion model is the mathematical assumption about how the thing we're tracking is moving from one time step to next
        #in this case it is a constant velocity motion model
        F= np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=float)
        # @ is numpy's matrix multiplication operator. multiply the state transition matrix with the current state to get the new estimated state  
        predicted_state= F @ self.state

        #next we handle the uncertainty of the prediction
        #covariance is how uncertain filter is about the current state but when we predict the state forward in time, it also asks how uncertain it is about the predicted state
        #so predicting the covariance is basically asking- 
        #given how uncertain i was about the previous state, and how the motion model transforms that state, how uncertain should i be about my next prediction

        #Q represents the uncertainty in motion model. it is assumed constant velocity but actual landmark can accelerate, decelerate
        Q= np.zeros((4,4),dtype=float)

        #predicted covariance
        predicted_covariance= F @ self.covariance @ F.T + Q
        

        return predicted_state, predicted_covariance

#creating the vision thread
class VisionThread:

    #this sets the starting properties(variables) of an object. it helps configure each object. it lets the user customize the object at creation - without this each object created would be identical
    #__init__ is a special method that python calls automatically when an object is created. it is a setup script that builds the objects initial data. assigns the initial state of the object
    #self is the object being initialized
    #normally all those variables that would be global, are assigned here in a Class. the vision thread class now own these variables 
    def __init__ (self):

        #the value that these functions return are going to be used in further functions. the initialize functions are called only once- here
        #initialize the camera component
        self.cap= initialize_camera()
        #initialize the pose estimator. since while configuring the pose estimator, the path to the callback function is to be given, pass that path to the initialize function- by which it PoseLandmarkerOptions can access it
        #since the callback function is a method of the vision thread, its path becomes self.pose_result_callback
        self.detector= initialize_pose_estimator(self.pose_result_callback)
        #initialize the frame number
        self.frame_number=0
        #to store the camera frames that have not been sent to mediapipe yet, pending for inference
        #this is in order to recover the things that callback function wont give
        #it is stored as a dictionary where the key is the timestamp_ms and value is CameraFrame object
        #this is just a temporary matching mechanism for matching CameraFrame and PoseFrame
        self.pending_frames= {}
        #create the tracker or filte
        #create a bounded queue that would be used to send the pose_frame object from one thread to another thread
        self.pose_queue= queue.Queue(maxsize=10)
        #to control whether the thread is running or not
        self.running= True


    #the method that will contain the continuous loop- it will run till the parameter that controls this thread becomes false
    #all the continuously run tasks would be done through this. this methods job is to run the entire vision pipeline continuously
    def run(self):
        while self.running:
            #the runtime function to acquire camera frames and store the result in a variable
            camera_frame= capture_frame(self.cap, self.frame_number)

            #store the CameraFrame until mediapipe finshes inference. the same timestamp will be returned by the callback and is unique so is used as key
            #used to recover the lost parameters that mediapipe inference wont return on callback, but are needed further in the pipeline
            #the PoseLandmarker Result would only give the landmarks, output image and timestamp_ms. other things from Camera_Frame are lost
            #camera_frame only temporarily stores this, since this is asynchronous, camera_frame may have another frame stored before mediapipe finishes the inference, so the data will be lost, thus it is stored in another dictionary but only for some time- not permanently
            #so frame is first captured and CameraFrame is stored before submitting it to mediapipe
            #to append any value in a dictionary- dict_name[key]=value
            self.pending_frames[camera_frame.timestamp_ms]= camera_frame

            #now pass the frame to the pose estimator. this function does not have or return the pose inference result- the callback function does- which mediapipe calls automatically
            estimate_pose(self.detector, camera_frame)
            #now increment the frame number
            self.frame_number += 1

    #since the callback function would require internal parameters of pending frames, it is better to make it a method in this class in order to give it direct access
    #for example it needs access to pending frames- whose owner is the vision thread
    #the callback function that is automatically called by mediapipe to store result after inference is done
    #the parameters (and their order) of the callback function are determined by mediapipe and are standard
    #result is a PoseLandmarkerResult object and contains the normalized and world landmarks. output_image is mpImage of the processed frame and timestamp_ms is the timestamp passed by user
    #since this is a method of a class, it needs self as one of the argument passed (it is a bound method, so python automatically supplied self)
    #mediapipe would not see self, python will internally handle it (python supplies it when the bound method is invoked)
    def pose_result_callback(self, result, output_image, timestamp_ms):
        #first step would be to retrieve the corresponding camera_frame using timestamp_ms as the key
        #this camera_frame (which is a CameraFrame object) was appended in the pending_frames dictionary
        #.pop retrieves the CameraFrame for that timestamp as well as removes it from pending_frames
        camera_frame= self.pending_frames.pop(timestamp_ms, None)
        #None is used as a safety fallback. If for any reason, mediapipe gives a timestamp that is not in pending frames then the function would return None instead of crashing

        #if there is no camera_frame, then return as is
        if camera_frame is None:
            return

        #next step would be to construct PoseFrame. there are two things now- result and camera_frame
        #from result we need the landmarks
        pose_frame= PoseFrame(
            landmarks= result.pose_landmarks,
            rgb_frame= camera_frame.rgb_frame,
            depth_frame= camera_frame.depth_frame,
            frame_number= camera_frame.frame_number,
            timestamp_ms= camera_frame.timestamp_ms
        )

        #next, we require to send this complete pose_frame outside the callback function
        #for now, this stays local. when the callback finishes, the FeatureExtractionThread has no way to receive it 
        #for it to be transmitted from one thread to another, it needs a thread-safe handoff mechanism- a bounded queue
        #put the pose frame in the queue to send to the next pipeline stage
        #in cases where the queue becomes full, it should discard the oldest frames
        #only put would say that "put this into queue. if queue full, wait till space is available"
        #but in this case, we cannot wait for the space to become available
        #so we do put_nowait- "put this item in queue right now. if space is not available, do not wait- raise queue.Full"
        try:
            self.pose_queue.put_nowait(pose_frame)

        except queue.Full:
            try:
                self.pose_queue.get_nowait()
                self.pose_queue.put_nowait(pose_frame)

            except queue.Empty:
                pass

vision_thread = VisionThread()
vision_thread.run()          