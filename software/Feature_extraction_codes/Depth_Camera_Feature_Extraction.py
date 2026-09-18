import pyrealsense2 as rs
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

DB3_PATH= "something"
MODEL_PATH= "something"

SIDE="right"
LANDMARK_INDICES= {
  "right": dict(shoulder=12, hip=24, knee=26, ankle=28, heel=30, foot=32),
  "left":  dict(shoulder=11, hip=23, knee=25, ankle=27, heel=29, foot=31),
}
#function to setup realsense
#this starts the pipeline- configures the realsense data source so that frames can actually flow into the python program
#in this case, the data source is not a physical camera but a pre recorded db3 file
def init_realsense():
    #create pipeline
    pipeline= rs.pipeline() #rs.pipeline() creates a realsense pipeline object
    #create configuration
    config= rs.config() #pipeline manages the frame flow. config object tells the pipeline where the frames should come from and what streams are needed. this is the object used to configure the pipeline
    #specify that the source for this pipeline is a recorded db3 file and not a physical camera
    #define source
    config.enable_device_from_file(DB3_PATH)
    #enable_stream is used to get a particular stream (rgb, depth) at the resolution needed. since a prerecorded file is used here, this wont be used as the recorded file already contains the stream profiles
    #enable_stream is used in order to configure a stream
    #rs.pipeline() and pipeline.start(config) together equivalent to cap=VideoCapture()
    #to start the pipeline. until now objects are only created and configured
    #this tells the realsense system to start providing frames from the configured source 
    #activate the pipeline. use this configuration (config) and begin streaming data from that source- the db3 file 
    #activate data flow. start reading/playing the recorded data through realsense pipeline
    pipeline.start(config)
    #return the pipeline object to actually read the frames 
    return pipeline 

def get_frames(pipeline):

    #to retrieve one frame set. contains both rgb and depth frame
    #this function is equivalent to cap.read(), except this has both rgb and depth
    #this waits for the next frameset and returns a rs.composite_frame/frameset object when one becomes available- which is "frames"
    frames= pipeline.wait_for_frames()
    return frames

def extact_frames(frames):

    #extract the rgb and depth frame from the frameset
    rgb_frame= frames.get_color_frame() #realsense color frame object
    depth_frame= frames.get_depth_frame() #realsense depth frame object
    #if either frame is not available return None
    if not rgb_frame or not depth_frame:
        print("Rgb/Depth frame not available")
        return None, None
    return rgb_frame, depth_frame

#opencv directly loads the image as a numpy array into python memory, so frame is already a numpy array 
#rgb_frame and depth_frame are realsense frame objects
#these objects contain the sensor frame data and also carry realsense specific information like timestamp, frame number, stream information (info about the color, depth stream like resolution, fps, format)
def process_frames(rgb_frame, depth_frame):
    #since mediapipe requires a numpy array and these are realsense frame objects, they first need to converted to numpy arrays
    #so get_data() gives only the frame data and then asanyarray() converts it into a numpy array
    #get_data gives a buffer/pointer to underlying RGB8 pixel data
    rgb_image= np.asanyarray(rgb_frame.get_data())
    depth_image= np.asanyarray(depth_frame.get_data())
    #these are numpy arrays
    #so unlike opencv which directly gives a numpy array from cap.read(), 
    #for realsense, from .db3 file we first get a realsense frameset object, then extract a frame object from that

    #after creating a numpy array, now make the rgb_image into a mpimage 
    #the tasks api requires a mediapipe image. rgb_image is a numpy array, but mediapipe tasks needs an mp.image object
    #mp.Image is a container. Instead of using a plain image object, mediapipe creates its own image class
    #mediapipe requires mp.Image() which explicitly describes the image. It wants image format information, metadata, etc. 
    mp_image= mp.Image(
        image_format= mp.ImageFormat.SRGB,
        data= rgb_image
    )

    #now since this is a video, mediapipe needs monotonically increasing timestamp
    #this timestamp will be obtained from the rgb_frame object
    timestamp_ms= int(rgb_frame.get_timestamp())

    return mp_image, depth_image, timestamp_ms

#now create the poselandmarker 
def init_landmarker(model_path):
    base_options= python.BaseOptions(
        model_asset_path= model_path        
    )

    options= vision.PoseLandmarkerOptions(
        base_options= base_options,
        running_mode= vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.6,
        min_tracking_confidence=0.5,
        output_segmentation_masks=True
    )
    #create the detector object that would then be used to find the landmarks
    detector= vision.PoseLandmarker.create_from_options(options)

    return detector

#run the PoseLandmarker and return the result
def detect_landmarks(detector, mp_image, timestamp_ms):
    result= detector.detect_for_video(
        mp_image,
        timestamp_ms
    )
    #it returns the PoseLandmarkerResult object that contains the detected pose landmarks information
    return result

def extract_landmarks(result, side):
    #a valid output will only be detected if the PoseLandmarkerResult object return something
    #check whether this result actually exists
    if not result.pose_landmarks:
        return None

    #landmarks now contains 33 landmarks (x,y,z,visibility) where the coordinates are normalized (0-1)
    landmarks= result.pose_landmarks[0]

    lm_indices= LANDMARK_INDICES[side]
    selected_landmarks= {}

    for name,index in lm_indices.items():
        selected_landmarks[name]= landmarks[index]

    return selected_landmarks

#one important issue is that rgb image is 1280x720 wheras depth image is 848x480
#poselandmarker gives this in normalized form (0-1)
#so first convert these in pixels 
def landmarks_to_pixels(landmarks, image_width, image_height):

    if landmarks is None:
        return None

    #make an empty array to save the pixel landmarks
    pixel_landmarks= {}

    for name, landmark in landmarks.items():
        x_px= int(landmark.x*image_width)
        y_px= int(landmark.y*image_height)

        pixel_landmarks[name]= (x_px, y_px)

        return pixel_landmarks

    #z coord- getting full 3d coordinate - what is the point?
    #main continuous pipeline



