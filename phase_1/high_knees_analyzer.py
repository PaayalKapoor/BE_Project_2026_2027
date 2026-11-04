import cv2
import mediapipe as mp
import numpy as np

window_name="High Knees"
summary_window_name="Summary"
stop_flag=False
stop_button_rect= [0, 0, 0, 0]
summary_close_flag= False
summary_close_rect= (200, 130, 310, 175)


def get_stop_button_rect(frame):
    h, w= frame.shape[:2]
    x2, y2= w - 15, h - 15
    return [x2 - 90, y2 - 35, x2, y2]


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
    for cx, cy in [(x1+radius, y1+radius), (x2-radius, y1+radius), (x1+radius, y2-radius), (x2-radius, y2-radius)]: cv2.circle(image, (cx, cy), radius, color, -1, cv2.LINE_AA)


def draw_stop_button(frame):
    x1, y1, x2, y2 = stop_button_rect
    rounded_rect(frame, (x1, y1, x2, y2), (0, 0, 255), radius=10)
    text = "STOP"
    tw, th = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    cv2.putText(frame, text, (x1 + (x2-x1-tw)//2, y1 + (y2-y1+th)//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def make_summary_image(reps):
    summary = np.zeros((220, 340, 3), np.uint8)
    summary[:] = (40, 40, 40)
    cv2.putText(summary, "SUMMARY", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(summary, f"Total Reps: {reps}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    rounded_rect(summary, summary_close_rect, (0, 0, 255), radius=12)
    cv2.putText(summary, "Close", (summary_close_rect[0]+25, summary_close_rect[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return summary


def show_summary(reps):
    global summary_close_flag
    summary_close_flag= False
    cv2.namedWindow(summary_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(summary_window_name, 340, 220)
    cv2.setMouseCallback(summary_window_name, on_summary_mouse, summary_close_rect)
    while True:
        cv2.imshow(summary_window_name, make_summary_image(reps))
        key = cv2.waitKey(20) & 0xFF
        if key == 27 or summary_close_flag:
            break
    cv2.destroyWindow(summary_window_name)


def calculate_angle(a, b, c):
    a, b, c= np.array(a), np.array(b), np.array(c)
    radians= np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle= np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360- angle
    return angle

def angle_from_vertical(top, bottom):
    top, bottom = np.array(top), np.array(bottom)
    vec = bottom - top                          
    vertical = np.array([0.0, 1.0])           
    cos_a = np.dot(vec, vertical) / (np.linalg.norm(vec) + 1e-6)
    cos_a = np.clip(cos_a, -1.0, 1.0)
    return np.degrees(np.arccos(cos_a))


mp_pose= mp.solutions.pose
mp_drawing= mp.solutions.drawing_utils

KNEE_UP_ANGLE= 100
KNEE_DOWN_ANGLE= 140
KNEE_LIFT_RATIO= 0.06
KNEE_BEND_MIN= 70    
TORSO_LEAN_MAX= 20   
THIGH_LIFT_MIN= 40   

counter= 0
left_knee_up= False
right_knee_up= False
left_angle= 0.0
right_angle= 0.0
left_knee_bend= 0.0
right_knee_bend= 0.0
torso_angle= 0.0
left_thigh_angle= 0.0
right_thigh_angle= 0.0

cap= cv2.VideoCapture(0)

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 854, 480)
cv2.setMouseCallback(window_name, on_mouse, stop_button_rect)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        stop_button_rect[:] = get_stop_button_rect(frame)

        image= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable= False
        results= pose.process(image)
        image.flags.writeable= True
        image= cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        is_side_view= False
        feedback= ""

        try:
            landmarks= results.pose_landmarks.landmark

            ls= landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            rs= landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            shoulder_width= abs(ls.x - rs.x)
            is_side_view= shoulder_width < 0.18

            l_shoulder= [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,  landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
            l_hip= [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
            l_knee= [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]
            l_ankle= [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]   # NEW

            r_shoulder= [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
            r_hip= [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]
            r_knee= [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y]
            r_ankle= [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y]  # NEW

            left_angle= calculate_angle(l_shoulder, l_hip, l_knee)
            right_angle= calculate_angle(r_shoulder, r_hip, r_knee)

            left_knee_bend= calculate_angle(l_hip, l_knee, l_ankle)
            right_knee_bend= calculate_angle(r_hip, r_knee, r_ankle)

            mid_shoulder= [(l_shoulder[0]+r_shoulder[0])/2, (l_shoulder[1]+r_shoulder[1])/2]
            mid_hip= [(l_hip[0]+r_hip[0])/2, (l_hip[1]+r_hip[1])/2]
            torso_angle= angle_from_vertical(mid_shoulder, mid_hip)

            left_thigh_angle= angle_from_vertical(l_hip, l_knee)
            right_thigh_angle= angle_from_vertical(r_hip, r_knee)

            l_hip_y, l_knee_y= l_hip[1], l_knee[1]
            r_hip_y, r_knee_y= r_hip[1], r_knee[1]

            left_knee_lifted= l_knee_y < l_hip_y - KNEE_LIFT_RATIO
            right_knee_lifted= r_knee_y < r_hip_y - KNEE_LIFT_RATIO

            left_knee_down= left_angle  > KNEE_DOWN_ANGLE and l_knee_y > l_hip_y
            right_knee_down= right_angle > KNEE_DOWN_ANGLE and r_knee_y > r_hip_y

            if left_knee_lifted and left_angle < KNEE_UP_ANGLE and not left_knee_up and not right_knee_up:
                counter     += 1
                left_knee_up = True
            if left_knee_down:
                left_knee_up = False

            if right_knee_lifted and right_angle < KNEE_UP_ANGLE and not right_knee_up and not left_knee_up:
                counter      += 1
                right_knee_up = True
            if right_knee_down:
                right_knee_up = False

            if torso_angle > TORSO_LEAN_MAX:
                feedback = f"Stand upright! Torso lean: {torso_angle:.0f}deg"

            elif left_knee_up:
                if left_thigh_angle < THIGH_LIFT_MIN:
                    feedback = f"Drive left knee higher! Thigh: {left_thigh_angle:.0f}deg"
                elif left_knee_bend > KNEE_BEND_MIN:
                    feedback = f"Bend left knee more! Bend: {left_knee_bend:.0f}deg"
                elif left_angle > 80:
                    feedback = "Tuck left heel up more"

            elif right_knee_up:
                if right_thigh_angle < THIGH_LIFT_MIN:
                    feedback = f"Drive right knee higher! Thigh: {right_thigh_angle:.0f}deg"
                elif right_knee_bend > KNEE_BEND_MIN:
                    feedback = f"Bend right knee more! Bend: {right_knee_bend:.0f}deg"
                elif right_angle > 80:
                    feedback = "Tuck right heel up more"

            else:
                if not left_knee_down or not right_knee_down:
                    feedback = "Fully lower your leg"

        except Exception:
            pass

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.putText(image, 'REPS',        (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),   1)
        cv2.putText(image, str(counter),  (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 2,   (0, 0, 0), 2)

        if not is_side_view:
            cv2.putText(image, "Turn Side-On to Camera", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if feedback:
            cv2.putText(image, feedback, (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        else:
            cv2.putText(image, "Good form!", (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        cv2.putText(image, f"L Hip angle: {left_angle:.0f} deg",
                    (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(image, f"R Hip angle: {right_angle:.0f} deg",
                    (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
       
        cv2.putText(image, f"L Knee bend: {left_knee_bend:.0f} deg",
                    (10, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(image, f"R Knee bend: {right_knee_bend:.0f} deg",
                    (10, 268), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.putText(image, f"Torso angle: {torso_angle:.0f} deg",
                    (10, 312), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.putText(image, f"L Thigh lift: {left_thigh_angle:.0f} deg",
                    (10, 354), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(image, f"R Thigh lift: {right_thigh_angle:.0f} deg",
                    (10, 377), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        draw_stop_button(image)
        cv2.imshow(window_name, image)

        if stop_flag or (cv2.waitKey(10) & 0xFF == ord('q')):
            break

cap.release()
cv2.destroyAllWindows()

print(f"\nSession summary:")
print(f"  Total reps: {counter}")

show_summary(counter)