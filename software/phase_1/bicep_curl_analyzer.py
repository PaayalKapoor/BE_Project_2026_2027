import cv2
import mediapipe as mp
import numpy as np

window_name = "PowerPulse - Bicep Curl Analyzer"
summary_window_name = "Bicep Summary"
stop_flag = False
stop_button_rect = [0, 0, 0, 0]
summary_close_flag = False
summary_close_rect = (200, 130, 310, 175)

def get_stop_button_rect(frame):
    h, w = frame.shape[:2]
    x2 = w - 15
    y2 = h - 15
    x1 = x2 - 150
    y1 = y2 - 55
    return [x1, y1, x2, y2]


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
    cv2.circle(image, (x1 + radius, y1 + radius), radius, color, -1, cv2.LINE_AA)
    cv2.circle(image, (x2 - radius, y1 + radius), radius, color, -1, cv2.LINE_AA)
    cv2.circle(image, (x1 + radius, y2 - radius), radius, color, -1, cv2.LINE_AA)
    cv2.circle(image, (x2 - radius, y2 - radius), radius, color, -1, cv2.LINE_AA)


def draw_stop_button(frame):
    x1, y1, x2, y2 = stop_button_rect
    rounded_rect(frame, (x1, y1, x2, y2), (0, 0, 255), radius=15)
    text = "STOP"
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
    text_x = x1 + (x2 - x1 - text_size[0]) // 2
    text_y = y1 + (y2 - y1 + text_size[1]) // 2
    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    return frame


def show_frame(frame):
    draw_stop_button(frame)
    cv2.imshow(window_name, frame)


def make_summary_image(total_reps, avg_accuracy, left_reps, right_reps):
    summary = np.zeros((220, 340, 3), np.uint8)
    summary[:] = (40, 40, 40)
    cv2.putText(summary, "SUMMARY", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(summary, f"Total Reps: {total_reps}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(summary, f"Left: {left_reps}  Right: {right_reps}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(summary, f"Accuracy: {avg_accuracy:.1f}%", (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    rounded_rect(summary, summary_close_rect, (0, 0, 255), radius=12)
    cv2.putText(summary, "Close", (summary_close_rect[0] + 25, summary_close_rect[1] + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return summary


def show_summary(total_reps, avg_accuracy, left_reps, right_reps):
    global summary_close_flag
    summary_close_flag = False
    cv2.namedWindow(summary_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(summary_window_name, 340, 220)
    cv2.setMouseCallback(summary_window_name, on_summary_mouse, summary_close_rect)
    while True:
        summary_image = make_summary_image(total_reps, avg_accuracy, left_reps, right_reps)
        cv2.imshow(summary_window_name, summary_image)
        key = cv2.waitKey(20) & 0xFF
        if key == 27 or summary_close_flag:
            break
    cv2.destroyWindow(summary_window_name)


ALPHA= 0.5   
MAX_ELBOW_DELTA= 15    #max degrees elbow angle can change per frame
MAX_TORSO_DELTA= 10    #max degrees torso lean can change per frame
MAX_UPPER_DELTA= 15    #max degrees upper arm drift can change per frame
S3_MIN_FRAMES= 4     #frames arm must stay curled to lock a rep
S1_MIN_FRAMES= 3     #frames arm must stay extended to confirm rep count
REP_ACCURACY_THRESHOLD= 60.0  #minimum % accuracy to count a rep



def calculate_joint_angles(x1, y1, x2, y2, x3, y3):
    cosine_angle = (
        (x1 - x2) * (x3 - x2) + (y1 - y2) * (y3 - y2)
    ) / (
        np.sqrt((x1 - x2)**2 + (y1 - y2)**2) *
        np.sqrt((x3 - x2)**2 + (y3 - y2)**2)
    )
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))


def calculate_torso_lean(shoulder_x, shoulder_y, hip_x, hip_y):
    cosine_angle = ((shoulder_y - hip_y) * (-1)) / (
        np.sqrt((shoulder_x - hip_x)**2 + (shoulder_y - hip_y)**2)
    )
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    lean = np.degrees(np.arccos(cosine_angle))
    return lean if shoulder_x < hip_x else -lean


def calculate_upper_arm_drift(shoulder_x, shoulder_y, elbow_x, elbow_y):
    cosine_angle = ((elbow_y - shoulder_y) * 1) / (
        np.sqrt((elbow_x - shoulder_x)**2 + (elbow_y - shoulder_y)**2)
    )
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))


def is_front_facing(landmarks):
    left_shoulder  = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    shoulder_width = abs(left_shoulder.x - right_shoulder.x)
    if shoulder_width < 0.20:
        return False, "Please Face the Camera Directly"
    return True, "Good Position"


def get_feedback(state, elbow_angle, torso_lean, upper_arm):
    if state == "S1":
        if elbow_angle < 140:
            return "Extend Arm Fully Before Next Rep", True
        if abs(torso_lean) > 25:
            return "Stand Upright: Reset Posture", True
        if upper_arm > 30:
            return "Bring Elbows Back to Your Sides", True
        return "Good Starting Position", False

    elif state == "S2":
        return "Curling", False

    elif state == "S3":
        if abs(torso_lean) > 35:
            return "Too Much Body Lean: Reduce Weight", True
        if upper_arm > 40:
            return "Elbow Drifted Forward: Tuck Back", True
        if elbow_angle > 75:
            return "Curl Higher: Full Range of Motion", True
        if elbow_angle < 30:
            return "Too Far: Control the Contraction", True
        return "Good Contraction: Squeeze!", False

    return "", False


mp_pose= mp.solutions.pose
mp_drawing= mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
    model_complexity=1,
    smooth_landmarks=True
)


def make_arm_state():
    return {
        "state": "S1",
        "reps": 0,
        "rep_lock": False,
        "initialized": False,
        "error": False,
        "prev_elbow": 0.0,
        "prev_torso":0.0,
        "prev_upper_arm": 0.0,
        "s3_frame_count": 0,
        "s1_frame_count": 0,
        "frame_scores": [],
        "rep_accuracies": [],
        "current_rep_accuracy": 0.0,
        "feedback": "Good Starting Position",
        "elbow_angle": 0.0,
    }

arms = {
    "L": make_arm_state(),
    "R": make_arm_state(),
}

LANDMARKS = {
    "L": {
        "shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
        "elbow": mp_pose.PoseLandmark.LEFT_ELBOW,
        "wrist": mp_pose.PoseLandmark.LEFT_WRIST,
        "hip": mp_pose.PoseLandmark.LEFT_HIP,
    },
    "R": {
        "shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
        "elbow": mp_pose.PoseLandmark.RIGHT_ELBOW,
        "wrist": mp_pose.PoseLandmark.RIGHT_WRIST,
        "hip": mp_pose.PoseLandmark.RIGHT_HIP,
    },
}

DISPLAY = {
    "L": {"label": "LEFT",  "color_label": (255, 200, 0)},
    "R": {"label": "RIGHT", "color_label": (0, 200, 255)},
}


video = cv2.VideoCapture(0)
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 640, 480)
cv2.setMouseCallback(window_name, on_mouse, stop_button_rect)

while True:
    isTrue, frame = video.read()
    if not isTrue:
        break
    stop_button_rect[:] = get_stop_button_rect(frame)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results   = pose.process(rgb_frame)

    if results.pose_landmarks:
        h, w, _ = frame.shape
        landmarks = results.pose_landmarks.landmark

        is_valid, position_msg = is_front_facing(landmarks)
        if not is_valid:
            cv2.putText(frame, position_msg,
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            show_frame(frame)
            if stop_flag or cv2.waitKey(20) & 0xFF == ord('q'):
                break
            continue

        for side, arm in arms.items():
            lm  = LANDMARKS[side]
            vis = landmarks[lm["elbow"]].visibility

            if vis < 0.6:
                arm["feedback"] = "Arm Not Visible"
                continue

            def px(lm_id):
                lmk = landmarks[lm_id]
                return int(lmk.x * w), int(lmk.y * h)

            sx, sy = px(lm["shoulder"])
            ex, ey = px(lm["elbow"])
            wx, wy = px(lm["wrist"])
            hx, hy = px(lm["hip"])

            #raw angles
            elbow_raw = calculate_joint_angles(sx, sy, ex, ey, wx, wy)
            torso_raw = calculate_torso_lean(sx, sy, hx, hy)
            upper_arm_raw = calculate_upper_arm_drift(sx, sy, ex, ey)

            if not arm["initialized"]:
                arm["prev_elbow"]= elbow_raw
                arm["prev_torso"]= torso_raw
                arm["prev_upper_arm"]= upper_arm_raw
                arm["initialized"]= True

            elbow_raw = np.clip(elbow_raw, arm["prev_elbow"] - MAX_ELBOW_DELTA, arm["prev_elbow"] + MAX_ELBOW_DELTA)
            torso_raw = np.clip(torso_raw, arm["prev_torso"] - MAX_TORSO_DELTA, arm["prev_torso"] + MAX_TORSO_DELTA)
            upper_arm_raw = np.clip(upper_arm_raw, arm["prev_upper_arm"] - MAX_UPPER_DELTA, arm["prev_upper_arm"] + MAX_UPPER_DELTA)

            elbow_angle= ALPHA * arm["prev_elbow"] + (1 - ALPHA) * elbow_raw
            torso_lean= ALPHA * arm["prev_torso"] + (1 - ALPHA) * torso_raw
            upper_arm= ALPHA * arm["prev_upper_arm"] + (1 - ALPHA) * upper_arm_raw

            arm["prev_elbow"] = elbow_angle
            arm["prev_torso"] = torso_lean
            arm["prev_upper_arm"] = upper_arm
            arm["elbow_angle"] = elbow_angle

            if elbow_angle <= 90:
                if arm["state"] != "S3":
                    arm["frame_scores"] = []       
                arm["state"]          = "S3"
                arm["s1_frame_count"] = 0           

                arm["s3_frame_count"] += 1

                if arm["s3_frame_count"] >= S3_MIN_FRAMES:
                    arm["rep_lock"] = True

                if arm["rep_lock"]:
                    elbow_score= max(0, 1 - (abs(elbow_angle - 45) / 40))
                    torso_score= max(0, 1 - (abs(torso_lean)       / 35))
                    upper_arm_score= max(0, 1 - (abs(upper_arm - 10)   / 30))
                    frame_score= (elbow_score* 0.50 +torso_score* 0.30 + upper_arm_score * 0.20)
                    arm["frame_scores"].append(frame_score)

            elif elbow_angle > 155:
                arm["s3_frame_count"] = 0           
                arm["s1_frame_count"] += 1

                if arm["s1_frame_count"] >= S1_MIN_FRAMES:
                    if arm["rep_lock"] and not arm["error"]:
                        if arm["frame_scores"]:
                            arm["current_rep_accuracy"] = (
                                sum(arm["frame_scores"]) / len(arm["frame_scores"])
                            ) * 100
                        
                            if arm["current_rep_accuracy"] >= REP_ACCURACY_THRESHOLD:
                                arm["reps"] += 1
                                arm["rep_accuracies"].append(arm["current_rep_accuracy"])
                            else:
                                arm["feedback"] = (
                                    f"Rep Not Counted - "
                                    f"{arm['current_rep_accuracy']:.0f}% Accuracy"
                                )
                    arm["rep_lock"]= False
                    arm["error"]= False
                    arm["frame_scores"]= []
                    arm["state"]= "S1"

            else:
                arm["s3_frame_count"]= 0
                arm["s1_frame_count"]= 0
                arm["state"]= "S2"

            if arm["state"] != "S1" or arm["s1_frame_count"] > 1:
                fb, err = get_feedback(
                    arm["state"], elbow_angle, torso_lean, upper_arm)
                arm["feedback"]= fb
                arm["error"]= err

        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        panel_w = w // 2
        cv2.line(frame, (panel_w, 0), (panel_w, 215), (80, 80, 80), 1)

        for side, arm in arms.items():
            x_off= 30 if side == "L" else panel_w + 30
            label_color = DISPLAY[side]["color_label"]
            feedback= arm["feedback"]

            if "Good" in feedback:
                fb_color = (0, 255, 0)
            elif arm["error"] or "Not Counted" in feedback:
                fb_color = (0, 0, 255)
            else:
                fb_color = (0, 165, 255)

            cv2.putText(frame, DISPLAY[side]["label"] + " ARM", (x_off, 35),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, label_color, 2)
            cv2.putText(frame, feedback, (x_off, 65),  cv2.FONT_HERSHEY_SIMPLEX, 0.6,  fb_color, 2)
            cv2.putText(frame, "State: " + arm["state"], (x_off, 95),  cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
            cv2.putText(frame, "Reps: " + str(arm["reps"]), (x_off, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.8,  (255, 255, 255), 2)
            if arm["reps"] > 0:
                cv2.putText(frame, f"Last Rep: {arm['current_rep_accuracy']:.1f}%",
                            (x_off, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            if arm["rep_accuracies"]:
                avg = sum(arm["rep_accuracies"]) / len(arm["rep_accuracies"])
                cv2.putText(frame, f"Avg: {avg:.1f}%",
                            (x_off, 178), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(frame, f"Elbow: {arm['elbow_angle']:.1f}",
                        (x_off, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (180, 180, 180), 1)

    show_frame(frame)
    if stop_flag or cv2.waitKey(20) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
all_reps = arms["L"]["reps"] + arms["R"]["reps"]
all_scores = arms["L"]["rep_accuracies"] + arms["R"]["rep_accuracies"]
avg_accuracy = sum(all_scores) / len(all_scores) if all_scores else 0.0
show_summary(all_reps, avg_accuracy, arms["L"]["reps"], arms["R"]["reps"])
