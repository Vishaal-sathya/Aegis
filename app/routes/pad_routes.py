from flask import Blueprint, request, jsonify, render_template
import cv2, base64, time, random
import numpy as np
import mediapipe as mp

pad_bp = Blueprint("pad", __name__)

# ------------------------------
# MediaPipe setup
# ------------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    refine_landmarks=True,
    max_num_faces=1
)


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
NOSE_TIP = 1

def np_point(landmarks, idx):
    return np.array([landmarks[idx].x, landmarks[idx].y], dtype=np.float32)

def eye_aspect_ratio(landmarks, eye_idx):
    p1, p2, p3, p4, p5, p6 = [np_point(landmarks, i) for i in eye_idx]
    return (np.linalg.norm(p2-p6) + np.linalg.norm(p3-p5)) / (2.0 * np.linalg.norm(p1-p4) + 1e-6)

def head_turn_direction(landmarks, left_thresh=0.35, right_thresh=0.65):
    nose_x = float(landmarks[NOSE_TIP].x)
    if nose_x < left_thresh:
        return "right"
    elif nose_x > right_thresh:
        return "left"
    return "center"

# ------------------------------
# Alignment check
# ------------------------------
def check_alignment(landmarks, frame_shape):
    h, w, _ = frame_shape
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]

    x1, x2 = min(xs) * w, max(xs) * w
    y1, y2 = min(ys) * h, max(ys) * h
    face_w, face_h = x2 - x1, y2 - y1

    if face_w < 0.2 * w or face_h < 0.2 * h:
        return False, "Face too far/small"

    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    if cx < 0.3 * w or cx > 0.7 * w or cy < 0.3 * h or cy > 0.7 * h:
        return False, "Face not centered"

    return True, "Face aligned"

# ------------------------------
# Challenge system with timeout
# ------------------------------
ALL_CHALLENGES = ["alignment", "blink", "turn_left", "turn_right"]
challenge_list = []
challenge_index = 0
challenge_start_time = None
CHALLENGE_TIMEOUT = 10  # seconds

def reset_challenges():
    global challenge_list, challenge_index, challenge_start_time
    challenge_list = random.sample(ALL_CHALLENGES, len(ALL_CHALLENGES))
    challenge_index = 0
    challenge_start_time = time.time()

@pad_bp.route("/start_session", methods=["POST"])
def start_session():
    reset_challenges()
    return jsonify({"status": "ok", "message": "New challenge session started"})

# ------------------------------
# Challenge instructions
# ------------------------------
CHALLENGE_INSTRUCTIONS = {
    "alignment": "Please center your face in the camera",
    "blink": "Blink your eyes",
    "turn_left": "Turn your face to the left",
    "turn_right": "Turn your face to the right",
}

@pad_bp.route("/process_frame", methods=["POST"])
def process_frame():
    global challenge_index, challenge_start_time

    data = request.json
    if not data or "frame" not in data:
        return jsonify({"challenge": "failed", "message": "⚠️ No frame received", "passed": False})

    try:
        img_data = data["frame"].split(",")[1]
        img = base64.b64decode(img_data)
        np_img = np.frombuffer(img, np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"challenge": "failed", "message": "⚠️ Invalid frame", "passed": False})
    except Exception as e:
        return jsonify({"challenge": "failed", "message": f"⚠️ Frame decode error: {str(e)}", "passed": False})

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)

    if challenge_index >= len(challenge_list):
        return jsonify({"challenge": "done", "message": "✅ All challenges passed!", "passed": True})

    current_challenge = challenge_list[challenge_index]
    elapsed = time.time() - challenge_start_time

    # Use friendly instructions by default
    status = {
        "challenge": current_challenge,
        "passed": False,
        "message": CHALLENGE_INSTRUCTIONS.get(current_challenge, "Follow the challenge")
    }

    if elapsed > CHALLENGE_TIMEOUT:
        reset_challenges()
        return jsonify({"challenge": "failed", "message": "❌ Spoof Detected (timeout)", "passed": False})

    if not res.multi_face_landmarks:
        status["message"] = "No face detected"
        return jsonify(status)

    landmarks = res.multi_face_landmarks[0].landmark

    if current_challenge == "alignment":
        ok, msg = check_alignment(landmarks, frame.shape)
        status["message"] = msg if not ok else "✅ Face centered"
        if ok:
            status["passed"] = True
            challenge_index += 1
            challenge_start_time = time.time()
            if challenge_index < len(challenge_list):
                status["next_challenge"] = CHALLENGE_INSTRUCTIONS[challenge_list[challenge_index]]

    elif current_challenge == "blink":
        ear_l = eye_aspect_ratio(landmarks, LEFT_EYE)
        ear_r = eye_aspect_ratio(landmarks, RIGHT_EYE)
        if ear_l < 0.2 and ear_r < 0.2:
            status["message"] = "✅ Blink detected"
            status["passed"] = True
            challenge_index += 1
            challenge_start_time = time.time()
            if challenge_index < len(challenge_list):
                status["next_challenge"] = CHALLENGE_INSTRUCTIONS[challenge_list[challenge_index]]

    elif current_challenge == "turn_left":
        direction = head_turn_direction(landmarks)
        status["message"] = "Please turn your face left"
        if direction == "left":
            status["message"] = "✅ Face turned left"
            status["passed"] = True
            challenge_index += 1
            challenge_start_time = time.time()
            if challenge_index < len(challenge_list):
                status["next_challenge"] = CHALLENGE_INSTRUCTIONS[challenge_list[challenge_index]]

    elif current_challenge == "turn_right":
        direction = head_turn_direction(landmarks)
        status["message"] = "Please turn your face right"
        if direction == "right":
            status["message"] = "✅ Face turned right"
            status["passed"] = True
            challenge_index += 1
            challenge_start_time = time.time()
            if challenge_index < len(challenge_list):
                status["next_challenge"] = CHALLENGE_INSTRUCTIONS[challenge_list[challenge_index]]

    if challenge_index >= len(challenge_list):
        reset_challenges()
        return jsonify({"challenge": "done", "message": "✅ All challenges passed!", "passed": True})

    return jsonify(status)

# @pad_bp.route("/")
# def index():
#     from ..config import PAD_MODE
#     return render_template("index.html", pad_mode=PAD_MODE)
