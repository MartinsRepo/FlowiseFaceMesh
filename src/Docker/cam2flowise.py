# Docker Version
#
# Features: 
# Captures frames from the Webcam, generates a Facemesh and
# sends the Json -yfied data to Flowise
#
# Proto file compilation:
# protoc -I =. --python_out=. facedata.proto

# Microsoft LiveCam - Adjust image
# keyboard shortcuts that you can use to manage the zoom out/in feature of camera: 
# Zoom Out = Ctrl + Minus Key, Zoom In = Ctrl + Plus key, Zoom to 100% = Ctrl + Zero key. 

import sys, os, cv2, time, math, zmq, json, requests, multiprocessing
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
import mediapipe as mp
from dotenv import load_dotenv
from multiprocessing import Queue, Process

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

load_dotenv()
FLOW_ID = os.getenv('FLOWID')
FLOWISE_API_URL = f"http://flowise:3000/api/v1/prediction/{FLOW_ID}"
FLASK_API_URL = "http://192.168.0.228:5000/api/interpret"

face_oval_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361,
                     288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149,
                     150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,
                     67, 109]

enum_LM = ["nose", "chin", "left eye", "upper lid left", "lower lid left",
           "right eye", "upper lid right", "lower lid right",
           "left mouth", "right mouth", "upper lip", "lower lip"]

mesh_indices = [19, 152, 226, 27, 23, 446, 257, 253, 57, 287, 0, 7]

def init_detector():
    base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1
    )
    return vision.FaceLandmarker.create_from_options(options)

def detect_landmarks(detector, image, width, height):
    result = detector.detect(image)
    if not result.face_landmarks:
        return None, None
    landmarks = result.face_landmarks[0]
    all_points = [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]
    points = [(all_points[idx][0], all_points[idx][1]) for idx in mesh_indices]
    return points, all_points

def render_mesh(image, landmarks):
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=landmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
    )
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=landmarks,
        connections=mp_face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
    )

def GenFaceMesh(facemarkers, all_points):
    packedlM = []
    packedFO = []
    for name, (x, y) in zip(enum_LM, facemarkers):
        packedlM.append({"name": name, "x": x, "y": y})
    for idx in face_oval_indices:
        x, y = all_points[idx]
        packedFO.append({"name": f"oval_{idx}", "x": x, "y": y})
    return {"filename": "Processed frame data", "packedlM": packedlM, "packedFO": packedFO}

def convert_protobuf_to_dict(input_dict):
    frame_data = {'message': input_dict.get('filename', 'Filename not found')}
    landmarks = [
        {"x": lm.get("x"), "y": lm.get("y"), "name": lm.get("name")}
        for lm in input_dict.get("packedlM", [])
    ]
    frame_data['landmarks'] = landmarks
    return frame_data

def cam_capture(raw_frame_queue):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret:
            raw_frame_queue.put(frame)
        time.sleep(0.01)

def landmark_process(raw_frame_queue, annotated_frame_queue, llm_queue):
    detector = init_detector()
    mesh_drawer = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
    prev_landmarks = None

    while True:
        frame = raw_frame_queue.get()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        cur_pts, all_pts = detect_landmarks(detector, mp_image, width, height)

        if cur_pts:
            if prev_landmarks is None or any(abs(x1 - x0) > 3 or abs(y1 - y0) > 3 for (x0, y0), (x1, y1) in zip(prev_landmarks, cur_pts)):
                msg = GenFaceMesh(cur_pts, all_pts)
                payload = {"question": convert_protobuf_to_dict(msg)}
                llm_queue.put(payload)
                prev_landmarks = cur_pts.copy()

        result = mesh_drawer.process(rgb)
        if result.multi_face_landmarks:
            for fl in result.multi_face_landmarks:
                render_mesh(frame, fl)

        annotated_frame_queue.put(frame)


def zmq_streamer(annotated_frame_queue):
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:5555")
    while True:
        frame = annotated_frame_queue.get()
        _, buffer = cv2.imencode('.jpg', frame)
        socket.send(buffer.tobytes())


def llm_process(llm_queue, flask_queue):
    while True:
        payload = llm_queue.get()
        try:
            response = requests.post(FLOWISE_API_URL, json=payload)
            if response.status_code == 200:
                data = response.json()
                print("LLM Output:", data.get("text"))
                flask_queue.put({"text": data.get("text")})
            else:
                print("LLM Error:", response.status_code)
        except Exception as e:
            print("LLM call failed:", e)


def flask_process(flask_queue):
    while True:
        payload = flask_queue.get()
        try:
            print("\n--- Sending to Flask API ---")
            response = requests.post(FLASK_API_URL, json=payload)
            print("Flask response:", response.status_code)
        except Exception as e:
            print("Flask call failed:", e)


def main():
    raw_frame_queue = Queue()
    annotated_frame_queue = Queue()
    llm_queue = Queue()
    flask_queue = Queue()

    processes = [
        Process(target=cam_capture, args=(raw_frame_queue,), daemon=True),
        Process(target=landmark_process, args=(raw_frame_queue, annotated_frame_queue, llm_queue), daemon=True),
        Process(target=zmq_streamer, args=(annotated_frame_queue,), daemon=True),
        Process(target=llm_process, args=(llm_queue, flask_queue), daemon=True),
        Process(target=flask_process, args=(flask_queue,), daemon=True)
    ]

    for p in processes:
        p.start()
    for p in processes:
        p.join()

if __name__ == '__main__':
    main()




