# Proto file compilation:
# protoc -I =. --python_out=. facemesh.proto

# Microsoft LiveCam - Adjust image
# keyboard shortcuts that you can use to manage the zoom out/in feature of camera: 
# Zoom Out = Ctrl + Minus Key, Zoom In = Ctrl + Plus key, Zoom to 100% = Ctrl + Zero key. 


import sys, os, cv2, time, math
import numpy as np
from dotenv import dotenv_values
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
import mediapipe as mp
import ecal.core.core as ecal_core
from ecal.core.publisher import ProtoPublisher
import facemesh_pb2

# drawing utilities from solutions API
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# load values from .env
envpath = os.getcwd()+"/.env"
print('Environmentpath',envpath)
config = dotenv_values(envpath)
print('Actual configuration',config)
if config['USECAM']=='True':
    usecam = True
else:
    usecam = False

# Resolve absolute path to Mediapipe model file
script_dir = os.path.dirname(os.path.abspath(__file__))
# Default model filename adjacent to project root
default_model = os.path.abspath(os.path.join(script_dir, '', 'face_landmarker_v2_with_blendshapes.task'))
model_path = os.getenv('MODEL_PATH', default_model)
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Mediapipe model file not found at {model_path}")

face_oval_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361,
                     288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149,
                     150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,
                     67, 109]

# Landmark indices for face features of interest (12 points)
enum_LM = ["nose", "chin", "left eye", "upper lid left", "lower lid left",
           "right eye", "upper lid right", "lower lid right",
           "left mouth", "right mouth", "upper lip", "lower lip"]
# Corresponding mesh indices
mesh_indices = [19, 152, 226, 27, 23, 446, 257, 253, 57, 287, 0, 7]

# Globals for image dims and previous landmarks
global_width,global_height = 1,1
prev_landmarks = None

# Initialize Mediapipe face landmarker
def init_detector():
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1
    )
    return vision.FaceLandmarker.create_from_options(options)

def detect_landmarks(detector, image):
    result = detector.detect(image)
    if not result.face_landmarks:
        return None, None
    # extract 2D points and all mesh points
    landmarks = result.face_landmarks[0]
    all_points = [(int(lm.x * global_width), int(lm.y * global_height)) for lm in landmarks]
    points = []
    for idx in mesh_indices:
        x, y = all_points[idx]
        points.append((x, y))
    return points, all_points

# Decide if publish: compute fraction of points moved more than threshold (10% of frame width or height)
def should_publish(prev, curr, frac=0.008):
    if prev is None:
        return True
    moved = 0
    threshold_x = global_width * frac
    threshold_y = global_height * frac
    for (x0, y0), (x1, y1) in zip(prev, curr):
        if abs(x1 - x0) > threshold_x or abs(y1 - y0) > threshold_y:
            moved += 1
    # publish if more than 10% of landmarks moved
    return moved >= math.ceil(len(curr) * frac)

# Setup eCAL publisher
def EcalSetup():
    ecal_core.initialize(sys.argv, "Camera")
    ecal_core.set_process_state(1, 1, "Ready to publish")
    return ProtoPublisher("FaceCoords", facemesh_pb2.FD)

# Populate protobuf message
def PubFaceMesh(facemarkers, all_points):
    msg = facemesh_pb2.FD()
    msg.filename = 'Ecal2Flowise'
    msg.framewidth = global_width
    msg.frameheight = global_height
    # add selected landmarks
    for name, (x, y) in zip(enum_LM, facemarkers):
        lm = msg.packedlM.add()
        lm.name = name
        lm.x = x
        lm.y = y
    # add full face oval
    for idx in face_oval_indices:
        x, y = all_points[idx]
        ov = msg.packedFO.add()
        ov.name = f"oval_{idx}"
        ov.x = x
        ov.y = y
    return msg

def render_mesh(image, landmarks):
        # Draw tessellation
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )
        # Draw contours
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
        )


if __name__ == '__main__':
    detector = init_detector()
    pub = EcalSetup()
    
    # FaceMesh solution for drawing
    mesh_drawer = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

    if usecam:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open webcam.")
            sys.exit(1)
        while ecal_core.ok():
            ret, frame = cap.read()
            if not ret:
                break
            # convert and update dims
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            global_height, global_width = rgb.shape[:2]
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            cur_pts, all_pts = detect_landmarks(detector, mp_image)
            
            if cur_pts and should_publish(prev_landmarks, cur_pts):
                msg = PubFaceMesh(cur_pts, all_pts)
                pub.send(msg)
                prev_landmarks = cur_pts.copy()
                
            # Render mesh overlay
            result = mesh_drawer.process(rgb)
            if result.multi_face_landmarks:
                for fl in result.multi_face_landmarks:
                    render_mesh(frame, fl)
                    
            cv2.imshow("Facemesh", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
    
        image_path = os.path.join(os.getcwd(), os.getcwd()+"/src/"+config['FOLDER'], config['IMAGE'])
        bgr = cv2.imread(image_path)
        if bgr is None:
            print(f"❌ Cannot load image {image_path}"); sys.exit(1)
        frame_h, frame_w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image.create_from_file(image_path)
        
        pts, all_pts = detect_landmarks(detector, mp_img)
        if pts:
            pub.send(PubFaceMesh(pts, all_pts))
        
        # Render on static image
        result = mesh_drawer.process(rgb)
        if result.multi_face_landmarks:
            for fl in result.multi_face_landmarks:
                render_mesh(bgr, fl)
        
        while ecal_core.ok():
            cv2.imshow('TestImage', bgr)
            if cv2.waitKey(500) & 0xFF == 27:
                break
            
ecal_core.finalize()       







