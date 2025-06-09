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


import sys, os, cv2, time, math, zmq, json, requests
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
import mediapipe as mp
from dotenv import load_dotenv

# drawing utilities from solutions API
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:5555")  # Binds to all interfaces on port 5555

# Load configuration from .env file
load_dotenv()
FLOW_ID = os.getenv('FLOWID')
FLOWISE_API_URL = f"http://flowise:3000/api/v1/prediction/{FLOW_ID}"


# Default model filename adjacent to project root
default_model = 'face_landmarker_v2_with_blendshapes.task'

face_oval_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361,
                     288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149,
                     150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,
                     67, 109]

# Landmark indices for face features of interest (12 points)
enum_LM = ["nose", "chin", "left eye", "upper lid left", "lower lid left",
           "right eye", "upper lid right", "lower lid right",
           "left mouth", "right mouth", "upper lip", "lower lip"]
           
face_data_dict = {
	    "filename": 'Processed frame data',
	    "packedlM": [],  # List to hold landmark dictionaries
	    "packedFO": []   # List to hold face oval dictionaries
	}

# Corresponding mesh indices
mesh_indices = [19, 152, 226, 27, 23, 446, 257, 253, 57, 287, 0, 7]

# Globals for image dims and previous landmarks
global_width,global_height = 1,1
prev_landmarks = None

# Initialize Mediapipe face landmarker
def init_detector():
	base_options = python.BaseOptions(model_asset_path=default_model)
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

def GenFaceMesh(facemarkers, all_points):
	
    # add selected landmarks
	for name, (x, y) in zip(enum_LM, facemarkers):
		lm_dict = {
			"name": name,
			"x": x,
			"y": y
		}
		face_data_dict["packedlM"].append(lm_dict)

    # add full face oval
	for idx in face_oval_indices:
		x, y = all_points[idx]
		ov_dict = {
			"name": f"oval_{idx}",
			"x": x,
			"y": y
		}
		face_data_dict["packedFO"].append(ov_dict)
        
	return face_data_dict
	

def convert_protobuf_to_dict(input_dict):

    """
    Convert a face data dictionary (expected to be similar to the protobuf structure)
    to a simplified dict with 'message' (filename) and 'landmarks' list.

    Args:
        input_dict (dict): A dictionary containing face data.
                           Expected keys: 'filename' (string) and 'packedlM' (list of dicts).
                           Each dict in 'packedlM' is expected to have 'x', 'y', 'name' keys.

    Returns:
        dict: A simplified dictionary with 'message' (filename) and 'landmarks' list.
    """
    # Access 'filename' using dictionary key lookup.
    # Using .get() with a default value to handle cases where the key might be missing.
    frame_data = {'message': input_dict.get('filename', 'Filename not found')}

    landmarks = []
    # Iterate through the list of landmark dictionaries under the 'packedlM' key.
    # Using .get() with an empty list as default to prevent errors if 'packedlM' is missing.
    for lm_dict in input_dict.get('packedlM', []):
        landmarks.append({
            'x': lm_dict.get('x'),
            'y': lm_dict.get('y'),
            'name': lm_dict.get('name')
        })
    frame_data['landmarks'] = landmarks
    return frame_data
    

def handle_llm_response(response):
    """
    Process and display the response from Flowise.
    """
    try:
        data = response.json()
    except json.JSONDecodeError:
        print("Invalid JSON response:", response.text)
        return

    if response.status_code == 200:
        print("Status Code:", response.status_code)
        print("Output from LLM:")
        #llm_text = response_json.get("text")
        llm_text = data.get("text")
        print(llm_text)
        print("Chat ID:", data.get('chatId'), "Message ID:", data.get('chatMessageId'))

        # Send to Flask
        print("\n--- Sending to Flask API ---")
        #flask_response = requests.post(FLASK_API_URL, json={"text": llm_text})

        if flask_response.status_code == 200:
            print("✅ Flask API Success:")
            #print(flask_response.json().get("message"))
        else:
            print("❌ Flask API Error:")
            print("Status Code:", flask_response.status_code)
            print("Response:", flask_response.text)
        print('Output finished')
    else:
        print(f"Error {response.status_code}: {data.get('message')}")
        if data.get('stack'):
            print(data.get('stack'))


if __name__ == '__main__':
	detector = init_detector()

	# FaceMesh solution for drawing
	mesh_drawer = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

	cap = cv2.VideoCapture(0)
	if not cap.isOpened():
		sys.exit(1)
	while True:
		ret, frame = cap.read()
		if not ret:
			break
		# convert and update dims
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		global_height, global_width = rgb.shape[:2]
		mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
		cur_pts, all_pts = detect_landmarks(detector, mp_image)
        
		if cur_pts and should_publish(prev_landmarks, cur_pts):
			msg = GenFaceMesh(cur_pts, all_pts)
			frame_data = convert_protobuf_to_dict(msg)
			payload = {"question": frame_data}
			response = requests.post(FLOWISE_API_URL, json=payload)
			handle_llm_response(response)
			
			prev_landmarks = cur_pts.copy()
        
            
		# Render mesh overlay
		result = mesh_drawer.process(rgb)
		if result.multi_face_landmarks:
			for fl in result.multi_face_landmarks:
				render_mesh(frame, fl)
        
		# send frame outside the docker container   
		_, buffer = cv2.imencode('.jpg', frame)
		socket.send(buffer.tobytes())
		time.sleep(0.03)  # ~30 FPS
            
	cap.release()

    






