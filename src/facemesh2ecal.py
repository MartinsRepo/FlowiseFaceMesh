# protoc -I =. --python_out=. facemesh.proto


from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import sys, os, cv2, time
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import ecal.core.core as ecal_core
from ecal.core.publisher import ProtoPublisher
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import facemesh_pb2

# load configuration from .env file
load_dotenv() 
usecam = str(os.getenv('USECAM'))
picfolder = str(os.getenv('FOLDER'))
pic = str(os.getenv('IMAGE'))


face_oval_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361,
                     288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149,
                     150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,
                     67, 109]

enum_LM = ["nose", "chin", "left eye", "upper lid left", "lower lid left", "right eye", "upper lid right", "lower lid right", "left mouth", "right mouth", "upper lip", "lower lip"]

heigt = 1
width = 1

base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
								output_face_blendshapes=True,
								output_facial_transformation_matrixes=True,
								num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)


def draw_landmarks_on_image(rgb_image, detection_result):
	face_landmarks_list = detection_result.face_landmarks
	annotated_image = np.copy(rgb_image)

	# Loop through the detected faces to visualize.
	for idx in range(len(face_landmarks_list)):
		face_landmarks = face_landmarks_list[idx]

	# Draw the face landmarks.
	face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
	face_landmarks_proto.landmark.extend([
	  landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
	])

	solutions.drawing_utils.draw_landmarks(
		image=annotated_image,
		landmark_list=face_landmarks_proto,
		connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
		landmark_drawing_spec=None,
		connection_drawing_spec=mp.solutions.drawing_styles
		.get_default_face_mesh_tesselation_style())
	solutions.drawing_utils.draw_landmarks(
		image=annotated_image,
		landmark_list=face_landmarks_proto,
		connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
		landmark_drawing_spec=None,
		connection_drawing_spec=mp.solutions.drawing_styles
		.get_default_face_mesh_contours_style())
	solutions.drawing_utils.draw_landmarks(
		image=annotated_image,
		landmark_list=face_landmarks_proto,
		connections=mp.solutions.face_mesh.FACEMESH_IRISES,
		  landmark_drawing_spec=None,
		  connection_drawing_spec=mp.solutions.drawing_styles
		  .get_default_face_mesh_iris_connections_style())

	return annotated_image

'''
def plot_face_blendshapes_bar_graph(face_blendshapes):
	# Extract the face blendshapes category names and scores.
	face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
	face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
	# The blendshapes are ordered in decreasing score value.
	face_blendshapes_ranks = range(len(face_blendshapes_names))

	fig, ax = plt.subplots(figsize=(12, 12))
	bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
	ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
	ax.invert_yaxis()

	# Label each bar with values
	for score, patch in zip(face_blendshapes_scores, bar.patches):
		plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

	ax.set_xlabel('Score')
	ax.set_title("Face Blendshapes")
	plt.tight_layout()
	plt.show()
''' 
  
def CalcMesh(image):
	global height, width
	
	ip_arr = None
	all_points = None
    
	# get 2d canonical face matrix and central image points
	def getImagePoints(landmarks, iw, ih):
		global height, width
		
		faceXY = []
        
		if detection_result.face_landmarks:
        
			full_landmarks = detection_result.face_landmarks[0]
			all_points = [(int(lm.x * width), int(lm.y * height)) for lm in full_landmarks]
			first_face_landmarks = detection_result.face_landmarks[0]
            
			for landmark in first_face_landmarks:
				x = int(landmark.x*iw)
				y = int(landmark.y*iw)
				faceXY.append((x, y))	# put all xy points in neat array
                
			image_points = np.array([
				faceXY[19],     # "nose"
				faceXY[152],    # "chin"
				faceXY[226],    # "left eye"
				faceXY[27],     # "upper lid left"
				faceXY[23],     # "lower lid left"
				faceXY[446],    # "right eye"
				faceXY[257],    # "upper lid right"
				faceXY[253],    # "lower lid right"
				faceXY[57],     # "left mouth"
				faceXY[287],    # "right mouth"
				faceXY[0],      # "upper lip"
				faceXY[7],      # "lower lip"
			], dtype="double")
            
		return image_points, all_points

	
	#Detect face landmarks from the input image.
	detection_result = detector.detect(image)
	
	# If face detected, crop to bounding box
	if detection_result.face_landmarks:
		face = detection_result.face_landmarks[0]

		xs = [lm.x for lm in face]
		ys = [lm.y for lm in face]

		# Expand bounding box with margin
		min_x, max_x = max(min(xs) - 0.1, 0.0), min(max(xs) + 0.1, 1.0)
		min_y, max_y = max(min(ys) - 0.1, 0.0), min(max(ys) + 0.1, 1.0)

		img_np = image.numpy_view()
		ih, iw = img_np.shape[:2]

		x1, x2 = int(min_x * iw), int(max_x * iw)
		y1, y2 = int(min_y * ih), int(max_y * ih)

		# Crop and resize to original shape for re-detection
		cropped = img_np[y1:y2, x1:x2]
		cropped_resized = cv2.resize(cropped, (iw, ih))

		# Convert back to MediaPipe Image and detect again
		focus_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cropped_resized)
		detection_result = detector.detect(focus_image)
		annotated_image = draw_landmarks_on_image(cropped_resized, detection_result)
	else:
		annotated_image = image.numpy_view()
		
	height, width, _ = annotated_image.shape
		
	if detection_result.face_landmarks:
		for face_landmarks in detection_result.face_landmarks:
			ip_arr, all_points = getImagePoints(face_landmarks, width, height)
	
	return annotated_image, ip_arr, all_points
	
	'''
	if len(detection_result.face_landmarks)>0:

		#process the detection result. In this case, visualize it.
		annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)

		height, width, _ = annotated_image.shape
		
		if detection_result.face_landmarks:
			for face_landmarks in detection_result.face_landmarks:
				ip_arr, all_points = getImagePoints(face_landmarks, width, height)
		
		return annotated_image, ip_arr, all_points
		
	else: 
		return image, None, None
	'''

def EcalSetup():
	# initialize eCAL API
	ecal_core.initialize(sys.argv, "Camera")

	# set process state
	ecal_core.set_process_state(1, 1, "Ready to publish")

	pub = ProtoPublisher("FaceCoords",facemesh_pb2.FD)

	return pub

def PubFaceMesh(facemarkers, all_points):

	data2ecal = facemesh_pb2.FD()
	data2ecal.filename    = 'Ecal2Flowise'
	data2ecal.framewidth  = width
	data2ecal.frameheight = height

	new_LMlist = data2ecal.packedlM
	enumLM_list = enumerate(enum_LM)
	
	if facemarkers is not None:
		for (x, y) in facemarkers:

			# Iterating list using enumerate to get both index and element
			try:
				# pack into the list
				new_block = facemesh_pb2.Landmarks()
				nxt_lm = next(enumLM_list)
				new_block.name = str(nxt_lm[1])
				new_block.x = int(x)
				new_block.y = int(y)
				new_LMlist.append(new_block)   

			except ValueError:
				pass
				
		# Extract face oval separately
		oval_list = data2ecal.packedFO
		
		for idx in face_oval_indices:
			try:
				 x, y = all_points[idx]
				 oval_point = facemesh_pb2.FaceOval()
				 oval_point.name = f"oval_{int(idx)}"
				 oval_point.x = int(x)
				 oval_point.y = int(y)
				 oval_list.append(oval_point)
			except IndexError:
				pass
    
	return data2ecal

if __name__ == "__main__":
	print('##################################################')
	print('#')
	print('# Bringing Webcam Frames to ECAL Protobuf Messages')
	print('#')    
	print('##################################################')
	print('#')
	
	# initialize ECAL message
	pub = EcalSetup()
    
	if usecam == "True":
    	
		cap = cv2.VideoCapture(0)
        
		if not cap.isOpened():
			print("❌ Could not open webcam.")
			sys.exit(1)
        
		while ecal_core.ok():
			ret, frame = cap.read()
			if not ret:
				print("❌ Failed to grab frame.")
				break

			# Convert OpenCV BGR to RGB
			frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

			# Convert to MediaPipe Image
			mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

			# Run detection
			annotated_image, facemarkers, all_points = CalcMesh(mp_image)
			
			data2ecal = PubFaceMesh(facemarkers, all_points)
			pub.send(data2ecal)

			# Show annotated frame
			cv2.imshow("Facemesh", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
			
			if cv2.waitKey(1) & 0xFF == 27:  # ESC key to break
				break

		cap.release()
		cv2.destroyAllWindows()
    
	else:
    
		#Load the input image.
		image = os.getcwd()+'/'+picfolder+pic	

		image = mp.Image.create_from_file(image)

		annotated_image, facemarkers, all_points = CalcMesh(image)

		data2ecal =  PubFaceMesh(facemarkers,all_points)

		window_name = 'Testimage'
		while ecal_core.ok():
			pub.send(data2ecal)
			time.sleep(0.5)
			cv2.imshow(window_name, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
			cv2.waitKey(500)
        
	# finalize eCAL API
	ecal_core.finalize()





