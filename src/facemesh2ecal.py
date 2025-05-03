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

enum_LM = ["nose", "chin", "left eye", "upper lid left", "lower lid left", "right eye", "upper lid right", "lower lid right", "left mouth", "right mouth", "upper lip", "lower lip"]

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
  
def CalcMesh(image):
    
    # get 2d canonical face matrix and central image points
    def getImagePoints(landmarks, iw, ih):
        faceXY = []
        
        if detection_result.face_landmarks:
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
            
        return image_points

    base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                        output_face_blendshapes=True,
                                        output_facial_transformation_matrixes=True,
                                        num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)
    
    #Detect face landmarks from the input image.
    detection_result = detector.detect(image)
    
    #process the detection result. In this case, visualize it.
    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    
    height, width, _ = annotated_image.shape
    
    if detection_result.face_landmarks:
        for face_landmarks in detection_result.face_landmarks:
            ip_arr = getImagePoints(face_landmarks, width, height)
    
    return annotated_image, ip_arr

def EcalSetup():
    # initialize eCAL API
    ecal_core.initialize(sys.argv, "Camera")

    # set process state
    ecal_core.set_process_state(1, 1, "Ready to publish")
    
    pub = ProtoPublisher("FaceCoords",facemesh_pb2.FD)

    return pub

def PubFaceMesh(facemarkers):

    data2ecal = facemesh_pb2.FD()
    data2ecal.filename = 'Test'
    
    new_LMlist = data2ecal.packedlM
    enumLM_list = enumerate(enum_LM)
    
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
    
    return data2ecal

if __name__ == "__main__":
    print('##################################################')
    print('#')
    print('# Bringing Webcam Frames to ECAL Protobuf Messages')
    print('#')    
    print('##################################################')
    print('#')
    
    #Load the input image.
    if usecam == "False":
    	image = mp.Image.create_from_file("image.png")
    
    annotated_image, facemarkers = CalcMesh(image)
    
    # initialize ECAL message
    pub = EcalSetup()
    
    data2ecal =  PubFaceMesh(facemarkers)
    
    window_name = 'image'
    while ecal_core.ok():
        pub.send(data2ecal)
        time.sleep(0.5)
        cv2.imshow(window_name, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(500)
        
    # finalize eCAL API
    ecal_core.finalize()



