'''
This test module shows simplyfied the complete toolchain:
* Getting a (simulated) Ecal message as input to the flowise model
* running a chatGPT 4o chat model in flowise chained with a POST API chained
* running  a FLASK server to receive the chatGPT output from the flowise model

Martin Hummel
2025/06/01
'''
import requests
import json
import time
import facemesh_pb2

FLOW_ID = "43db1b2b-1ceb-44a4-a227-0ae90738fe47"  # Substitute with your Flow-ID
FLOWISE_API_URL = f"http://localhost:8000/api/v1/prediction/{FLOW_ID}"

FLASK_API_URL = "http://192.168.0.228:5000/api/interpret" # Substitute with the local ip address of your machine

# Static provided payload as landmarks
payload = {
    'question': {
        'message': 'Ecal2Flowise',
        'landmarks': [
            {'x': 418, 'y': 198, 'name': 'nose'},
            {'x': 426, 'y': 284, 'name': 'chin'},
            {'x': 370, 'y': 169, 'name': 'left eye'},
            {'x': 384, 'y': 151, 'name': 'upper lid left'},
            {'x': 390, 'y': 172, 'name': 'lower lid left'},
            {'x': 480, 'y': 157, 'name': 'right eye'},
            {'x': 457, 'y': 142, 'name': 'upper lid right'},
            {'x': 456, 'y': 165, 'name': 'lower lid right'},
            {'x': 391, 'y': 239, 'name': 'left mouth'},
            {'x': 460, 'y': 233, 'name': 'right mouth'},
            {'x': 422, 'y': 221, 'name': 'upper lip'},
            {'x': 379, 'y': 167, 'name': 'lower lip'}
        ]
    }
}

# sdditional Face Oval landmarks
face_oval_data = [
    {'name': 'oval_10', 'x': 415, 'y': 109},
    {'name': 'oval_338', 'x': 434, 'y': 107},
    {'name': 'oval_297', 'x': 453, 'y': 108},
    {'name': 'oval_332', 'x': 472, 'y': 112},
    {'name': 'oval_284', 'x': 487, 'y': 121},
    {'name': 'oval_251', 'x': 499, 'y': 133},
    {'name': 'oval_389', 'x': 507, 'y': 147},
    {'name': 'oval_356', 'x': 512, 'y': 166},
    {'name': 'oval_454', 'x': 514, 'y': 183},
    {'name': 'oval_323', 'x': 515, 'y': 200},
    {'name': 'oval_361', 'x': 514, 'y': 219},
    {'name': 'oval_288', 'x': 509, 'y': 238},
    {'name': 'oval_397', 'x': 501, 'y': 253},
    {'name': 'oval_365', 'x': 491, 'y': 264},
    {'name': 'oval_379', 'x': 479, 'y': 272},
    {'name': 'oval_378', 'x': 468, 'y': 277},
    {'name': 'oval_400', 'x': 455, 'y': 280},
    {'name': 'oval_377', 'x': 442, 'y': 283},
    {'name': 'oval_152', 'x': 426, 'y': 284},
    {'name': 'oval_148', 'x': 412, 'y': 285},
    {'name': 'oval_176', 'x': 402, 'y': 283},
    {'name': 'oval_149', 'x': 394, 'y': 281},
    {'name': 'oval_150', 'x': 386, 'y': 278},
    {'name': 'oval_136', 'x': 378, 'y': 272},
    {'name': 'oval_172', 'x': 373, 'y': 265},
    {'name': 'oval_58', 'x': 369, 'y': 252},
    {'name': 'oval_132', 'x': 366, 'y': 233},
    {'name': 'oval_93', 'x': 364, 'y': 216},
    {'name': 'oval_234', 'x': 363, 'y': 199},
    {'name': 'oval_127', 'x': 360, 'y': 183},
    {'name': 'oval_162', 'x': 359, 'y': 165},
    {'name': 'oval_21', 'x': 359, 'y': 151},
    {'name': 'oval_54', 'x': 363, 'y': 137},
    {'name': 'oval_103', 'x': 370, 'y': 126},
    {'name': 'oval_67', 'x': 383, 'y': 117},
    {'name': 'oval_109', 'x': 397, 'y': 112}
]


def generate_Ecal_sim_msg():

  # Create an instance of the FD message
  fd_message = facemesh_pb2.FD()

  # Populate the simple fields
  fd_message.filename = payload['question']['message']
  fd_message.framewidth = 640  # Assuming default values as not in payload
  fd_message.frameheight = 480 # Assuming default values as not in payload

  # Populate the repeated Landmarks field
  for lm_data in payload['question']['landmarks']:
      landmark = fd_message.packedlM.add()
      landmark.name = lm_data['name']
      landmark.x = lm_data['x']
      landmark.y = lm_data['y']

  # Populate the repeated FaceOval field
  for fo_data in face_oval_data:
      face_oval = fd_message.packedFO.add()
      face_oval.name = fo_data['name']
      face_oval.x = fo_data['x']
      face_oval.y = fo_data['y']

  # Serialize the message to a byte string
  ecal_payload_bytes = fd_message.SerializeToString()

  return fd_message


def convert_protobuf_to_dict(pb_msg):
    """
    Convert Facemesh protobuf message to a dict with filename and landmarks list.
    """
    frame_data = {'message': pb_msg.filename}
    landmarks = []
    for lm in pb_msg.packedlM:
        landmarks.append({'x': lm.x, 'y': lm.y, 'name': lm.name})
    frame_data['landmarks'] = landmarks
    return frame_data


ecalmsg = generate_Ecal_sim_msg()

frame_data = convert_protobuf_to_dict(ecalmsg)

response = requests.post(
    FLOWISE_API_URL,
    json={"question": frame_data} # Send the JSON string instead of the object
)

print("\n--- Answer from Flowise ---")

while True:
    try:
        response_json = response.json()
        if response.status_code == 200:
            print("Status Code:", response.status_code)
            print("Output from LLM:")
            llm_text = response_json.get("text")
            print(llm_text)
            print("Chat ID:", response_json.get("chatId"))
            print("Chat Message ID:", response_json.get("chatMessageId"))

            # Send to Flask
            print("\n--- Sending to Flask API ---")
            flask_response = requests.post(FLASK_API_URL, json={"text": llm_text})

            if flask_response.status_code == 200:
                print("✅ Flask API Success:")
                print(flask_response.json().get("message"))
            else:
                print("❌ Flask API Error:")
                print("Status Code:", flask_response.status_code)
                print("Response:", flask_response.text)
        else:
            print("Error trying to get an answer!")
            print("Status Code:", response.status_code)
            print("Error Message:", response_json.get("message"))
            print("Stack Trace:", response_json.get("stack"))

    except json.JSONDecodeError:
        print("Error: Not valid JSON answer fromFlowise!")
        print("Text:", response.text)

    print("--- Ready ---")
    time.sleep(0.5)

