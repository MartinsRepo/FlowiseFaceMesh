
import os, sys, time, json, requests
from threading import Timer
import ecal.core.core as ecal_core
from ecal.core.subscriber import ProtoSubscriber
from queue import Queue
from dotenv import load_dotenv

frame_queue = Queue(maxsize=1000)
return_queue = Queue(maxsize=1000)

import facemesh_pb2

# load configuration from .env file
load_dotenv() 
flowid = str(os.getenv('FLOWID'))

FLOW_ID = flowid  # Flowise Project ID
FLOWISE_API_URL = f"http://localhost:8000/api/v1/prediction/{FLOW_ID}"

def convert_protobuf_to_dict(pb_msg):
    frame_data = {'message': pb_msg.filename}
    landmarks_data = []
    for lm in pb_msg.packedlM:
        landmarks_data.append({
            'x': lm.x,
            'y': lm.y,
            'name': lm.name
        })
    frame_data['landmarks'] = landmarks_data
    return frame_data

def ecal_subscriber():
    print("Ecal Process starting")
    ecal_core.initialize(sys.argv, "Camera")
    sub = ProtoSubscriber("FaceCoords", facemesh_pb2.FD)

    def callback(topic_name, msg, time):
        try:
            frame_data = convert_protobuf_to_dict(msg)
            frame_queue.put(frame_data, block=False)
        except frame_queue.full:
            print("Queue full. Dropping frame.")
        except Exception as e:
            print(f"Error processing message: {e}")
            frame_queue.put({"error": str(e)})

    sub.set_callback(callback)
    return sub

def ecalrunner():
    if ecal_core.ok():
        frame_dict = frame_queue.get()
        payload = {"question": frame_dict}
        print("Payload being sent to Flowise:")
        print(json.dumps(payload, indent=2))

        response = requests.post(
            FLOWISE_API_URL,
            json={"question": payload}  # Send as object, not JSON string
        )
        return_queue.put(response)

    ecal_timer = Timer(0.5, ecalrunner).start()

def json2flowise():
    response = return_queue.get()
    try:
        response_json = response.json()
        if response.status_code == 200:
            print("Status Code:", response.status_code)
            print("Output from LLM:")
            print(response_json.get("text"))
            print("Chat ID:", response_json.get("chatId"))
            print("Chat Message ID:", response_json.get("chatMessageId"))
        else:
            print("Error trying to get an answer!")
            print("Status Code:", response.status_code)
            print("Error Message:", response_json.get("message"))
            print("Stack Trace:", response_json.get("stack"))
    except json.JSONDecodeError:
        print("Error: Not valid JSON answer from Flowise!")
        print("Text:", response.text)

    flowise_timer = Timer(0.5, json2flowise)
    flowise_timer.start()

if __name__ == "__main__":
    print('##################################################')
    print('#')
    print('# Receiving Ecal messages and bringing to json format')
    print('#')
    print('##################################################')
    print('#')

    sub = ecal_subscriber()
    ecalthread = Timer(0.5, ecalrunner).start()
    flowisethread = Timer(0.5, json2flowise).start()

    try:
        while ecal_core.ok():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        ecal_core.finalize()
