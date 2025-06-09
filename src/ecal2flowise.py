'''

Martin Hummel
2025/06/01
'''
import os
import sys
import time
import json
import threading
import requests
import ecal.core.core as ecal_core
from ecal.core.subscriber import ProtoSubscriber
import facedata_pb2
from dotenv import load_dotenv
#from queue import Queue


# Load configuration from .env file
load_dotenv()
FLOW_ID = os.getenv('FLOWID')
FLOWISE_API_URL = f"http://localhost:8000/api/v1/prediction/{FLOW_ID}"

FLASK_API_URL = "http://192.168.0.228:5000/api/interpret" # Substitute with the local ip address of your machine

# Lock to ensure only one message is processed at a time
task_lock = threading.Lock()

'''
frame_queue = Queue()


def llm_queue_worker():
    """
    Continuously process frames from the queue in order.
    """
    while True:
        frame_data = frame_queue.get()
        try:
            llm_worker(frame_data)
        except Exception as e:
            print(f"Worker error: {e}")
        frame_queue.task_done()


def callback(topic_name, msg, t):
    """
    eCAL subscriber callback: adds the frame to the processing queue.
    """
    try:
        frame_data = convert_protobuf_to_dict(msg)
        frame_queue.put(frame_data)
        print(f"📥 Frame queued. Queue size: {frame_queue.qsize()}")
    except Exception as e:
        print(f"Callback error: {e}")
'''

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
        flask_response = requests.post(FLASK_API_URL, json={"text": llm_text})

        if flask_response.status_code == 200:
            print("✅ Flask API Success:")
            print(flask_response.json().get("message"))
        else:
            print("❌ Flask API Error:")
            print("Status Code:", flask_response.status_code)
            print("Response:", flask_response.text)
        print('Output finished')
    else:
        print(f"Error {response.status_code}: {data.get('message')}")
        if data.get('stack'):
            print(data.get('stack'))


def llm_worker(frame_data):
    """
    Send frame data to Flowise and handle the response.
    """
    try:
        print("Sending payload to Flowise...")
        payload = {"question": frame_data}

        start_time = time.time()
        response = requests.post(FLOWISE_API_URL, json=payload)
        duration = time.time() - start_time

        print(f"\n⏱️ Flowise response time: {duration:.2f} seconds\n")

        # Attach timing to the response object (optional for downstream)
        response.elapsed_time = duration

        handle_llm_response(response)
    except Exception as e:
        print(f"Flowise request failed: {e}")
    finally:
        task_lock.release()


def callback(topic_name, msg, t):
    """
    eCAL subscriber callback: acquires lock, converts message, and starts LLM worker thread.
    """
    # Only process if no other task is running
    if task_lock.acquire(blocking=False):
        try:
            frame_data = convert_protobuf_to_dict(msg)
            threading.Thread(target=llm_worker, args=(frame_data,), daemon=True).start()
            # Start background thread to process the queue
            #threading.Thread(target=llm_queue_worker, daemon=True).start()
        except Exception as e:
            print(f"Callback error: {e}")
            task_lock.release()
    #else:
    #    print("LLM still processing previous message. Dropping new message.")


def main():
    print("Initializing eCAL...")
    ecal_core.initialize(sys.argv, "Camera")
    sub = ProtoSubscriber("FaceData", facedata_pb2.FaceData)
    sub.set_callback(callback)

    try:
        while ecal_core.ok():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        ecal_core.finalize()


if __name__ == "__main__":
    main()
