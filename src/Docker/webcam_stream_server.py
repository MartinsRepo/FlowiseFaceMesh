import cv2, time
import zmq
import numpy as np
from flask import Flask, Response, render_template_string, request, jsonify
from flask_cors import CORS
import threading
import ecal.core.core as ecal_core
from ecal.core.publisher import ProtoPublisher
import modeloutput_pb2
import textwrap

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# initialise Ecal
ecal_core.initialize([], "LLM Answer")
pub = ProtoPublisher("from Flask",modeloutput_pb2.OUT)

# ZeroMQ setup for video stream
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://localhost:5555")  # Connect to the source of your video stream
socket.setsockopt_string(zmq.SUBSCRIBE, "") # Subscribe to all topics (empty string)

# Global variables for video frame
latest_frame = None
frame_lock = threading.Lock() # To protect latest_frame from concurrent access

# GLOBAL VARIABLES FOR LLM TEXT OUTPUT AND ITS LOCK (THESE ARE THE ONES THAT WERE MISSING/INCORRECTLY PLACED)
latest_llm_text = ""
llm_last_updated_time = 0 # Initialize with a timestamp (e.g., 0 or time.time() * 1000)
llm_text_lock = threading.Lock() # THIS NEEDS TO BE DEFINED GLOBALLY

def zmq_receiver():
    """Receives frames from ZeroMQ and updates the latest_frame global variable."""
    global latest_frame
    print("ZeroMQ receiver thread started...")
    while True:
        try:
            msg = socket.recv()
            npimg = np.frombuffer(msg, dtype=np.uint8)
            frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

            if frame is not None:
                with frame_lock:
                    latest_frame = frame
            time.sleep(0.01) # Small delay to prevent busy-waiting
        except zmq.Again:
            continue # No message yet, try again
        except Exception as e:
            print(f"ZeroMQ receiver error: {e}")
            break
    socket.close()
    context.term()

def generate_frames():
    """Generates JPEG frames for Flask video streaming."""
    global latest_frame
    while True:
        with frame_lock:
            if latest_frame is None:
                # You might want to yield a blank/placeholder image here or wait longer
                time.sleep(0.1)
                continue

            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', latest_frame)
            if not ret:
                time.sleep(0.1) # Prevent busy loop on encoding failure
                continue

        frame_bytes = buffer.tobytes()

        # Yield the frame in multipart/x-mixed-replace format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03) # Adjust this for desired frame rate (e.g., ~30 fps)
        
def genECALdata(text):
    pub2ecal = modeloutput_pb2.OUT()
    pub2ecal.headline = "LLM Answer"
    pub2ecal.text = text
    pub.send(pub2ecal)
    

@app.route('/')
def index():
    """Renders the HTML page that displays the video stream and LLM text."""
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Remote Webcam & LLM Stream</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; margin-top: 20px; background-color: #f0f0f0; }
            h1, h2 { color: #333; }
            #video-container { margin-bottom: 20px; }
            img { border: 2px solid #ccc; max-width: 90%; height: auto; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
            #llm-output-container {
                margin: 20px auto;
                padding: 15px;
                border: 1px solid #ddd;
                border-radius: 8px;
                background-color: #fff;
                width: 80%;
                max-width: 800px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                text-align: left;
            }
            #llm-text {
                font-size: 1.1em;
                color: #555;
                white-space: pre-wrap; /* Preserves whitespace and line breaks */
                word-wrap: break-word; /* Breaks long words */
            }
            .timestamp {
                font-size: 0.8em;
                color: #888;
                margin-top: 5px;
                text-align: right;
            }
        </style>
    </head>
    <body>
        <h1>Live Webcam Stream from Docker</h1>
        <div id="video-container">
            <img src="/video_feed" alt="Live Stream">
        </div>

        <h2>LLM Interpretation</h2>
        <div id="llm-output-container">
            <p id="llm-text">Waiting for LLM output...</p>
            <div id="llm-timestamp" class="timestamp"></div>
        </div>

        <script>
            // Function to fetch LLM text and update the display
            function fetchLlmText() {
                fetch('/api/get_llm_text')
                    .then(response => response.json())
                    .then(data => {
                        const llmTextDiv = document.getElementById('llm-text');
                        const llmTimestampDiv = document.getElementById('llm-timestamp');
                        if (data.text) {
                            llmTextDiv.textContent = data.text;
                            llmTimestampDiv.textContent = `Last updated: ${new Date(data.timestamp).toLocaleTimeString()}`;
                        } else {
                            llmTextDiv.textContent = 'No LLM output yet or error fetching.';
                            llmTimestampDiv.textContent = '';
                        }
                    })
                    .catch(error => {
                        console.error('Error fetching LLM text:', error);
                        document.getElementById('llm-text').textContent = 'Error loading LLM text.';
                        document.getElementById('llm-timestamp').textContent = '';
                    });
            }

            // Fetch text initially and then every 2 seconds
            fetchLlmText(); // Fetch immediately on load
            setInterval(fetchLlmText, 2000); // Fetch every 2 seconds
        </script>
    </body>
    </html>
    """)

@app.route('/video_feed')
def video_feed():
    """Provides the video stream as a multipart response."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/interpret', methods=['POST'])
def interpret_text():
    """Receives LLM text from the Docker container."""
    global latest_llm_text # Declare global to modify the variable
    global llm_last_updated_time # Declare global for timestamp

    if not request.is_json:
        print(f"[{time.ctime()}] Received non-JSON request")
        return jsonify({"status": "error", "message": "Request must be JSON"}), 400

    data = request.get_json()
    llm_text = data.get('text')

    if llm_text:
        print(f"[{time.ctime()}] Received LLM text from Docker: {llm_text}")
        
        interpretation_text = data["text"]
        wrapper = textwrap.TextWrapper(width=80, subsequent_indent='  ')
        formatted = "\n".join(wrapper.fill(line) if not line.startswith("-") else wrapper.fill(line) for line in interpretation_text.splitlines())

        # send Ecal data
        genECALdata(formatted)
        
        with llm_text_lock: # Protect shared variable access
            latest_llm_text = llm_text # Store the latest text
            llm_last_updated_time = time.time() * 1000 # Store timestamp in milliseconds for JS
        return jsonify({"status": "success", "message": "Text received and processed on host"}), 200
    else:
        print(f"[{time.ctime()}] Received request with no 'text' field")
        return jsonify({"status": "error", "message": "No 'text' field in request body"}), 400

@app.route('/api/get_llm_text', methods=['GET'])
def get_llm_text():
    """Provides the latest LLM text to the web browser."""
    with llm_text_lock: # Protect shared variable access
        return jsonify({
            "text": latest_llm_text,
            "timestamp": llm_last_updated_time
        })

if __name__ == '__main__':
    # Initialize timestamp
    llm_last_updated_time = time.time() * 1000 # Milliseconds for JS Date object

    # Start the ZeroMQ receiver in a separate thread
    zmq_thread = threading.Thread(target=zmq_receiver, daemon=True)
    zmq_thread.start()

    # Start the Flask web server
    print("Flask web server starting on http://0.0.0.0:5000") # Ensure this matches your FLASK_API_URL in cam2flowise.py
    app.run(host='0.0.0.0', port=5000, debug=True)
    ecal_core.finalize()
