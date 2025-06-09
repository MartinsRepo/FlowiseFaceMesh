# visualize_zmq.py – Run this on the host machine
import cv2
import zmq
import numpy as np

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://localhost:5555")  # or use Docker IP
socket.setsockopt(zmq.SUBSCRIBE, b"")

while True:
    msg = socket.recv()
    npimg = np.frombuffer(msg, dtype=np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if frame is not None:
        cv2.imshow("Remote Webcam Stream", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()

