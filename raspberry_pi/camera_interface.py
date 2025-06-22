import cv2
from picamera2 import Picamera2
import numpy as np
from multiprocessing import shared_memory
import time

# Initialize the camera
picam2 = Picamera2(camera_num=0)
config = picam2.create_preview_configuration()
picam2.configure(config)
picam2.start()

# Frame dimensions
frame_shape = (300, 300, 3)  # Adjust according to your camera resolution
frame_size = np.prod(frame_shape) * np.dtype(np.uint8).itemsize

# Create shared memory block
shm = shared_memory.SharedMemory(create=True, size=frame_size, name="video_frame")

try:
    while True:
        frame = picam2.capture_array()  # Capture image as NumPy array
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV

##        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        resized_frame = cv2.resize(frame, (300, 300))  # Resize to match shared memory

        # Copy frame to shared memory
        shm.buf[:frame_size] = resized_frame.flatten().tobytes()

##        cv2.imshow("DSI Camera Preview", resized_frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

cv2.destroyAllWindows()
shm.close()
shm.unlink()  # Clean up shared memory
