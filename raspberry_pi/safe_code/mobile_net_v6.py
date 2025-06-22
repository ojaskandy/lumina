import cv2
import numpy as np
import speech_recognition as sr
import threading
import queue
import time
from multiprocessing import shared_memory
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter, run_inference

# Configuration
MODEL_PATH = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
LABEL_PATH = 'coco_labels.txt'
SHM_NAME = 'video_frame'
SHM_FRAME_SHAPE = (300, 300, 3)
TOP_K = 10
THRESHOLD = 0.3
MIN_MATCH_THRESHOLD = 25  # Lower for smoother tracking
EXPAND_PIXELS = 40

# COCO labels with their class IDs
coco_labels = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus", 6: "train", 7: "truck", 8: "boat",
    9: "traffic light", 10: "fire hydrant", 12: "stop sign", 13: "parking meter", 14: "bench", 15: "bird", 16: "cat",
    17: "dog", 18: "horse", 19: "sheep", 20: "cow", 21: "elephant", 22: "bear", 23: "zebra", 24: "giraffe",
    26: "backpack", 27: "umbrella", 30: "handbag", 31: "tie", 32: "suitcase", 33: "frisbee", 34: "skis",
    35: "snowboard", 36: "sports ball", 37: "kite", 38: "baseball bat", 39: "baseball glove", 40: "skateboard",
    41: "surfboard", 42: "tennis racket", 43: "bottle", 45: "wine glass", 46: "cup", 47: "fork", 48: "knife",
    49: "spoon", 50: "bowl", 51: "banana", 52: "apple", 53: "sandwich", 54: "orange", 55: "broccoli", 56: "carrot",
    57: "hot dog", 58: "pizza", 59: "donut", 60: "cake", 61: "chair", 62: "couch", 63: "potted plant", 64: "bed",
    66: "dining table", 69: "toilet", 71: "tv", 72: "laptop", 73: "mouse", 74: "remote", 75: "keyboard",
    76: "cell phone", 77: "microwave", 78: "oven", 79: "toaster", 80: "sink", 81: "refrigerator", 83: "book",
    84: "clock", 85: "vase", 86: "scissors", 87: "teddy bear", 88: "hair drier", 89: "toothbrush"
}

# Convert label list to lowercase label → ID map for fast lookup
label_to_id = {v.lower(): k for k, v in coco_labels.items()}

class VoiceCommandListener:
    def __init__(self, command_queue):
        self.command_queue = command_queue
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.listening = True
        
        # Calibrate microphone
        print("Calibrating microphone for ambient noise...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
        print("Microphone calibrated. Ready to listen for commands.")
    
    def listen_for_commands(self):
        """Main listening loop running in separate thread"""
        print("Voice commands active. Say any COCO object name to start tracking.")
        print("Available objects: person, car, chair, bottle, laptop, etc.")
        
        while self.listening:
            try:
                with self.microphone as source:
                    # Listen for audio with timeout
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=4)
                    
                try:
                    # Recognize speech
                    command = self.recognizer.recognize_google(audio).lower()
                    print(f"Heard: '{command}'")
                    
                    # Parse for COCO object names
                    words = command.split()
                    detected_objects = []
                    
                    for word in words:
                        if word in label_to_id:
                            detected_objects.append(word)
                    
                    # Send first detected object to tracking system
                    if detected_objects:
                        target_object = detected_objects[0]
                        print(f"Command received: Track '{target_object}'")
                        self.command_queue.put(target_object)
                    else:
                        # Check for common variations
                        for word in words:
                            if word in ['stop', 'quit', 'exit']:
                                self.command_queue.put('STOP')
                                return
                            elif word in ['reset', 'restart', 'new']:
                                self.command_queue.put('RESET')
                
                except sr.UnknownValueError:
                    pass
                except sr.RequestError as e:
                    print(f"Speech recognition error: {e}")
                    time.sleep(1)
                    
            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                print(f"Audio error: {e}")
                time.sleep(1)
    
    def stop(self):
        self.listening = False

class SimpleSmoothTracker:
    def __init__(self):
        # SIFT setup
        self.sift = cv2.SIFT_create(nfeatures=2000)
        
        # FLANN matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Tracking state
        self.tracker = None
        self.tracking_active = False
        self.tracker_id = 1
        self.min_matches = MIN_MATCH_THRESHOLD
        
        # Smoothing and position tracking
        self.current_center = None
        self.previous_center = None
        self.smoothing_factor = 0.7  # For smooth movement
        self.frame_center = None  # Frame center for direction calculation
        
        # Direction tracking for servo control
        self.direction_x = "CENTER"  # LEFT, CENTER, RIGHT
        self.direction_y = "CENTER"  # UP, CENTER, DOWN
        self.position_history = []  # For movement prediction

    def set_frame_center(self, frame_shape):
        """Set frame center for direction calculation"""
        h, w = frame_shape[:2]
        self.frame_center = (w // 2, h // 2)

    def extract_features_with_surrounding(self, bbox, frame):
        """Extract features from bbox and surrounding area"""
        x, y, w, h = bbox
        frame_h, frame_w = frame.shape[:2]
        
        # Calculate expanded region
        expand_x = max(EXPAND_PIXELS, w // 4)
        expand_y = max(EXPAND_PIXELS, h // 4)
        
        x_exp = max(0, x - expand_x)
        y_exp = max(0, y - expand_y)
        w_exp = min(frame_w - x_exp, w + 2 * expand_x)
        h_exp = min(frame_h - y_exp, h + 2 * expand_y)
        
        expanded_roi = frame[y_exp:y_exp+h_exp, x_exp:x_exp+w_exp].copy()
        
        if expanded_roi.size == 0:
            return None, None, None
            
        gray = cv2.cvtColor(expanded_roi, cv2.COLOR_BGR2GRAY)
        kp, desc = self.sift.detectAndCompute(gray, None)
        
        if desc is None or len(desc) < self.min_matches:
            return None, None, None
            
        # Adjust keypoint coordinates
        adjusted_kp = []
        for keypoint in kp:
            new_kp = cv2.KeyPoint(
                keypoint.pt[0] + x_exp,
                keypoint.pt[1] + y_exp,
                keypoint.size,
                keypoint.angle,
                keypoint.response,
                keypoint.octave,
                keypoint.class_id
            )
            adjusted_kp.append(new_kp)
        
        return adjusted_kp, desc, expanded_roi

    def initialize_tracker(self, bbox, frame, label):
        """Initialize tracker"""
        x, y, w, h = bbox
        
        # Ensure bbox is within frame bounds
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)

        if w < 20 or h < 20:
            return False

        # Set frame center
        self.set_frame_center(frame.shape)

        # Extract features
        kp, desc, template_img = self.extract_features_with_surrounding((x, y, w, h), frame)
        
        if kp is None or desc is None:
            return False

        # Store tracker data
        self.tracker = {
            'id': self.tracker_id,
            'original_bbox': (x, y, w, h),
            'template_kp': kp,
            'template_desc': desc,
            'label': label
        }
        
        # Initialize position
        self.current_center = (x + w//2, y + h//2)
        self.previous_center = self.current_center
        self.tracking_active = True
        
        print(f"Tracker initialized - ID: {self.tracker_id}, Label: {label}")
        return True

    def smooth_position(self, new_center):
        """Apply smoothing to position for stable tracking"""
        if self.current_center is None:
            return new_center
        
        # Smooth the position
        smooth_x = int(self.smoothing_factor * self.current_center[0] + 
                      (1 - self.smoothing_factor) * new_center[0])
        smooth_y = int(self.smoothing_factor * self.current_center[1] + 
                      (1 - self.smoothing_factor) * new_center[1])
        
        return (smooth_x, smooth_y)

    def calculate_direction(self, center):
        """Calculate object direction relative to frame center for servo control"""
        if self.frame_center is None:
            return
        
        frame_cx, frame_cy = self.frame_center
        obj_x, obj_y = center
        
        # Calculate horizontal direction
        x_diff = obj_x - frame_cx
        if abs(x_diff) < 50:  # Dead zone
            self.direction_x = "CENTER"
        elif x_diff > 0:
            self.direction_x = "RIGHT"
        else:
            self.direction_x = "LEFT"
        
        # Calculate vertical direction
        y_diff = obj_y - frame_cy
        if abs(y_diff) < 50:  # Dead zone
            self.direction_y = "CENTER"
        elif y_diff > 0:
            self.direction_y = "DOWN"
        else:
            self.direction_y = "UP"

    def track_using_stored_features(self, frame):
        """Track object with smooth position updates"""
        if not self.tracking_active or self.tracker is None:
            return None

        # Extract features from current frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, desc = self.sift.detectAndCompute(gray, None)

        if desc is None or len(desc) < 2:
            # Return last known position for continuity
            if self.current_center:
                return {
                    'center': self.current_center,
                    'direction_x': self.direction_x,
                    'direction_y': self.direction_y,
                    'tracking_quality': 'poor'
                }
            return None

        try:
            # Match features
            matches = self.flann.knnMatch(self.tracker['template_desc'], desc, k=2)
            
            # Apply ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

            if len(good_matches) >= self.min_matches:
                # Get matched points
                src_pts = np.float32([self.tracker['template_kp'][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                # Find homography
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                if H is not None:
                    # Transform original bbox
                    x, y, w, h = self.tracker['original_bbox']
                    corners = np.float32([[x, y], [x+w, y], [x+w, y+h], [x, y+h]]).reshape(-1, 1, 2)
                    transformed_corners = cv2.perspectiveTransform(corners, H)
                    
                    # Calculate new center
                    new_center = np.mean(transformed_corners, axis=0).astype(int)[0]
                    
                    # Apply smoothing
                    self.previous_center = self.current_center
                    self.current_center = self.smooth_position(tuple(new_center))
                    
                    # Calculate direction for servo control
                    self.calculate_direction(self.current_center)
                    
                    return {
                        'center': self.current_center,
                        'corners': transformed_corners,
                        'direction_x': self.direction_x,
                        'direction_y': self.direction_y,
                        'matches': np.sum(mask) if mask is not None else len(good_matches),
                        'tracking_quality': 'good'
                    }
            
            # Poor tracking - return smoothed last position
            if self.current_center:
                return {
                    'center': self.current_center,
                    'direction_x': self.direction_x,
                    'direction_y': self.direction_y,
                    'tracking_quality': 'poor'
                }
                    
        except Exception as e:
            print(f"Tracking error: {e}")
        
        return None

    def reset_tracker(self):
        """Reset tracker"""
        self.tracker = None
        self.tracking_active = False
        self.tracker_id += 1
        self.current_center = None
        self.previous_center = None
        self.direction_x = "CENTER"
        self.direction_y = "CENTER"

    def draw_simple_tracking(self, frame, result):
        """Draw simple tracking visualization"""
        if result:
            # Draw center dot (always visible)
            color = (0, 255, 0) if result['tracking_quality'] == 'good' else (0, 255, 255)
            cv2.circle(frame, result['center'], 8, color, -1)
            
            # Draw simple bounding box if available
            if 'corners' in result:
                corners = np.int32(result['corners']).reshape((-1, 1, 2))
                cv2.polylines(frame, [corners], True, color, 2)
            
            # Draw crosshair at frame center for reference
            if self.frame_center:
                cx, cy = self.frame_center
                cv2.line(frame, (cx-20, cy), (cx+20, cy), (128, 128, 128), 1)
                cv2.line(frame, (cx, cy-20), (cx, cy+20), (128, 128, 128), 1)
            
            # Simple direction text for servo control
            direction_text = f"X:{result['direction_x']} Y:{result['direction_y']}"
            cv2.putText(frame, direction_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Position for servo control (you can use this for servo movement)
            if self.frame_center:
                offset_x = result['center'][0] - self.frame_center[0]
                offset_y = result['center'][1] - self.frame_center[1]
                cv2.putText(frame, f"Offset: X:{offset_x} Y:{offset_y}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame

def find_largest_target_object(objects, labels, target_object, scale_x, scale_y):
    """Find the largest bounding box of target object"""
    largest_obj = None
    largest_area = 0
    
    for obj in objects:
        label = labels.get(obj.id, '')
        if label != target_object:
            continue
            
        bbox = obj.bbox.scale(scale_x, scale_y)
        area = (bbox.xmax - bbox.xmin) * (bbox.ymax - bbox.ymin)
        
        if area > largest_area:
            largest_area = area
            largest_obj = {
                'bbox': bbox,
                'label': label,
                'area': area
            }
    
    return largest_obj

def main():
    # Initialize MobileNet interpreter
    interpreter = make_interpreter(MODEL_PATH)
    interpreter.allocate_tensors()
    labels = read_label_file(LABEL_PATH)
    inference_size = input_size(interpreter)

    # Initialize shared memory
    shm = shared_memory.SharedMemory(name=SHM_NAME)
    frame_np = np.ndarray(SHM_FRAME_SHAPE, dtype=np.uint8, buffer=shm.buf)

    # Initialize simple smooth tracker
    tracker = SimpleSmoothTracker()

    # Initialize voice command system
    command_queue = queue.Queue()
    voice_listener = VoiceCommandListener(command_queue)
    
    # Start voice recognition in separate thread
    voice_thread = threading.Thread(target=voice_listener.listen_for_commands)
    voice_thread.daemon = True
    voice_thread.start()

    print("=== Simple Smooth Object Tracker for Servo Control ===")
    print("✓ Smooth position tracking")
    print("✓ Direction detection (LEFT/RIGHT/CENTER, UP/DOWN/CENTER)")
    print("✓ Always visible tracking dot")
    print("Say any COCO object name to start tracking")
    print("Waiting for voice commands...")

    # System state
    target_object = None
    detection_phase = False
    waiting_for_command = True

    try:
        while True:
            frame = frame_np.copy()
            
            # Check for voice commands
            if not command_queue.empty():
                command = command_queue.get()
                
                if command == 'STOP':
                    break
                elif command == 'RESET':
                    tracker.reset_tracker()
                    target_object = None
                    detection_phase = False
                    waiting_for_command = True
                    print("System reset. Waiting for new voice command...")
                elif command in label_to_id:
                    target_object = command
                    detection_phase = True
                    waiting_for_command = False
                    tracker.reset_tracker()
                    print(f"Voice command received: Tracking '{target_object}'")
            
            # PHASE 0: Waiting for voice command
            if waiting_for_command:
                cv2.putText(frame, "Waiting for voice command...", (10, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, "Say object name: person, chair, car, etc.", (10, 130), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # PHASE 1: Detection
            elif detection_phase and not tracker.tracking_active and target_object:
                rgb_resized = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Run MobileNet inference
                run_inference(interpreter, rgb_resized.tobytes())
                objs = get_objects(interpreter, THRESHOLD)[:TOP_K]
                
                height, width, _ = frame.shape
                scale_x, scale_y = width / inference_size[0], height / inference_size[1]

                # Find the largest target object
                largest_obj = find_largest_target_object(objs, labels, target_object, scale_x, scale_y)
                
                if largest_obj:
                    bbox = largest_obj['bbox']
                    x0, y0, x1, y1 = int(bbox.xmin), int(bbox.ymin), int(bbox.xmax), int(bbox.ymax)
                    
                    # Convert to (x, y, w, h) format
                    tracker_bbox = (x0, y0, x1 - x0, y1 - y0)
                    
                    # Initialize tracker
                    success = tracker.initialize_tracker(tracker_bbox, frame, largest_obj['label'])
                    if success:
                        detection_phase = False
                        print(f"Object '{target_object}' detected! Starting smooth tracking...")
                
                # Show detection status
                cv2.putText(frame, f"Detecting {target_object}...", (10, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # PHASE 2: Simple Smooth Tracking
            elif tracker.tracking_active:
                # Track object
                result = tracker.track_using_stored_features(frame)
                
                if result:
                    frame = tracker.draw_simple_tracking(frame, result)
                    
                    # Print servo control info (you can use this for actual servo control)
                    if result['direction_x'] != "CENTER" or result['direction_y'] != "CENTER":
                        print(f"Servo Control: X={result['direction_x']}, Y={result['direction_y']}")

            # Display frame
            cv2.imshow('Simple Smooth Object Tracker', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Cleanup
        voice_listener.stop()
        cv2.destroyAllWindows()
        shm.close()
        print("Cleanup completed")

if __name__ == '__main__':
    main()
