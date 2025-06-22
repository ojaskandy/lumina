import cv2
import numpy as np
import speech_recognition as sr
import threading
import queue
import time
import lgpio
from multiprocessing import shared_memory
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter, run_inference

# ===== Servo Configuration =====
SERVO_PIN_X = 14              # Horizontal servo (BCM pin 14)
SERVO_PIN_Y = 15              # Vertical servo (BCM pin 15) - add second servo
PWM_FREQ = 50                 # 50 Hz signal (20ms period)
MIN_PULSE_US = 500            # 0.5 ms = 0°
MAX_PULSE_US = 2400           # 2.4 ms = ~180°
SAFE_MIN_ANGLE = 0            # Minimum safe angle
SAFE_MAX_ANGLE = 180          # Maximum safe angle

# Servo control parameters
SERVO_CENTER_X = 90           # Center position for X servo
SERVO_CENTER_Y = 90           # Center position for Y servo
SERVO_STEP_SIZE = 0.5         # Smaller steps for smoother movement
SERVO_RESPONSE_ZONE = 80      # Dead zone around center (pixels)
SERVO_MAX_STEP = 2.0          # Maximum step size for fast movement
SERVO_SMOOTHING = 0.85        # Servo position smoothing factor

# ===== Vision Configuration =====
MODEL_PATH = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
LABEL_PATH = 'coco_labels.txt'
SHM_NAME = 'video_frame'
SHM_FRAME_SHAPE = (300, 300, 3)
TOP_K = 10
THRESHOLD = 0.3
MIN_MATCH_THRESHOLD = 25
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

class ServoController:
    def __init__(self):
        # GPIO Setup
        self.h = lgpio.gpiochip_open(0)
        lgpio.gpio_claim_output(self.h, SERVO_PIN_X)
        lgpio.gpio_claim_output(self.h, SERVO_PIN_Y)
        
        # Servo positions
        self.current_angle_x = SERVO_CENTER_X
        self.current_angle_y = SERVO_CENTER_Y
        self.target_angle_x = SERVO_CENTER_X
        self.target_angle_y = SERVO_CENTER_Y
        
        # Smoothing variables
        self.smoothed_angle_x = float(SERVO_CENTER_X)
        self.smoothed_angle_y = float(SERVO_CENTER_Y)
        
        # Initialize servos to center
        self.set_servo_angle(SERVO_PIN_X, SERVO_CENTER_X)
        self.set_servo_angle(SERVO_PIN_Y, SERVO_CENTER_Y)
        
        print(f"Servos initialized at center position: X={SERVO_CENTER_X}°, Y={SERVO_CENTER_Y}°")

    def angle_to_duty(self, angle):
        """Convert angle to PWM duty cycle"""
        angle = max(SAFE_MIN_ANGLE, min(SAFE_MAX_ANGLE, angle))
        pulse_width = MIN_PULSE_US + (angle / 180.0) * (MAX_PULSE_US - MIN_PULSE_US)
        period_us = 1_000_000 / PWM_FREQ
        return (pulse_width / period_us) * 100

    def set_servo_angle(self, pin, angle):
        """Set servo to specific angle"""
        duty = self.angle_to_duty(angle)
        lgpio.tx_pwm(self.h, pin, PWM_FREQ, duty)

    def calculate_servo_movement(self, object_center, frame_center, bbox_info=None):
        """Calculate servo movement with enhanced stability using bbox info"""
        if frame_center is None:
            return
        
        frame_cx, frame_cy = frame_center
        obj_x, obj_y = object_center
        
        # Calculate pixel offset from center
        offset_x = obj_x - frame_cx
        offset_y = obj_y - frame_cy
        
        # Use bounding box information for stability correction
        if bbox_info:
            # Get bbox center for additional stability
            bbox_center_x = (bbox_info['x'] + bbox_info['w'] / 2)
            bbox_center_y = (bbox_info['y'] + bbox_info['h'] / 2)
            
            # Blend tracking dot position with bbox center (weighted average)
            blend_factor = 0.7  # Favor tracking dot but use bbox for stability
            obj_x = int(blend_factor * obj_x + (1 - blend_factor) * bbox_center_x)
            obj_y = int(blend_factor * obj_y + (1 - blend_factor) * bbox_center_y)
            
            # Recalculate offsets with blended position
            offset_x = obj_x - frame_cx
            offset_y = obj_y - frame_cy
        
        # Apply dead zone - don't move servo if object is close to center
        if abs(offset_x) < SERVO_RESPONSE_ZONE and abs(offset_y) < SERVO_RESPONSE_ZONE:
            return  # Stay at current position
        
        # Calculate movement based on offset (with scaling)
        # Larger offsets = faster movement, smaller offsets = slower movement
        movement_scale_x = min(abs(offset_x) / 200.0, 1.0)  # Scale movement based on distance
        movement_scale_y = min(abs(offset_y) / 200.0, 1.0)
        
        # Calculate step size (adaptive based on distance)
        step_x = SERVO_STEP_SIZE + (SERVO_MAX_STEP - SERVO_STEP_SIZE) * movement_scale_x
        step_y = SERVO_STEP_SIZE + (SERVO_MAX_STEP - SERVO_STEP_SIZE) * movement_scale_y
        
        # Determine movement direction and calculate target angles
        if offset_x > SERVO_RESPONSE_ZONE:  # Object is to the right, move servo right
            self.target_angle_x = min(SAFE_MAX_ANGLE, self.target_angle_x + step_x)
        elif offset_x < -SERVO_RESPONSE_ZONE:  # Object is to the left, move servo left
            self.target_angle_x = max(SAFE_MIN_ANGLE, self.target_angle_x - step_x)
        
        if offset_y > SERVO_RESPONSE_ZONE:  # Object is down, move servo down
            self.target_angle_y = min(SAFE_MAX_ANGLE, self.target_angle_y + step_y)
        elif offset_y < -SERVO_RESPONSE_ZONE:  # Object is up, move servo up
            self.target_angle_y = max(SAFE_MIN_ANGLE, self.target_angle_y - step_y)

    def smooth_servo_movement(self):
        """Apply smoothing to servo movement for fluid motion"""
        # Smooth the angles
        self.smoothed_angle_x = (SERVO_SMOOTHING * self.smoothed_angle_x + 
                                (1 - SERVO_SMOOTHING) * self.target_angle_x)
        self.smoothed_angle_y = (SERVO_SMOOTHING * self.smoothed_angle_y + 
                                (1 - SERVO_SMOOTHING) * self.target_angle_y)
        
        # Update current angles
        self.current_angle_x = self.smoothed_angle_x
        self.current_angle_y = self.smoothed_angle_y
        
        # Move servos
        self.set_servo_angle(SERVO_PIN_X, self.current_angle_x)
        self.set_servo_angle(SERVO_PIN_Y, self.current_angle_y)

    def get_servo_status(self):
        """Get current servo status for display"""
        return {
            'x_angle': self.current_angle_x,
            'y_angle': self.current_angle_y,
            'x_target': self.target_angle_x,
            'y_target': self.target_angle_y
        }

    def reset_to_center(self):
        """Reset servos to center position"""
        self.target_angle_x = SERVO_CENTER_X
        self.target_angle_y = SERVO_CENTER_Y
        self.smoothed_angle_x = float(SERVO_CENTER_X)
        self.smoothed_angle_y = float(SERVO_CENTER_Y)
        self.set_servo_angle(SERVO_PIN_X, SERVO_CENTER_X)
        self.set_servo_angle(SERVO_PIN_Y, SERVO_CENTER_Y)
        print("Servos reset to center position")

    def cleanup(self):
        """Clean up GPIO resources"""
        lgpio.tx_pwm(self.h, SERVO_PIN_X, 0, 0)
        lgpio.tx_pwm(self.h, SERVO_PIN_Y, 0, 0)
        lgpio.gpiochip_close(self.h)
        print("Servo GPIO cleaned up")

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
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=4)
                    
                try:
                    command = self.recognizer.recognize_google(audio).lower()
                    print(f"Heard: '{command}'")
                    
                    words = command.split()
                    detected_objects = []
                    
                    for word in words:
                        if word in label_to_id:
                            detected_objects.append(word)
                    
                    if detected_objects:
                        target_object = detected_objects[0]
                        print(f"Command received: Track '{target_object}'")
                        self.command_queue.put(target_object)
                    else:
                        for word in words:
                            if word in ['stop', 'quit', 'exit']:
                                self.command_queue.put('STOP')
                                return
                            elif word in ['reset', 'restart', 'new', 'center']:
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

class EnhancedSmoothTracker:
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
        
        # Position tracking
        self.current_center = None
        self.previous_center = None
        self.smoothing_factor = 0.7
        self.frame_center = None
        
        # Bounding box tracking for stability
        self.current_bbox = None
        self.bbox_history = []
        self.bbox_smoothing_factor = 0.6

    def set_frame_center(self, frame_shape):
        """Set frame center for direction calculation"""
        h, w = frame_shape[:2]
        self.frame_center = (w // 2, h // 2)

    def extract_features_with_surrounding(self, bbox, frame):
        """Extract features from bbox and surrounding area"""
        x, y, w, h = bbox
        frame_h, frame_w = frame.shape[:2]
        
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
        """Initialize tracker with bounding box"""
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
        
        # Initialize positions
        self.current_center = (x + w//2, y + h//2)
        self.previous_center = self.current_center
        self.current_bbox = {'x': x, 'y': y, 'w': w, 'h': h}
        self.tracking_active = True
        
        print(f"Enhanced tracker initialized - ID: {self.tracker_id}, Label: {label}")
        return True

    def smooth_position(self, new_center):
        """Apply smoothing to position"""
        if self.current_center is None:
            return new_center
        
        smooth_x = int(self.smoothing_factor * self.current_center[0] + 
                      (1 - self.smoothing_factor) * new_center[0])
        smooth_y = int(self.smoothing_factor * self.current_center[1] + 
                      (1 - self.smoothing_factor) * new_center[1])
        
        return (smooth_x, smooth_y)

    def smooth_bbox(self, new_bbox):
        """Apply smoothing to bounding box"""
        if self.current_bbox is None:
            return new_bbox
        
        smooth_x = int(self.bbox_smoothing_factor * self.current_bbox['x'] + 
                      (1 - self.bbox_smoothing_factor) * new_bbox['x'])
        smooth_y = int(self.bbox_smoothing_factor * self.current_bbox['y'] + 
                      (1 - self.bbox_smoothing_factor) * new_bbox['y'])
        smooth_w = int(self.bbox_smoothing_factor * self.current_bbox['w'] + 
                      (1 - self.bbox_smoothing_factor) * new_bbox['w'])
        smooth_h = int(self.bbox_smoothing_factor * self.current_bbox['h'] + 
                      (1 - self.bbox_smoothing_factor) * new_bbox['h'])
        
        return {'x': smooth_x, 'y': smooth_y, 'w': smooth_w, 'h': smooth_h}

    def track_using_stored_features(self, frame):
        """Enhanced tracking with bbox information"""
        if not self.tracking_active or self.tracker is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, desc = self.sift.detectAndCompute(gray, None)

        if desc is None or len(desc) < 2:
            if self.current_center and self.current_bbox:
                return {
                    'center': self.current_center,
                    'bbox': self.current_bbox,
                    'tracking_quality': 'poor'
                }
            return None

        try:
            matches = self.flann.knnMatch(self.tracker['template_desc'], desc, k=2)
            
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

            if len(good_matches) >= self.min_matches:
                src_pts = np.float32([self.tracker['template_kp'][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                if H is not None:
                    # Transform original bbox
                    x, y, w, h = self.tracker['original_bbox']
                    corners = np.float32([[x, y], [x+w, y], [x+w, y+h], [x, y+h]]).reshape(-1, 1, 2)
                    transformed_corners = cv2.perspectiveTransform(corners, H)
                    
                    # Calculate new center and bbox
                    new_center = np.mean(transformed_corners, axis=0).astype(int)[0]
                    
                    # Calculate bounding box from corners
                    x_coords = transformed_corners[:, 0, 0]
                    y_coords = transformed_corners[:, 0, 1]
                    new_bbox = {
                        'x': int(np.min(x_coords)),
                        'y': int(np.min(y_coords)),
                        'w': int(np.max(x_coords) - np.min(x_coords)),
                        'h': int(np.max(y_coords) - np.min(y_coords))
                    }
                    
                    # Apply smoothing
                    self.previous_center = self.current_center
                    self.current_center = self.smooth_position(tuple(new_center))
                    self.current_bbox = self.smooth_bbox(new_bbox)
                    
                    return {
                        'center': self.current_center,
                        'bbox': self.current_bbox,
                        'corners': transformed_corners,
                        'matches': np.sum(mask) if mask is not None else len(good_matches),
                        'tracking_quality': 'good'
                    }
            
            # Poor tracking - return smoothed last position
            if self.current_center and self.current_bbox:
                return {
                    'center': self.current_center,
                    'bbox': self.current_bbox,
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
        self.current_bbox = None

    def draw_enhanced_tracking(self, frame, result):
        """Draw enhanced tracking visualization with servo info"""
        if result:
            # Draw blue tracking dot (main target for servo)
            cv2.circle(frame, result['center'], 10, (255, 0, 0), -1)  # Blue dot
            cv2.circle(frame, result['center'], 12, (0, 255, 255), 2)  # Yellow outline
            
            # Draw bounding box
            if 'bbox' in result:
                bbox = result['bbox']
                color = (0, 255, 0) if result['tracking_quality'] == 'good' else (0, 255, 255)
                cv2.rectangle(frame, (bbox['x'], bbox['y']), 
                            (bbox['x'] + bbox['w'], bbox['y'] + bbox['h']), color, 2)
            
            # Draw crosshair at frame center
            if self.frame_center:
                cx, cy = self.frame_center
                cv2.line(frame, (cx-30, cy), (cx+30, cy), (128, 128, 128), 2)
                cv2.line(frame, (cx, cy-30), (cx, cy+30), (128, 128, 128), 2)
                cv2.circle(frame, (cx, cy), 5, (128, 128, 128), -1)
                
                # Draw response zone
                cv2.circle(frame, (cx, cy), SERVO_RESPONSE_ZONE, (64, 64, 64), 1)
        
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

    # Initialize enhanced tracker and servo controller
    tracker = EnhancedSmoothTracker()
    servo_controller = ServoController()

    # Initialize voice command system
    command_queue = queue.Queue()
    voice_listener = VoiceCommandListener(command_queue)
    
    # Start voice recognition in separate thread
    voice_thread = threading.Thread(target=voice_listener.listen_for_commands)
    voice_thread.daemon = True
    voice_thread.start()

    print("=== Enhanced Servo Object Tracking System ===")
    print("✓ Smooth servo movement with dual-axis control")
    print("✓ Blue dot tracking for precise servo following")
    print("✓ Bounding box stabilization")
    print("✓ Adaptive response zones and smoothing")
    print("Say any COCO object name to start tracking")
    print("Say 'reset' or 'center' to reset servos to center")

    # System state
    target_object = None
    detection_phase = False
    waiting_for_command = True
    last_servo_update = time.time()

    try:
        while True:
            frame = frame_np.copy()
            current_time = time.time()
            
            # Check for voice commands
            if not command_queue.empty():
                command = command_queue.get()
                
                if command == 'STOP':
                    break
                elif command == 'RESET':
                    tracker.reset_tracker()
                    servo_controller.reset_to_center()
                    target_object = None
                    detection_phase = False
                    waiting_for_command = True
                    print("System reset. Servos centered. Waiting for new command...")
                elif command in label_to_id:
                    target_object = command
                    detection_phase = True
                    waiting_for_command = False
                    tracker.reset_tracker()
                    print(f"Voice command received: Tracking '{target_object}' with servo control")
            
            # PHASE 0: Waiting for voice command
            if waiting_for_command:
                cv2.putText(frame, "Waiting for voice command...", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, "Say object name or 'reset' to center servos", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show servo status
                servo_status = servo_controller.get_servo_status()
                cv2.putText(frame, f"Servo X: {servo_status['x_angle']:.1f}° Y: {servo_status['y_angle']:.1f}°", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # PHASE 1: Detection
            elif detection_phase and not tracker.tracking_active and target_object:
                rgb_resized = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                run_inference(interpreter, rgb_resized.tobytes())
                objs = get_objects(interpreter, THRESHOLD)[:TOP_K]
                
                height, width, _ = frame.shape
                scale_x, scale_y = width / inference_size[0], height / inference_size[1]

                largest_obj = find_largest_target_object(objs, labels, target_object, scale_x, scale_y)
                
                if largest_obj:
                    bbox = largest_obj['bbox']
                    x0, y0, x1, y1 = int(bbox.xmin), int(bbox.ymin), int(bbox.xmax), int(bbox.ymax)
                    
                    tracker_bbox = (x0, y0, x1 - x0, y1 - y0)
                    
                    success = tracker.initialize_tracker(tracker_bbox, frame, largest_obj['label'])
                    if success:
                        detection_phase = False
                        print(f"Object '{target_object}' detected! Starting servo tracking...")
                
                cv2.putText(frame, f"Detecting {target_object}...", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # PHASE 2: Enhanced Tracking with Servo Control
            elif tracker.tracking_active:
                # Track object
                result = tracker.track_using_stored_features(frame)
                
                if result:
                    # Draw tracking visualization
                    frame = tracker.draw_enhanced_tracking(frame, result)
                    
                    # Update servo position based on blue dot position
                    if tracker.frame_center and 'center' in result and 'bbox' in result:
                        servo_controller.calculate_servo_movement(
                            result['center'], 
                            tracker.frame_center, 
                            result['bbox']
                        )
                    
                    # Apply smooth servo movement (update at ~20Hz for smooth motion)
                    if current_time - last_servo_update > 0.05:  # 50ms = 20Hz
                        servo_controller.smooth_servo_movement()
                        last_servo_update = current_time
                    
                    # Display tracking and servo information
                    servo_status = servo_controller.get_servo_status()
                    
                    # Show servo angles
                    cv2.putText(frame, f"Servo X: {servo_status['x_angle']:.1f}° (→{servo_status['x_target']:.1f}°)", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(frame, f"Servo Y: {servo_status['y_angle']:.1f}° (→{servo_status['y_target']:.1f}°)", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Show object position relative to center
                    if tracker.frame_center:
                        offset_x = result['center'][0] - tracker.frame_center[0]
                        offset_y = result['center'][1] - tracker.frame_center[1]
                        cv2.putText(frame, f"Object Offset: X:{offset_x:+4d} Y:{offset_y:+4d}", 
                                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    
                    # Show tracking quality
                    quality_color = (0, 255, 0) if result['tracking_quality'] == 'good' else (0, 255, 255)
                    cv2.putText(frame, f"Tracking: {result['tracking_quality'].upper()}", 
                               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, quality_color, 1)
                    
                    # Show servo response zone status
                    if tracker.frame_center:
                        offset_x = result['center'][0] - tracker.frame_center[0]
                        offset_y = result['center'][1] - tracker.frame_center[1]
                        
                        if abs(offset_x) < SERVO_RESPONSE_ZONE and abs(offset_y) < SERVO_RESPONSE_ZONE:
                            cv2.putText(frame, "SERVO: LOCKED ON TARGET", (10, 150), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        else:
                            direction = ""
                            if abs(offset_x) > SERVO_RESPONSE_ZONE:
                                direction += "LEFT " if offset_x < 0 else "RIGHT "
                            if abs(offset_y) > SERVO_RESPONSE_ZONE:
                                direction += "UP" if offset_y < 0 else "DOWN"
                            cv2.putText(frame, f"SERVO: MOVING {direction.strip()}", (10, 150), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            # Always show servo control instructions
            cv2.putText(frame, "Follow the BLUE DOT | Say 'reset' to center servos", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Display frame
            cv2.imshow('Enhanced Servo Object Tracking', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):  # Manual reset with 'r' key
                tracker.reset_tracker()
                servo_controller.reset_to_center()
                target_object = None
                detection_phase = False
                waiting_for_command = True
                print("Manual reset: Servos centered, waiting for new command...")

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Cleanup
        voice_listener.stop()
        servo_controller.cleanup()
        cv2.destroyAllWindows()
        shm.close()
        print("Cleanup completed - servos stopped, GPIO released")

if __name__ == '__main__':
    main()
