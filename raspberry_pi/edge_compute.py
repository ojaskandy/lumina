import cv2
import numpy as np
import speech_recognition as sr
import threading
import queue
import time
import requests
import json
import pygame
import os
import asyncio
from gpiozero import Servo, Button
from time import sleep
from multiprocessing import shared_memory
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.edgetpu import make_interpreter, run_inference
from io import BytesIO
from PIL import Image
from lmnt.api import Speech
import dotenv

# Load environment variables for LMNT API
dotenv.load_dotenv()

# ===== GPIO Configuration =====
GPIO_PIN = 14  # Servo pin
MIN_PW = 0.0005
MAX_PW = 0.0024

# Tactile switch pins - Three switches
TRACKING_PIN = 17      # Start microphone and tracking
RESET_PIN = 27         # Reset system
GEMINI_PIN = 22        # Capture image and call Gemini API

# Create servo and button instances
servo = Servo(GPIO_PIN, min_pulse_width=MIN_PW, max_pulse_width=MAX_PW)
btn_tracking = Button(TRACKING_PIN, pull_up=True)
btn_reset = Button(RESET_PIN, pull_up=True)
btn_gemini = Button(GEMINI_PIN, pull_up=True)

# ===== Global Control Variables =====
system_state = "WAITING"  # WAITING, TRACKING, GEMINI_PROCESSING
global_reset_flag = False
gemini_request_flag = False

# ===== Servo Configuration =====
MIN_ANGLE = 0
MAX_ANGLE = 180
current_angle = 90
target_angle = 90

SERVO_LEFT = 0      
SERVO_CENTER = 90   
SERVO_RIGHT = 180   
SERVO_RESPONSE_ZONE = 50

# Parallel servo movement system
SERVO_INCREMENTS = 20
movement_active = False
movement_lock = threading.Lock()
servo_queue = queue.Queue()
servo_thread = None

# ===== Vision Configuration =====
MODEL_PATH = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
SHM_NAME = 'video_frame'
SHM_FRAME_SHAPE = (300, 300, 3)
TOP_K = 10
THRESHOLD = 0.3
MIN_MATCH_THRESHOLD = 25
EXPAND_PIXELS = 40

# COCO labels
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

label_to_id = {v.lower(): k for k, v in coco_labels.items()}

# ===== Button Handlers =====
def on_tracking_pressed():
    """Handle tracking button press (GPIO 17)"""
    global system_state
    if system_state == "WAITING":
        system_state = "TRACKING"
        print("🎯 TRACKING BUTTON PRESSED (GPIO 17) - Starting tracking mode...")

def on_reset_pressed():
    """Handle reset button press (GPIO 27)"""
    global global_reset_flag, system_state
    if system_state == "TRACKING":
        global_reset_flag = True
        print("🔄 RESET BUTTON PRESSED (GPIO 27) - Resetting system...")

def on_gemini_pressed():
    """Handle Gemini API button press (GPIO 22)"""
    global gemini_request_flag, system_state
    gemini_request_flag = True
    print("🤖 GEMINI BUTTON PRESSED (GPIO 22) - Starting API call...")

# Assign button callbacks
btn_tracking.when_pressed = on_tracking_pressed
btn_reset.when_pressed = on_reset_pressed
btn_gemini.when_pressed = on_gemini_pressed

# ===== Servo Functions =====
def angle_to_value(angle):
    """Convert 0–180° to -1.0 to +1.0 range for gpiozero"""
    angle = max(MIN_ANGLE, min(MAX_ANGLE, angle))
    return -(angle - 90) / 90

def update_servo(angle):
    """Update servo position immediately"""
    global current_angle
    pos = angle_to_value(angle)
    servo.value = pos
    current_angle = angle
    print(f"Servo: {angle}°")

def servo_movement_worker():
    """Parallel servo movement thread - processes movement commands"""
    global current_angle, movement_active
    
    print("Servo movement thread started")
    
    while True:
        try:
            target = servo_queue.get(timeout=1)
            
            if target == "STOP":
                break
                
            with movement_lock:
                movement_active = True
                
                angle_diff = target - current_angle
                if angle_diff == 0:
                    movement_active = False
                    servo_queue.task_done()
                    continue
                
                step_size = angle_diff / SERVO_INCREMENTS
                delay_per_step = 0.05
                
                print(f"Moving from {current_angle}° to {target}° in {SERVO_INCREMENTS} steps")
                
                for i in range(1, SERVO_INCREMENTS + 1):
                    if not movement_active or global_reset_flag:
                        break
                        
                    next_angle = current_angle + (step_size * i)
                    next_angle = max(MIN_ANGLE, min(MAX_ANGLE, int(next_angle)))
                    
                    pos = angle_to_value(next_angle)
                    servo.value = pos
                    print(f"Step {i}/{SERVO_INCREMENTS}: {next_angle}°")
                    
                    time.sleep(delay_per_step)
                
                if movement_active and not global_reset_flag:
                    current_angle = target
                    pos = angle_to_value(target)
                    servo.value = pos
                    print(f"Reached target: {target}°")
                
                movement_active = False
                servo_queue.task_done()
                
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Servo thread error: {e}")
            movement_active = False

def start_servo_thread():
    """Start the parallel servo movement thread"""
    global servo_thread
    servo_thread = threading.Thread(target=servo_movement_worker)
    servo_thread.daemon = True
    servo_thread.start()

def move_servo_to_angle(target):
    """Queue a servo movement command (non-blocking)"""
    global movement_active
    
    if target == current_angle:
        return
    
    if movement_active:
        movement_active = False
        time.sleep(0.1)
    
    while not servo_queue.empty():
        try:
            servo_queue.get_nowait()
            servo_queue.task_done()
        except queue.Empty:
            break
    
    servo_queue.put(target)
    print(f"Queued servo movement to {target}°")

def stop_servo_thread():
    """Stop the servo movement thread"""
    global movement_active, servo_thread
    movement_active = False
    if servo_thread and servo_thread.is_alive():
        servo_queue.put("STOP")
        servo_thread.join(timeout=2.0)

# ===== Gemini API Integration with LMNT Text-to-Speech =====
class GeminiAPI:
    def __init__(self, shared_memory_frame):
        self.api_url = "https://gemini-room-description.vercel.app/analyze"
        self.shared_frame = shared_memory_frame
        self.audio_playing = False
        self.processing = False
        
        # Initialize pygame mixer for audio playback
        try:
            pygame.mixer.init()
            print("🔊 Audio system initialized")
        except Exception as e:
            print(f"⚠️ Audio system initialization failed: {e}")
        
    def capture_from_shared_memory(self):
        """Capture image from shared memory camera feed"""
        try:
            # Get current frame from shared memory
            frame = self.shared_frame.copy()
            
            # Convert BGR to RGB (OpenCV uses BGR by default)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            
            # Convert to bytes for API transmission
            buffer = BytesIO()
            pil_image.save(buffer, format='JPEG', quality=85)
            buffer.seek(0)
            
            print("📸 Image captured from shared memory camera")
            return buffer.getvalue()
            
        except Exception as e:
            print(f"❌ Error capturing from shared memory: {e}")
            return None
    
    def send_to_gemini_api(self, image_data):
        """Send image to Gemini API and get text response"""
        try:
            print("📡 Sending image to Gemini API...")
            
            files = {
                'file': ('image.jpg', image_data, 'image/jpeg')
            }
            
            response = requests.post(
                self.api_url, 
                files=files,
                timeout=30
            )
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    print("🤖 GEMINI API RESPONSE:")
                    print("=" * 50)
                    print(json.dumps(result, indent=2))
                    print("=" * 50)
                    
                    # Extract text description from response
                    description = result.get('description') or result.get('text') or result.get('analysis') or str(result)
                    
                    if description and isinstance(description, str) and len(description.strip()) > 0:
                        print(f"📝 Extracted description: {description[:100]}...")
                        return description.strip()
                    else:
                        print("⚠️ No valid description found in response")
                        return None
                        
                except json.JSONDecodeError:
                    # Handle plain text response
                    text_response = response.text.strip()
                    print("🤖 GEMINI API RESPONSE (text):")
                    print("=" * 50)
                    print(text_response)
                    print("=" * 50)
                    
                    if len(text_response) > 0:
                        return text_response
                    else:
                        print("⚠️ Empty text response")
                        return None
            else:
                print(f"❌ Gemini API Error: Status {response.status_code}")
                print(response.text)
                return None
                
        except requests.exceptions.Timeout:
            print("❌ Gemini API: Request timed out")
            return None
        except requests.exceptions.ConnectionError:
            print("❌ Gemini API: Could not connect")
            return None
        except Exception as e:
            print(f"❌ Gemini API Error: {e}")
            return None
    
    async def generate_audio_with_lmnt(self, text):
        """Generate audio from text using LMNT API"""
        try:
            print("🎵 Generating speech with LMNT...")
            
            async with Speech() as speech:
                synthesis = await speech.synthesize(text, 'lily')  # Using 'lily' voice
            
            # Save audio to temporary file
            temp_audio_file = "lmnt_generated_speech.mp3"
            
            with open(temp_audio_file, 'wb') as f:
                f.write(synthesis['audio'])
            
            print("✅ Audio generated successfully")
            return temp_audio_file
            
        except Exception as e:
            print(f"❌ Error generating audio with LMNT: {e}")
            return None
    
    def play_audio_file(self, audio_file_path):
        """Play MP3 audio file"""
        try:
            print("🔊 Playing generated speech...")
            self.audio_playing = True
            
            # Load and play the audio file
            pygame.mixer.music.load(audio_file_path)
            pygame.mixer.music.play()
            
            # Wait for audio to finish playing
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
                if not self.audio_playing:  # Allow interruption
                    break
            
            self.audio_playing = False
            print("✅ Audio playback completed")
            
            # Clean up temporary file
            try:
                os.remove(audio_file_path)
                print("🧹 Temporary audio file cleaned up")
            except:
                pass
                
        except Exception as e:
            print(f"❌ Error playing audio: {e}")
            self.audio_playing = False
    
    def stop_audio(self):
        """Stop currently playing audio"""
        try:
            if self.audio_playing:
                pygame.mixer.music.stop()
                self.audio_playing = False
                print("🔇 Audio stopped")
        except Exception as e:
            print(f"Error stopping audio: {e}")
    
    def process_image(self):
        """Complete workflow: Image → Gemini API → LMNT TTS → Audio Playback"""
        if self.processing:
            print("⚠️ Already processing an image, please wait...")
            return
            
        self.processing = True
        
        try:
            print("🤖 Starting complete image-to-speech workflow...")
            
            # Step 1: Capture image from shared memory
            image_data = self.capture_from_shared_memory()
            if not image_data:
                print("❌ Failed to capture image")
                return
            
            # Step 2: Send to Gemini API and get text response
            description_text = self.send_to_gemini_api(image_data)
            if not description_text:
                print("❌ Failed to get description from Gemini API")
                return
            
            # Step 3: Generate audio using LMNT
            print("🎵 Converting text to speech...")
            audio_file = asyncio.run(self.generate_audio_with_lmnt(description_text))
            if not audio_file:
                print("❌ Failed to generate audio with LMNT")
                return
            
            # Step 4: Play the generated audio
            self.play_audio_file(audio_file)
            
            print("✅ Complete workflow finished successfully")
            
        except Exception as e:
            print(f"❌ Error in complete workflow: {e}")
        finally:
            self.processing = False

# ===== Tracking Classes =====
class ServoController:
    def __init__(self):
        update_servo(current_angle)
        start_servo_thread()
        print("Servo initialized with parallel movement thread (20 increments)")

    def calculate_servo_movement(self, object_center, frame_center, bbox_info=None):
        """Calculate servo movement with reset checking"""
        global target_angle, global_reset_flag
        
        if global_reset_flag:
            return
        
        frame_cx, frame_cy = frame_center
        obj_x, obj_y = object_center
        
        offset_x = obj_x - frame_cx
        
        if bbox_info:
            bbox_center_x = (bbox_info['x'] + bbox_info['w'] / 2)
            blend_factor = 0.7
            obj_x = int(blend_factor * obj_x + (1 - blend_factor) * bbox_center_x)
            offset_x = obj_x - frame_cx
        
        new_target_angle = target_angle
        
        if abs(offset_x) < SERVO_RESPONSE_ZONE:
            new_target_angle = SERVO_CENTER
        elif offset_x < -SERVO_RESPONSE_ZONE:
            new_target_angle = SERVO_LEFT
        elif offset_x > SERVO_RESPONSE_ZONE:
            new_target_angle = SERVO_RIGHT
        
        if new_target_angle != target_angle:
            target_angle = new_target_angle
            move_servo_to_angle(new_target_angle)

    def reset_to_center(self):
        """Reset servo to center"""
        global target_angle
        target_angle = SERVO_CENTER
        move_servo_to_angle(SERVO_CENTER)

    def cleanup(self):
        """Clean up servo"""
        global movement_active
        movement_active = False
        stop_servo_thread()
        servo.detach()
        print("Servo cleaned up")

class VoiceCommandListener:
    def __init__(self, command_queue):
        self.command_queue = command_queue
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.listening = True
        
        print("🎤 Calibrating microphone...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
        print("🎤 Ready for voice commands")
    
    def listen_for_commands(self):
        print("🎤 Say object name to start tracking")
        
        while self.listening and not global_reset_flag:
            try:
                with self.microphone as source:
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=4)
                    
                try:
                    command = self.recognizer.recognize_google(audio).lower()
                    print(f"🎤 Heard: {command}")
                    
                    words = command.split()
                    detected_objects = []
                    
                    for word in words:
                        if word in label_to_id:
                            detected_objects.append(word)
                    
                    if detected_objects:
                        target_object = detected_objects[0]
                        print(f"🎯 Tracking: {target_object}")
                        self.command_queue.put(target_object)
                
                except sr.UnknownValueError:
                    pass
                except sr.RequestError as e:
                    print(f"🎤 Recognition error: {e}")
                    time.sleep(1)
                    
            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                print(f"🎤 Audio error: {e}")
                time.sleep(1)
    
    def stop(self):
        self.listening = False

class EnhancedSmoothTracker:
    def __init__(self):
        self.sift = cv2.SIFT_create(nfeatures=2000)
        
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        self.tracker = None
        self.tracking_active = False
        self.tracker_id = 1
        self.min_matches = MIN_MATCH_THRESHOLD
        
        self.current_center = None
        self.previous_center = None
        self.smoothing_factor = 0.7
        self.frame_center = None
        
        self.current_bbox = None
        self.bbox_smoothing_factor = 0.6

    def set_frame_center(self, frame_shape):
        h, w = frame_shape[:2]
        self.frame_center = (w // 2, h // 2)

    def extract_features_with_surrounding(self, bbox, frame):
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
        if global_reset_flag:
            return False
            
        x, y, w, h = bbox
        
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)

        if w < 20 or h < 20:
            return False

        self.set_frame_center(frame.shape)

        kp, desc, template_img = self.extract_features_with_surrounding((x, y, w, h), frame)
        
        if kp is None or desc is None:
            return False

        self.tracker = {
            'id': self.tracker_id,
            'original_bbox': (x, y, w, h),
            'template_kp': kp,
            'template_desc': desc,
            'label': label
        }
        
        self.current_center = (x + w//2, y + h//2)
        self.previous_center = self.current_center
        self.current_bbox = {'x': x, 'y': y, 'w': w, 'h': h}
        self.tracking_active = True
        
        print(f"🎯 Tracker initialized: {label}")
        return True

    def smooth_position(self, new_center):
        if self.current_center is None:
            return new_center
        
        smooth_x = int(self.smoothing_factor * self.current_center[0] + 
                      (1 - self.smoothing_factor) * new_center[0])
        smooth_y = int(self.smoothing_factor * self.current_center[1] + 
                      (1 - self.smoothing_factor) * new_center[1])
        
        return (smooth_x, smooth_y)

    def smooth_bbox(self, new_bbox):
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
        if not self.tracking_active or self.tracker is None or global_reset_flag:
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
                    x, y, w, h = self.tracker['original_bbox']
                    corners = np.float32([[x, y], [x+w, y], [x+w, y+h], [x, y+h]]).reshape(-1, 1, 2)
                    transformed_corners = cv2.perspectiveTransform(corners, H)
                    
                    new_center = np.mean(transformed_corners, axis=0).astype(int)[0]
                    
                    x_coords = transformed_corners[:, 0, 0]
                    y_coords = transformed_corners[:, 0, 1]
                    new_bbox = {
                        'x': int(np.min(x_coords)),
                        'y': int(np.min(y_coords)),
                        'w': int(np.max(x_coords) - np.min(x_coords)),
                        'h': int(np.max(y_coords) - np.min(y_coords))
                    }
                    
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
        self.tracker = None
        self.tracking_active = False
        self.tracker_id += 1
        self.current_center = None
        self.previous_center = None
        self.current_bbox = None

    def draw_tracking(self, frame, result):
        if result and not global_reset_flag:
            cv2.circle(frame, result['center'], 8, (255, 0, 0), -1)
            
            if 'bbox' in result:
                bbox = result['bbox']
                color = (0, 255, 0) if result['tracking_quality'] == 'good' else (0, 255, 255)
                cv2.rectangle(frame, (bbox['x'], bbox['y']), 
                            (bbox['x'] + bbox['w'], bbox['y'] + bbox['h']), color, 2)
            
            if self.frame_center:
                cx, cy = self.frame_center
                cv2.line(frame, (cx-20, cy), (cx+20, cy), (128, 128, 128), 1)
                cv2.line(frame, (cx, cy-20), (cx, cy+20), (128, 128, 128), 1)
        
        return frame

def find_largest_target_object(objects, labels, target_object, scale_x, scale_y):
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

def wait_for_button_press():
    """Wait for initial button press and return the selected mode"""
    print("\n" + "="*60)
    print("🚀 THREE-SWITCH CONTROL SYSTEM READY")
    print("="*60)
    print("Press a tactile switch to choose mode:")
    print("🎯 GPIO 17 (TRACKING) - Start object tracking with voice")
    print("🤖 GPIO 22 (GEMINI) - Capture image and analyze with AI")
    print("🔄 GPIO 27 (RESET) - Available during tracking")
    print("="*60)
    
    while True:
        if btn_tracking.is_pressed:
            print("🎯 Tracking mode selected!")
            time.sleep(0.3)  # Debounce
            return "TRACKING"
        elif btn_gemini.is_pressed:
            print("🤖 Gemini API mode selected!")
            time.sleep(0.3)  # Debounce
            return "GEMINI"
        time.sleep(0.1)

def main():
    global system_state, global_reset_flag, gemini_request_flag
    
    # Initialize interpreter
    interpreter = make_interpreter(MODEL_PATH)
    interpreter.allocate_tensors()
    inference_size = input_size(interpreter)

    # Initialize shared memory
    shm = shared_memory.SharedMemory(name=SHM_NAME)
    frame_np = np.ndarray(SHM_FRAME_SHAPE, dtype=np.uint8, buffer=shm.buf)

    # Initialize Gemini API with shared memory
    gemini_api = GeminiAPI(frame_np)

    print("🚀 System initializing...")
    print("Servo moves in 20 increments using parallel thread")
    print("Using shared memory for camera feed")

    try:
        while True:
            # Reset all flags and state
            global_reset_flag = False
            gemini_request_flag = False
            system_state = "WAITING"
            
            # Wait for button press to choose mode
            mode = wait_for_button_press()
            
            if mode == "GEMINI":
                system_state = "GEMINI_PROCESSING"
                print("🤖 Starting Gemini API processing...")
                gemini_api.process_image()
                print("✅ Gemini processing complete. Returning to start...")
                continue
            
            elif mode == "TRACKING":
                system_state = "TRACKING"
                print("🎯 Starting tracking mode...")
                
                # Initialize tracking components
                tracker = EnhancedSmoothTracker()
                servo_controller = ServoController()
                
                # Initialize voice command system
                command_queue = queue.Queue()
                voice_listener = VoiceCommandListener(command_queue)
                
                # Start voice recognition thread
                voice_thread = threading.Thread(target=voice_listener.listen_for_commands)
                voice_thread.daemon = True
                voice_thread.start()
                
                # Tracking state variables
                target_object = None
                detection_phase = False
                waiting_for_command = True
                tracking_started = False
                
                # Tracking loop
                while system_state == "TRACKING" and not global_reset_flag:
                    # Check for Gemini button during tracking
                    if gemini_request_flag:
                        print("🤖 Gemini button pressed during tracking - processing API call...")
                        gemini_api.process_image()
                        gemini_request_flag = False
                        print("✅ Gemini processing complete. Continuing tracking...")
                    
                    # Voice command phase
                    if waiting_for_command:
                        if not command_queue.empty():
                            command = command_queue.get()
                            
                            if command in label_to_id:
                                target_object = command
                                detection_phase = True
                                waiting_for_command = False
                                tracking_started = True
                                tracker.reset_tracker()
                                voice_listener.stop()
                                print(f"🎯 Starting detection for: {target_object}")
                        
                        time.sleep(0.1)
                        continue
                    
                    # Camera processing phase
                    frame = frame_np.copy()
                    
                    # PHASE 1: Detection
                    if detection_phase and not tracker.tracking_active and target_object:
                        rgb_resized = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        run_inference(interpreter, rgb_resized.tobytes())
                        objs = get_objects(interpreter, THRESHOLD)[:TOP_K]
                        
                        height, width, _ = frame.shape
                        scale_x, scale_y = width / inference_size[0], height / inference_size[1]

                        largest_obj = find_largest_target_object(objs, coco_labels, target_object, scale_x, scale_y)
                        
                        if largest_obj:
                            bbox = largest_obj['bbox']
                            x0, y0, x1, y1 = int(bbox.xmin), int(bbox.ymin), int(bbox.xmax), int(bbox.ymax)
                            
                            tracker_bbox = (x0, y0, x1 - x0, y1 - y0)
                            
                            success = tracker.initialize_tracker(tracker_bbox, frame, largest_obj['label'])
                            if success:
                                detection_phase = False
                                print(f"🎯 Tracking started: {target_object}")

                    # PHASE 2: Tracking with Servo Control
                    elif tracker.tracking_active:
                        result = tracker.track_using_stored_features(frame)
                        
                        if result:
                            frame = tracker.draw_tracking(frame, result)
                            
                            # Update servo position based on tracking
                            if tracker.frame_center and 'center' in result and 'bbox' in result:
                                servo_controller.calculate_servo_movement(
                                    result['center'], 
                                    tracker.frame_center, 
                                    result['bbox']
                                )

                    # Display frame only when tracking is active
                    if tracking_started and not global_reset_flag:
                        # Add status information
                        status_text = f"Current: {current_angle}° | Target: {target_angle}° | Moving: {'Yes' if movement_active else 'No'}"
                        button_text = f"RESET: GPIO27 | GEMINI: GPIO22"
                        audio_text = f"Audio: {'Playing' if gemini_api.audio_playing else 'Ready'}"
                        processing_text = f"Processing: {'Yes' if gemini_api.processing else 'No'}"
                        
                        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        cv2.putText(frame, button_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                        cv2.putText(frame, f"Tracking: {target_object}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.putText(frame, audio_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                        cv2.putText(frame, processing_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        
                        cv2.imshow('Object Tracking', frame)
                    
                    # Check for keyboard input (backup controls)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        global_reset_flag = True
                        break
                    elif key == ord('r'):
                        global_reset_flag = True
                        break
                
                # Cleanup tracking mode
                voice_listener.stop()
                servo_controller.cleanup()
                cv2.destroyAllWindows()
                
                if global_reset_flag:
                    print("🔄 Resetting system...")
                else:
                    print("✅ Tracking session ended")

    except KeyboardInterrupt:
        print("\n🛑 System stopped by user")
    except Exception as e:
        print(f"❌ System error: {e}")
    finally:
        # Final cleanup
        try:
            voice_listener.stop()
        except:
            pass
        try:
            servo_controller.cleanup()
        except:
            pass
        try:
            gemini_api.stop_audio()
        except:
            pass
        cv2.destroyAllWindows()
        shm.close()
        print("🧹 Final cleanup completed")

if __name__ == '__main__':
    main()
