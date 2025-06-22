import cv2
import numpy as np
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
TARGET_OBJECT = "chair"
MIN_MATCH_THRESHOLD = 50
EXPAND_PIXELS = 30

class SIFTObjectTrackerCSI:
    def __init__(self):
        # SIFT setup with increased features for better tracking
        self.sift = cv2.SIFT_create(nfeatures=2000)
        
        # FLANN matcher for homography-based matching
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Tracking state - SINGLE OBJECT FOCUS
        self.tracker = None
        self.tracking_active = False
        self.tracker_id = 1
        self.min_matches = MIN_MATCH_THRESHOLD
        
        # Feature matching parameters
        self.match_ratio = 0.7
        self.ransac_threshold = 5.0

    def extract_features_with_surrounding(self, bbox, frame):
        """Extract features from bbox and surrounding area - ONCE ONLY"""
        x, y, w, h = bbox
        frame_h, frame_w = frame.shape[:2]
        
        # Calculate expanded region for feature extraction
        expand_x = max(EXPAND_PIXELS, w // 4)
        expand_y = max(EXPAND_PIXELS, h // 4)
        
        # Expanded bounding box for feature extraction
        x_exp = max(0, x - expand_x)
        y_exp = max(0, y - expand_y)
        w_exp = min(frame_w - x_exp, w + 2 * expand_x)
        h_exp = min(frame_h - y_exp, h + 2 * expand_y)
        
        # Extract region with surroundings
        expanded_roi = frame[y_exp:y_exp+h_exp, x_exp:x_exp+w_exp].copy()
        
        if expanded_roi.size == 0:
            return None, None, None
            
        gray = cv2.cvtColor(expanded_roi, cv2.COLOR_BGR2GRAY)
        kp, desc = self.sift.detectAndCompute(gray, None)
        
        if desc is None or len(desc) < self.min_matches:
            print(f"Insufficient features: {len(desc) if desc is not None else 0}/{self.min_matches}")
            return None, None, None
            
        # Adjust keypoint coordinates to global frame coordinates
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
        """Initialize tracker with features - CALLED ONCE"""
        x, y, w, h = bbox
        
        # Ensure bbox is within frame bounds
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)

        if w < 20 or h < 20:
            print("Bounding box too small to track.")
            return False

        # Extract features with surrounding area - THIS HAPPENS ONLY ONCE
        kp, desc, template_img = self.extract_features_with_surrounding((x, y, w, h), frame)
        
        if kp is None or desc is None:
            return False

        # Store the tracker data - PERMANENT MEMORY OF THE OBJECT
        self.tracker = {
            'id': self.tracker_id,
            'original_bbox': (x, y, w, h),
            'template_img': template_img,
            'template_kp': kp,
            'template_desc': desc,
            'label': label,
            'total_features': len(desc)
        }
        
        self.tracking_active = True
        
        print("=" * 60)
        print(f"🎯 TRACKER INITIALIZED - ID: {self.tracker_id}")
        print(f"📦 Object: {label}")
        print(f"🔍 Features extracted: {len(desc)}")
        print(f"📍 Bounding box: {bbox}")
        print(f"✅ Now tracking ONLY this object using stored features")
        print(f"🚫 MobileNet detection DISABLED - focusing on feature tracking")
        print("=" * 60)
        
        return True

    def track_using_stored_features(self, frame):
        """Track object using ONLY stored features - NO MORE DETECTION"""
        if not self.tracking_active or self.tracker is None:
            return None

        # Extract features from current frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, desc = self.sift.detectAndCompute(gray, None)

        if desc is None or len(desc) < 2:
            return None

        try:
            # Match against STORED template features
            matches = self.flann.knnMatch(self.tracker['template_desc'], desc, k=2)
            
            # Apply Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < self.match_ratio * n.distance:
                        good_matches.append(m)

            if len(good_matches) >= self.min_matches:
                # Extract matched keypoints using STORED template keypoints
                src_pts = np.float32([self.tracker['template_kp'][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                # Find homography using RANSAC
                H, mask = cv2.findHomography(
                    src_pts, dst_pts, 
                    cv2.RANSAC, 
                    self.ransac_threshold
                )
                
                if H is not None:
                    # Transform the ORIGINAL bounding box corners using stored bbox
                    x, y, w, h = self.tracker['original_bbox']
                    corners = np.float32([[x, y], [x+w, y], [x+w, y+h], [x, y+h]]).reshape(-1, 1, 2)
                    transformed_corners = cv2.perspectiveTransform(corners, H)
                    
                    # Calculate center of transformed bbox
                    center = np.mean(transformed_corners, axis=0).astype(int)[0]
                    
                    # Count inlier matches
                    inlier_matches = np.sum(mask) if mask is not None else len(good_matches)
                    
                    return {
                        'center': tuple(center),
                        'corners': transformed_corners,
                        'matches': inlier_matches,
                        'total_matches': len(good_matches),
                        'id': self.tracker['id'],
                        'label': self.tracker['label'],
                        'total_features': self.tracker['total_features']
                    }
                    
        except Exception as e:
            print(f"Feature tracking error: {e}")
        
        return None

    def draw_tracking_result(self, frame, result):
        """Draw tracking results"""
        if result:
            # Draw blue dot at center
            cv2.circle(frame, result['center'], 10, (255, 0, 0), -1)  # Blue dot
            
            # Draw transformed bounding box
            if 'corners' in result:
                corners = np.int32(result['corners']).reshape((-1, 1, 2))
                cv2.polylines(frame, [corners], True, (255, 0, 0), 2)
            
            # Draw comprehensive tracking info
            info_lines = [
                f"ID:{result['id']} {result['label']}",
                f"Matches: {result['matches']}/{result['total_matches']}",
                f"Features: {result['total_features']}"
            ]
            
            y_offset = result['center'][1] - 40
            for i, line in enumerate(info_lines):
                cv2.putText(frame, line, (result['center'][0]+15, y_offset + i*20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
        return frame

def find_largest_target_object(objects, labels, target_object, scale_x, scale_y):
    """Find the largest bounding box of target object - USED ONLY ONCE"""
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

    # Initialize SIFT tracker
    tracker = SIFTObjectTrackerCSI()

    print("SIFT Object Tracker - Single Object Focus Mode")
    print("=" * 50)
    print(f"🎯 Target: '{TARGET_OBJECT}'")
    print(f"📊 Min features: {MIN_MATCH_THRESHOLD}")
    print("🔄 Phase 1: MobileNet detection (find biggest object)")
    print("🔒 Phase 2: Pure feature tracking (no more detection)")
    print("=" * 50)
    print("Press 'r' to restart detection phase")
    print("Press 'q' to quit")

    detection_phase = True  # Flag to control when to stop detection

    try:
        while True:
            frame = frame_np.copy()
            
            # PHASE 1: Detection (only when not tracking)
            if detection_phase and not tracker.tracking_active:
                rgb_resized = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Run MobileNet inference - ONLY DURING DETECTION PHASE
                run_inference(interpreter, rgb_resized.tobytes())
                objs = get_objects(interpreter, THRESHOLD)[:TOP_K]
                
                height, width, _ = frame.shape
                scale_x, scale_y = width / inference_size[0], height / inference_size[1]

                # Find the largest target object
                largest_obj = find_largest_target_object(objs, labels, TARGET_OBJECT, scale_x, scale_y)
                
                if largest_obj:
                    bbox = largest_obj['bbox']
                    x0, y0, x1, y1 = int(bbox.xmin), int(bbox.ymin), int(bbox.xmax), int(bbox.ymax)
                    
                    # Convert to (x, y, w, h) format
                    tracker_bbox = (x0, y0, x1 - x0, y1 - y0)
                    
                    # Initialize tracker with features - THIS HAPPENS ONLY ONCE
                    success = tracker.initialize_tracker(tracker_bbox, frame, largest_obj['label'])
                    if success:
                        detection_phase = False  # STOP DETECTION FOREVER
                        print("🚫 Detection phase ended - now using ONLY stored features")
                
                # Show detection status
                cv2.putText(frame, f"Detecting {TARGET_OBJECT}...", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, "MobileNet active", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # PHASE 2: Pure Feature Tracking (using stored features only)
            elif tracker.tracking_active:
                # Track using ONLY stored features - NO MORE MOBILENET
                result = tracker.track_using_stored_features(frame)
                
                if result:
                    frame = tracker.draw_tracking_result(frame, result)
                    # Show tracking status
                    cv2.putText(frame, "Feature Tracking Mode", (10, frame.shape[0] - 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(frame, "MobileNet disabled", (10, frame.shape[0] - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                else:
                    cv2.putText(frame, "Tracking lost - features not matching", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Display frame
            cv2.imshow('Single Object Feature Tracker', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset to detection phase
                tracker.tracking_active = False
                tracker.tracker = None
                detection_phase = True
                print("\n🔄 Restarting detection phase...")
                print("🔍 MobileNet reactivated to find new target")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cv2.destroyAllWindows()
        shm.close()
        print("Cleanup completed")

if __name__ == '__main__':
    main()
