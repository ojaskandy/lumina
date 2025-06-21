import cv2
import numpy as np
import time

class SIFTObjectTracker:
    def __init__(self):
        self.sift = cv2.SIFT_create(nfeatures=1000)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        self.trackers = []  # List of tracked objects
        self.temp_bbox = None
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.min_matches = 10

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.temp_bbox = None

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.end_point = (x, y)
            self.temp_bbox = (
                min(self.start_point[0], x),
                min(self.start_point[1], y),
                abs(x - self.start_point[0]),
                abs(y - self.start_point[1])
            )

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.start_point and (x, y) != self.start_point:
                bbox = (
                    min(self.start_point[0], x),
                    min(self.start_point[1], y),
                    abs(x - self.start_point[0]),
                    abs(y - self.start_point[1])
                )
                self.temp_bbox = None
                print("Bounding box selected, extracting features...")
                self.add_tracker(bbox, param)

    def add_tracker(self, bbox, frame):
        x, y, w, h = bbox
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)

        if w < 20 or h < 20:
            return

        template_img = frame[y:y+h, x:x+w].copy()
        template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY) if len(template_img.shape) == 3 else template_img
        template_kp, template_desc = self.sift.detectAndCompute(template_gray, None)

        if template_desc is not None and len(template_desc) > self.min_matches:
            tracker = {
                'bbox': bbox,
                'template_img': template_img,
                'template_kp': template_kp,
                'template_desc': template_desc
            }
            self.trackers.append(tracker)
            print(f"Added tracker with {len(template_desc)} features")
        else:
            print("Not enough features detected in the selected region")

    def match_features(self, tracker, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        frame_kp, frame_desc = self.sift.detectAndCompute(frame_gray, None)

        if frame_desc is None or len(frame_desc) < 2:
            return None, None

        try:
            matches = self.flann.knnMatch(tracker['template_desc'], frame_desc, k=2)
            good_matches = [m for m, n in matches if len([m, n]) == 2 and m.distance < 0.7 * n.distance]
            return (good_matches, (tracker['template_kp'], frame_kp)) if len(good_matches) >= self.min_matches else (None, None)
        except Exception as e:
            print(f"Matching error: {e}")
            return None, None

    def estimate_object_position(self, tracker, matches, keypoints):
        if matches is None or keypoints is None:
            return None

        template_kp, frame_kp = keypoints
        template_pts = np.float32([template_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        frame_pts = np.float32([frame_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        try:
            homography, mask = cv2.findHomography(template_pts, frame_pts, cv2.RANSAC, 5.0)
            if homography is not None:
                h, w = tracker['template_img'].shape[:2]
                template_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                transformed_corners = cv2.perspectiveTransform(template_corners, homography)

                x_coords = transformed_corners[:, 0, 0]
                y_coords = transformed_corners[:, 0, 1]
                center_x = int(np.mean(x_coords))
                center_y = int(np.mean(y_coords))

                return {
                    'center': (center_x, center_y),
                    'matches': len([m for i, m in enumerate(matches) if mask[i]])
                }
        except Exception as e:
            print(f"Position estimation error: {e}")
        return None

    def draw_tracking_info(self, frame, results):
        result_frame = frame.copy()
        for result in results:
            if result is None:
                continue
            center = result['center']
            cv2.circle(result_frame, center, 8, (0, 0, 255), -1)
            cv2.putText(result_frame, f"Matches: {result['matches']}", (center[0] + 10, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return result_frame

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        cv2.namedWindow('Object Tracker')
        cv2.setMouseCallback('Object Tracker', self.mouse_callback, param=None)

        print("Instructions:")
        print("1. Draw bounding boxes for each object you want to track")
        print("2. Tracking will start automatically")
        print("3. Press 'r' to reset all tracked objects")
        print("4. Press 'q' to quit")

        fps_counter = 0
        fps_timer = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            display_frame = frame.copy()
            cv2.setMouseCallback('Object Tracker', self.mouse_callback, param=frame)

            if self.temp_bbox and self.drawing:
                x, y, w, h = self.temp_bbox
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(display_frame, 'Selecting...', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            results = []
            for tracker in self.trackers:
                matches, keypoints = self.match_features(tracker, frame)
                result = self.estimate_object_position(tracker, matches, keypoints)
                results.append(result)

            display_frame = self.draw_tracking_info(display_frame, results)

            fps_counter += 1
            if time.time() - fps_timer > 1.0:
                fps = fps_counter / (time.time() - fps_timer)
                fps_counter = 0
                fps_timer = time.time()
                cv2.putText(display_frame, f'FPS: {fps:.1f}', (display_frame.shape[1] - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow('Object Tracker', display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.trackers.clear()
                self.temp_bbox = None
                print("All trackers reset")

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = SIFTObjectTracker()
    tracker.run()
