#!/usr/bin/env python3
"""
Backend Camera AI - Capture images and send to Gemini backend
Captures camera images on keypress and sends to backend endpoint for analysis
"""

import cv2
import requests
import json
import base64
from datetime import datetime
import threading
import time
from pynput import keyboard
import os

class BackendCameraAI:
    def __init__(self):
        self.cap = None
        self.running = False
        self.processing = False
        self.backend_url = "https://gemini-room-description.vercel.app/analyze"
        
    def initialize_camera(self):
        """Initialize camera with error handling"""
        try:
            print("🔌 Initializing camera...")
            self.cap = cv2.VideoCapture(0)
            
            if not self.cap.isOpened():
                print("❌ Error: Could not open camera")
                return False
                
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            print("✅ Camera initialized successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Camera initialization error: {e}")
            return False
    
    def capture_image(self):
        """Capture image from camera"""
        if not self.cap or not self.cap.isOpened():
            print("❌ Camera not available")
            return None
            
        ret, frame = self.cap.read()
        if not ret:
            print("❌ Failed to capture image")
            return None
            
        return frame
    
    def send_to_backend(self, image_data):
        """Send image to backend endpoint"""
        try:
            print("🔄 Sending image to backend...")
            
            # Encode image as JPEG
            _, buffer = cv2.imencode('.jpg', image_data)
            
            # Prepare multipart form data with correct field name
            files = {
                'file': ('capture.jpg', buffer.tobytes(), 'image/jpeg')
            }
            
            # Make POST request
            response = requests.post(
                self.backend_url,
                files=files,
                timeout=30
            )
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    return result
                except json.JSONDecodeError:
                    return {"description": response.text}
            else:
                print(f"❌ Backend error: {response.status_code}")
                try:
                    error_detail = response.json()
                    print(f"❌ Error details: {error_detail}")
                except:
                    print(f"❌ Error response: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error: {e}")
            return None
        except Exception as e:
            print(f"❌ Error sending to backend: {e}")
            return None
    
    def save_image(self, image_data):
        """Save image locally with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"backend_capture_{timestamp}.jpg"
        
        try:
            cv2.imwrite(filename, image_data)
            return filename
        except Exception as e:
            print(f"❌ Error saving image: {e}")
            return None
    
    def process_capture(self):
        """Capture image and send to backend"""
        if self.processing:
            print("⏳ Already processing an image, please wait...")
            return
            
        self.processing = True
        
        try:
            print("=" * 60)
            print("📸 Capturing image...")
            
            # Capture image
            image = self.capture_image()
            if image is None:
                return
                
            print(f"📸 Image captured: {image.shape[:2]}")
            
            # Save image locally
            filename = self.save_image(image)
            if filename:
                print(f"💾 Image saved as: {filename}")
            
            # Send to backend
            result = self.send_to_backend(image)
            
            if result:
                print("🤖 Backend Response:")
                print("-" * 40)
                
                # Handle different response formats
                if isinstance(result, dict):
                    if 'description' in result:
                        print(result['description'])
                    elif 'analysis' in result:
                        print(result['analysis'])
                    else:
                        print(json.dumps(result, indent=2))
                else:
                    print(result)
                    
                print("-" * 40)
            else:
                print("❌ Failed to get response from backend")
                
            print("✅ Ready for next capture")
            
        except Exception as e:
            print(f"❌ Error during processing: {e}")
        finally:
            self.processing = False
    
    def on_key_press(self, key):
        """Handle key press events"""
        try:
            if key == keyboard.Key.space:
                # Capture and analyze image
                threading.Thread(target=self.process_capture, daemon=True).start()
                
            elif key.char == 'h':
                self.show_help()
                
            elif key.char == 'q':
                print("👋 Goodbye!")
                self.cleanup()
                return False
                
        except AttributeError:
            # Special keys (like space) don't have char attribute
            pass
        except Exception as e:
            print(f"❌ Key handler error: {e}")
    
    def show_help(self):
        """Display help information"""
        print("=" * 50)
        print("📋 HELP - Backend Camera AI Commands")
        print("=" * 50)
        print("🔹 Press SPACE - Capture image and send to backend")
        print("🔹 Press 'h' - Show this help")
        print("🔹 Press 'q' - Quit application")
        print("=" * 50)
        print(f"🌐 Backend URL: {self.backend_url}")
        print("📷 Images saved locally with timestamp")
        print("💡 Tip: Make sure you have good lighting for better analysis")
        print("=" * 50)
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
    
    def run(self):
        """Main application loop"""
        print("📷 Backend Camera AI Started!")
        print("📸 Press SPACE to capture and analyze")
        print("❌ Press 'q' to quit")
        print("=" * 50)
        
        # Initialize camera
        if not self.initialize_camera():
            print("❌ Failed to initialize camera. Exiting...")
            return
        
        self.running = True
        self.show_help()
        
        # Start keyboard listener
        print("🎮 System ready! Listening for keystrokes...")
        print("💡 Tip: Press SPACE to capture image and send to backend")
        
        with keyboard.Listener(on_press=self.on_key_press) as listener:
            try:
                listener.join()
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
            finally:
                self.cleanup()

def main():
    """Main function"""
    try:
        app = BackendCameraAI()
        app.run()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Application error: {e}")

if __name__ == "__main__":
    main() 