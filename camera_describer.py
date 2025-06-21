#!/usr/bin/env python3
"""
Camera Image Describer with Gemini API
Captures images from camera on keystroke and describes what's happening using Gemini Vision.
"""

import os
import time
import cv2
import numpy as np
from PIL import Image
from pynput import keyboard
import google.generativeai as genai
import threading

class CameraImageDescriber:
    def __init__(self, api_key):
        # Configure Gemini API
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        self.processing = False
        self.camera_active = False
        self.cap = None
        self.preview_supported = True
        
        print("🚀 Camera Image Describer Started!")
        print("📷 Press 'c' to start/stop camera preview")
        print("📸 Press 's' to capture image and get description")
        print("🔄 Press 'h' for help")
        print("❌ Press 'q' to quit")
        
    def initialize_camera(self):
        """Initialize camera connection"""
        try:
            print("🔌 Initializing camera...")
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("❌ Could not open camera index 0, trying alternatives...")
                for i in range(1, 5):
                    print(f"🔍 Trying camera index {i}...")
                    self.cap = cv2.VideoCapture(i)
                    if self.cap.isOpened():
                        print(f"✅ Successfully opened camera {i}")
                        break
                else:
                    print("❌ No cameras found")
                    return False
            
            # Set camera properties for better quality
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            print("✅ Camera initialized successfully!")
            return True
        except Exception as e:
            print(f"❌ Error initializing camera: {e}")
            return False
    
    def start_camera_preview(self):
        """Start camera preview in a separate thread"""
        if self.camera_active:
            print("📷 Camera preview already running")
            return
        
        if not self.cap or not self.cap.isOpened():
            if not self.initialize_camera():
                return
        
        if not self.preview_supported:
            print("📷 Camera preview not supported on this system")
            print("💡 You can still capture images with 's' key")
            return
        
        self.camera_active = True
        threading.Thread(target=self._camera_preview_loop, daemon=True).start()
        print("📷 Camera preview started! Press 'c' to stop, 's' to capture")
    
    def stop_camera_preview(self):
        """Stop camera preview"""
        self.camera_active = False
        try:
            cv2.destroyAllWindows()
        except:
            pass
        print("📷 Camera preview stopped")
    
    def _camera_preview_loop(self):
        """Camera preview loop running in separate thread"""
        try:
            cv2.namedWindow('Camera Preview', cv2.WINDOW_AUTOSIZE)
        except Exception as e:
            print(f"❌ Cannot create preview window: {e}")
            print("📷 Preview disabled, but capture still works with 's' key")
            self.preview_supported = False
            self.camera_active = False
            return
        
        while self.camera_active and self.cap and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to read from camera")
                    break
                
                # Add overlay text
                cv2.putText(frame, "Press 's' to capture & describe", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "Press 'c' to stop preview", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Camera Preview', frame)
                
                # Check for window close or ESC key
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                    
                # Check if window was closed
                try:
                    if cv2.getWindowProperty('Camera Preview', cv2.WND_PROP_VISIBLE) < 1:
                        break
                except:
                    break
                    
            except Exception as e:
                print(f"❌ Preview error: {e}")
                break
        
        self.camera_active = False
        try:
            cv2.destroyAllWindows()
        except:
            pass
    
    def capture_image(self):
        """Capture an image from camera and return as PIL Image"""
        try:
            if not self.cap or not self.cap.isOpened():
                if not self.initialize_camera():
                    return None
            
            # Capture frame
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Failed to capture image from camera")
                return None
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            image = Image.fromarray(frame_rgb)
            print(f"📸 Image captured: {image.size}")
            return image
            
        except Exception as e:
            print(f"❌ Error capturing image: {e}")
            return None
    
    def describe_image(self, image):
        """Send image to Gemini API and get description"""
        try:
            print("🔄 Analyzing image with Gemini...")
            
            # Create a detailed prompt for better descriptions
            prompt = """
            Describe what's happening in this image in detail. Focus on:
            - Main subjects and objects visible
            - Actions or activities taking place
            - Setting and environment
            - Notable details or interesting elements
            - Overall mood or atmosphere
            - If there are people, describe what they're doing
            
            Provide a clear, descriptive response in 2-3 sentences.
            """
            
            # Send to Gemini API
            response = self.model.generate_content([prompt, image])
            
            if response.text:
                return response.text.strip()
            else:
                return "❌ No description received from Gemini API"
                
        except Exception as e:
            return f"❌ Error analyzing image: {str(e)}"
    
    def process_capture(self):
        """Main processing function for image capture and description"""
        if self.processing:
            print("⏳ Already processing an image, please wait...")
            return
        
        self.processing = True
        
        try:
            print("\n" + "="*60)
            print("📸 Capturing image from camera...")
            
            # Capture image
            image = self.capture_image()
            if image is None:
                return
            
            # Get description from Gemini
            description = self.describe_image(image)
            
            # Display results
            print("\n🤖 Gemini's Description:")
            print("-" * 40)
            print(description)
            print("-" * 40)
            
            # Save image with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"camera_capture_{timestamp}.png"
            image.save(filename)
            print(f"💾 Image saved as: {filename}")
            
        except Exception as e:
            print(f"❌ Error processing capture: {e}")
        finally:
            self.processing = False
            print("\n✅ Ready for next capture (press 's')")
    
    def show_help(self):
        """Display help information"""
        print("\n" + "="*50)
        print("📋 HELP - Camera Image Describer Commands")
        print("="*50)
        print("🔹 Press 'c' - Start/stop camera preview (optional)")
        print("🔹 Press 's' - Capture image and describe")
        print("🔹 Press 'h' - Show this help")
        print("🔹 Press 'q' - Quit application")
        print("="*50)
        print("🤖 Using Gemini 1.5 Flash for image analysis")
        print("📷 Camera captures saved with timestamp")
        print("💡 Tip: You can capture without preview if needed")
        print("="*50)
    
    def cleanup(self):
        """Clean up camera resources"""
        self.camera_active = False
        if self.cap:
            self.cap.release()
        try:
            cv2.destroyAllWindows()
        except:
            pass
    
    def on_key_press(self, key):
        """Handle key press events"""
        try:
            if key.char == 's':
                print("\n📸 Image capture requested!")
                # Run in separate thread to avoid blocking
                threading.Thread(target=self.process_capture, daemon=True).start()
            elif key.char == 'c':
                if self.camera_active:
                    self.stop_camera_preview()
                else:
                    self.start_camera_preview()
            elif key.char == 'h':
                self.show_help()
            elif key.char == 'q':
                print("\n👋 Goodbye!")
                self.cleanup()
                return False  # Stop listener
        except AttributeError:
            # Special keys (like ctrl, alt, etc.) don't have char attribute
            pass
    
    def start(self):
        """Start the camera image describer system"""
        print("🎮 System ready! Listening for keystrokes...")
        print("💡 Tip: Press 's' to capture instantly, or 'c' for preview first")
        
        # Start keyboard listener
        with keyboard.Listener(on_press=self.on_key_press) as listener:
            listener.join()

def main():
    # Gemini API Key
    API_KEY = "AIzaSyAif2rtUE1NAiffCnk6L9Q2YT3_NhaJglA"
    
    try:
        describer = CameraImageDescriber(API_KEY)
        describer.start()
    except KeyboardInterrupt:
        print("\n\n👋 Application interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Make sure you have installed all required dependencies:")
        print("pip install -r requirements.txt")
    finally:
        # Cleanup
        try:
            cv2.destroyAllWindows()
        except:
            pass

if __name__ == "__main__":
    main() 