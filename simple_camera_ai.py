#!/usr/bin/env python3
"""
Simple Camera AI Describer
Press a key to capture image and get AI description from Gemini.
"""

import cv2
import time
from PIL import Image
from pynput import keyboard
import google.generativeai as genai

class SimpleCameraAI:
    def __init__(self, api_key):
        # Configure Gemini API
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("❌ Error: Could not open camera")
            exit(1)
        
        print("📷 Simple Camera AI Ready!")
        print("📸 Press SPACE to capture and describe")
        print("❌ Press 'q' to quit")
    
    def capture_image(self):
        """Capture image from camera and save as JPEG"""
        ret, frame = self.cap.read()
        if not ret:
            print("❌ Failed to capture image")
            return None
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        
        # Save image with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{timestamp}.jpg"
        image.save(filename, "JPEG", quality=85)
        
        print(f"📸 Image captured and saved: {filename}")
        return image, filename
    
    def describe_image(self, image):
        """Send image to Gemini and get description"""
        print("🤖 Sending to Gemini AI...")
        
        try:
            response = self.model.generate_content([
                "Describe what you see in this image in 2-3 clear sentences.",
                image
            ])
            
            if response.text:
                return response.text.strip()
            else:
                return "No description received"
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    def process_capture(self):
        """Main function: capture → send to Gemini → get result"""
        print("\n" + "="*50)
        
        # 1. Capture image
        result = self.capture_image()
        if not result:
            return
        
        image, filename = result
        
        # 2. Send to Gemini AI (backend server)
        description = self.describe_image(image)
        
        # 3. Display result
        print("🤖 AI Description:")
        print("-" * 30)
        print(description)
        print("-" * 30)
        print("✅ Ready for next capture")
    
    def on_key_press(self, key):
        """Handle key presses"""
        try:
            if key == keyboard.Key.space:
                self.process_capture()
            elif key.char == 'q':
                print("👋 Goodbye!")
                self.cleanup()
                return False
        except AttributeError:
            pass
    
    def cleanup(self):
        """Release camera"""
        if self.cap:
            self.cap.release()
    
    def start(self):
        """Start listening for key presses"""
        with keyboard.Listener(on_press=self.on_key_press) as listener:
            listener.join()

def main():
    # Gemini API Key
    API_KEY = "AIzaSyAif2rtUE1NAiffCnk6L9Q2YT3_NhaJglA"
    
    try:
        app = SimpleCameraAI(API_KEY)
        app.start()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 