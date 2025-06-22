import cv2
import requests
import json
import threading
import time
from pynput import keyboard
from io import BytesIO
from PIL import Image
import asyncio
from lmnt.api import Speech
import dotenv
import os   

dotenv.load_dotenv()  # Load environment variables from .env file


class CameraToGemini:
    def __init__(self):
        self.camera = None
        self.api_url = "https://gemini-room-description.vercel.app/analyze"
        self.running = True
        self.processing = False
        
    def initialize_camera(self):
        """Initialize the camera capture"""
        try:
            self.camera = cv2.VideoCapture(0)  # Use default camera
            if not self.camera.isOpened():
                print("Error: Could not open camera")
                return False
            
            # Set camera properties for better quality
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            print("Camera initialized successfully")
            return True
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
    
    def capture_image(self):
        """Capture an image from the camera"""
        if not self.camera or not self.camera.isOpened():
            print("Camera not available")
            return None
            
        try:
            ret, frame = self.camera.read()
            if not ret:
                print("Failed to capture image")
                return None
            
            # Convert BGR to RGB (OpenCV uses BGR by default)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            
            # Convert to bytes for API transmission
            buffer = BytesIO()
            pil_image.save(buffer, format='JPEG', quality=85)
            buffer.seek(0)  # Reset buffer position
            
            return buffer.getvalue()
        except Exception as e:
            print(f"Error capturing image: {e}")
            return None
    
    def send_to_api(self, image_data):
        """Send the captured image to the Gemini API and generate audio from the response"""
        try:
            print("Sending image to API...")
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
                    print("API Response:")
                    print(json.dumps(result, indent=2))                    # Extract the description text (adjust key as needed)
                    description = result.get('description') or result.get('text') or str(result)
                    if description:
                        print("Generating audio from description...")
                        # Synthesize speech using LMNT (async)
                        asyncio.run(self.generate_audio(description))
                    else:
                        print("No description found in API response.")
                except json.JSONDecodeError:
                    print("API Response (text):")
                    print(response.text)
            else:
                print(f"API Error: Status {response.status_code}")
                print(response.text)
        except requests.exceptions.Timeout:
            print("Error: Request timed out")
        except requests.exceptions.ConnectionError:
            print("Error: Could not connect to API")
        except Exception as e:
            print(f"Error sending to API: {e}")
    
    def process_image(self):
        """Complete image processing workflow"""
        if self.processing:
            print("Already processing an image, please wait...")
            return
            
        self.processing = True
        try:
            print("Capturing image...")
            image_data = self.capture_image()
            
            if image_data:
                self.send_to_api(image_data)
            else:
                print("Failed to capture image")
        finally:
            self.processing = False
    
    def on_key_press(self, key):
        """Handle keyboard press events"""
        if not self.running:
            return False
            
        try:
            print(f"Key pressed: {key.char if hasattr(key, 'char') and key.char else key}")
        except AttributeError:
            print(f"Special key pressed: {key}")
        
        # Start image processing in a separate thread to avoid blocking
        threading.Thread(target=self.process_image, daemon=True).start()
    
    def cleanup(self):
        """Clean up resources"""
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        print("Cleanup completed")
    
    async def generate_audio(self, text):
        """Generate audio from text using LMNT"""
        try:
            async with Speech() as speech:
                synthesis = await speech.synthesize(text, 'lily')
            
            # Save audio to file (LMNT returns MP3 format)
            with open('output.mp3', 'wb') as f:
                f.write(synthesis['audio'])
            print("Audio saved as output.mp3")
        except Exception as e:
            print(f"Error generating audio: {e}")
    
    def run(self):
        """Main program loop"""
        print("Initializing Camera to Gemini program...")
        
        if not self.initialize_camera():
            print("Failed to initialize camera. Please check your camera connection.")
            return
        
        print("Program started successfully!")
        print("Press any key to capture and analyze an image")
        print("Press Ctrl+C to exit")
        
        # Set up keyboard listener
        listener = keyboard.Listener(on_press=self.on_key_press)
        listener.start()
        
        try:
            # Keep the main thread alive
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nShutting down...")
            self.running = False
            listener.stop()
        finally:
            self.cleanup()

if __name__ == "__main__":
    app = CameraToGemini()
    app.run()