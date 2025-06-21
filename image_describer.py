#!/usr/bin/env python3
"""
Image Describer with Gemini API
Captures screenshots on keystroke and describes what's happening using Gemini Vision.
"""

import os
import time
import base64
from PIL import ImageGrab
from pynput import keyboard
import google.generativeai as genai
from io import BytesIO
import threading

class ImageDescriber:
    def __init__(self, api_key):
        # Configure Gemini API
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        self.processing = False
        
        print("🚀 Image Describer Started!")
        print("📸 Press 's' to take a screenshot and get description")
        print("🔄 Press 'h' for help")
        print("❌ Press 'q' to quit")
        
    def capture_screenshot(self):
        """Capture a screenshot and return as PIL Image"""
        try:
            screenshot = ImageGrab.grab()
            print(f"📸 Screenshot captured: {screenshot.size}")
            return screenshot
        except Exception as e:
            print(f"❌ Error capturing screenshot: {e}")
            return None
    
    def image_to_base64(self, image):
        """Convert PIL Image to base64 string"""
        try:
            buffer = BytesIO()
            image.save(buffer, format='PNG')
            img_bytes = buffer.getvalue()
            return base64.b64encode(img_bytes).decode('utf-8')
        except Exception as e:
            print(f"❌ Error converting image: {e}")
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
    
    def process_screenshot(self):
        """Main processing function for screenshot and description"""
        if self.processing:
            print("⏳ Already processing an image, please wait...")
            return
        
        self.processing = True
        
        try:
            print("\n" + "="*60)
            print("📸 Taking screenshot...")
            
            # Capture screenshot
            screenshot = self.capture_screenshot()
            if screenshot is None:
                return
            
            # Get description from Gemini
            description = self.describe_image(screenshot)
            
            # Display results
            print("\n🤖 Gemini's Description:")
            print("-" * 40)
            print(description)
            print("-" * 40)
            
            # Save screenshot with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.png"
            screenshot.save(filename)
            print(f"💾 Screenshot saved as: {filename}")
            
        except Exception as e:
            print(f"❌ Error processing screenshot: {e}")
        finally:
            self.processing = False
            print("\n✅ Ready for next screenshot (press 's')")
    
    def show_help(self):
        """Display help information"""
        print("\n" + "="*50)
        print("📋 HELP - Image Describer Commands")
        print("="*50)
        print("🔹 Press 's' - Take screenshot and describe")
        print("🔹 Press 'h' - Show this help")
        print("🔹 Press 'q' - Quit application")
        print("="*50)
        print("🤖 Using Gemini 1.5 Flash for image analysis")
        print("📸 Screenshots saved with timestamp")
        print("="*50)
    
    def on_key_press(self, key):
        """Handle key press events"""
        try:
            if key.char == 's':
                print("\n📸 Screenshot requested!")
                # Run in separate thread to avoid blocking
                threading.Thread(target=self.process_screenshot, daemon=True).start()
            elif key.char == 'h':
                self.show_help()
            elif key.char == 'q':
                print("\n👋 Goodbye!")
                return False  # Stop listener
        except AttributeError:
            # Special keys (like ctrl, alt, etc.) don't have char attribute
            pass
    
    def start(self):
        """Start the image describer system"""
        print("🎮 System ready! Listening for keystrokes...")
        print("💡 Tip: Make sure the content you want to describe is visible on screen")
        
        # Start keyboard listener
        with keyboard.Listener(on_press=self.on_key_press) as listener:
            listener.join()

def main():
    # Gemini API Key
    API_KEY = "AIzaSyAif2rtUE1NAiffCnk6L9Q2YT3_NhaJglA"
    
    try:
        describer = ImageDescriber(API_KEY)
        describer.start()
    except KeyboardInterrupt:
        print("\n\n👋 Application interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Make sure you have installed all required dependencies:")
        print("pip install -r requirements.txt")

if __name__ == "__main__":
    main() 