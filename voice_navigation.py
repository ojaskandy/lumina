#!/usr/bin/env python3
"""
Voice Navigation Script
Listens for 'm' key press, records speech, and matches keywords to MobileNet classes.
"""

import speech_recognition as sr
import pyaudio
from pynput import keyboard
import threading
import time
import re
from difflib import get_close_matches

# MobileNet class list
MOBILENET_CLASSES = {
    "0": "person", "1": "bicycle", "2": "car", "3": "motorcycle", "4": "airplane",
    "5": "bus", "6": "train", "7": "truck", "8": "boat", "9": "traffic light",
    "10": "fire hydrant", "12": "stop sign", "13": "parking meter", "14": "bench",
    "15": "bird", "16": "cat", "17": "dog", "18": "horse", "19": "sheep",
    "20": "cow", "21": "elephant", "22": "bear", "23": "zebra", "24": "giraffe",
    "26": "backpack", "27": "umbrella", "30": "handbag", "31": "tie", "32": "suitcase",
    "33": "frisbee", "34": "skis", "35": "snowboard", "36": "sports ball", "37": "kite",
    "38": "baseball bat", "39": "baseball glove", "40": "skateboard", "41": "surfboard",
    "42": "tennis racket", "43": "bottle", "45": "wine glass", "46": "cup", "47": "fork",
    "48": "knife", "49": "spoon", "50": "bowl", "51": "banana", "52": "apple",
    "53": "sandwich", "54": "orange", "55": "broccoli", "56": "carrot", "57": "hot dog",
    "58": "pizza", "59": "donut", "60": "cake", "61": "chair", "62": "couch",
    "63": "potted plant", "64": "bed", "66": "dining table", "69": "toilet", "71": "tv",
    "72": "laptop", "73": "mouse", "74": "remote", "75": "keyboard", "76": "cell phone",
    "77": "microwave", "78": "oven", "79": "toaster", "80": "sink", "81": "refrigerator",
    "83": "book", "84": "clock", "85": "vase", "86": "scissors", "87": "teddy bear",
    "88": "hair drier", "89": "toothbrush"
}

class VoiceNavigator:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.is_listening = False
        self.class_names = list(MOBILENET_CLASSES.values())
        
        # Adjust for ambient noise
        print("Adjusting for ambient noise... Please wait.")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
        print("Setup complete. Press 'm' to start voice navigation, 'q' to quit.")
    
    def listen_for_speech(self):
        """Record and convert speech to text"""
        try:
            print("\n🎤 Listening... Speak now!")
            with self.microphone as source:
                # Listen for audio with timeout
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
            
            print("🔄 Processing speech...")
            
            # Convert speech to text using Google's speech recognition
            text = self.recognizer.recognize_google(audio).lower()
            print(f"📝 You said: '{text}'")
            
            return text
            
        except sr.WaitTimeoutError:
            print("❌ No speech detected within timeout period.")
            return None
        except sr.UnknownValueError:
            print("❌ Could not understand the speech.")
            return None
        except sr.RequestError as e:
            print(f"❌ Error with speech recognition service: {e}")
            return None
    
    def extract_keywords(self, text):
        """Extract potential object keywords from text"""
        # Remove common navigation words and clean text
        navigation_words = [
            'take', 'me', 'to', 'a', 'an', 'the', 'find', 'locate', 'where', 'is',
            'go', 'navigate', 'move', 'walk', 'run', 'search', 'for', 'look', 'show'
        ]
        
        # Split text into words and filter out navigation words
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [word for word in words if word not in navigation_words]
        
        return keywords
    
    def find_mobilenet_matches(self, keywords):
        """Find matching MobileNet classes from keywords"""
        matches = []
        
        for keyword in keywords:
            # Direct match
            if keyword in self.class_names:
                matches.append(keyword)
            else:
                # Fuzzy matching for similar words
                close_matches = get_close_matches(keyword, self.class_names, n=1, cutoff=0.6)
                if close_matches:
                    matches.append(close_matches[0])
        
        return matches
    
    def process_voice_command(self):
        """Main processing function for voice commands"""
        if self.is_listening:
            return
        
        self.is_listening = True
        
        try:
            # Listen for speech
            text = self.listen_for_speech()
            
            if text:
                # Extract keywords
                keywords = self.extract_keywords(text)
                print(f"🔍 Extracted keywords: {keywords}")
                
                # Find MobileNet matches
                matches = self.find_mobilenet_matches(keywords)
                
                if matches:
                    for match in matches:
                        print(f"✅ ACKNOWLEDGED: Navigation request to '{match}' detected!")
                        print(f"🎯 Taking you to a {match}...")
                else:
                    print("❓ No recognized objects found in your speech.")
                    print("💡 Try saying something like: 'take me to a chair' or 'find a laptop'")
            
        finally:
            self.is_listening = False
    
    def on_key_press(self, key):
        """Handle key press events"""
        try:
            if key.char == 'm':
                if not self.is_listening:
                    print("\n" + "="*50)
                    print("🔊 Voice navigation activated!")
                    # Run speech processing in separate thread to avoid blocking
                    threading.Thread(target=self.process_voice_command, daemon=True).start()
                else:
                    print("⏳ Already listening... Please wait.")
            elif key.char == 'q':
                print("\n👋 Goodbye!")
                return False  # Stop listener
        except AttributeError:
            # Special keys (like ctrl, alt, etc.) don't have char attribute
            pass
    
    def start(self):
        """Start the voice navigation system"""
        print("🚀 Voice Navigation System Started")
        print("📋 Available MobileNet classes:", ", ".join(sorted(self.class_names)))
        print("\nControls:")
        print("  Press 'm' - Activate voice navigation")
        print("  Press 'q' - Quit application")
        print("  Say phrases like: 'take me to a chair', 'find a laptop', 'locate a person'")
        
        # Start keyboard listener
        with keyboard.Listener(on_press=self.on_key_press) as listener:
            listener.join()

def main():
    try:
        navigator = VoiceNavigator()
        navigator.start()
    except KeyboardInterrupt:
        print("\n\n👋 Application interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Make sure you have installed all required dependencies:")
        print("pip install -r requirements.txt")

if __name__ == "__main__":
    main() 