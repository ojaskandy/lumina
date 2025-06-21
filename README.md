# Voice Navigation System

A Python script that listens for keyboard input, records speech, converts it to text, and matches keywords with MobileNet object classes for navigation commands.

## Features

- 🎮 Press 'm' key to activate voice recognition
- 🎤 Records speech and converts to text using Google Speech Recognition
- 🔍 Extracts keywords from speech and matches them to MobileNet classes
- ✅ Provides acknowledgment when objects are detected
- 🔄 Supports fuzzy matching for similar words

## Setup

### Prerequisites

1. **Python 3.7+** installed
2. **Microphone** access
3. **Internet connection** (for Google Speech Recognition API)

### Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

**Note for macOS users:** You may need to install PortAudio first:
```bash
brew install portaudio
```

**Note for Linux users:** You may need to install additional packages:
```bash
sudo apt-get install python3-pyaudio portaudio19-dev
```

## Usage

1. Run the script:
```bash
python voice_navigation.py
```

2. **Controls:**
   - Press `m` key to activate voice navigation
   - Press `q` key to quit the application

3. **Voice Commands:**
   Say phrases like:
   - "Take me to a chair"
   - "Find a laptop"
   - "Locate a person"
   - "Go to the couch"

## How It Works

1. **Keyboard Listener**: Detects when the 'm' key is pressed
2. **Speech Recording**: Activates microphone and records speech (5-second timeout)
3. **Speech-to-Text**: Converts audio to text using Google's speech recognition
4. **Keyword Extraction**: Removes navigation words and extracts object keywords
5. **MobileNet Matching**: Matches keywords to MobileNet class names (with fuzzy matching)
6. **Acknowledgment**: Provides feedback when objects are detected

## MobileNet Classes Supported

The script currently recognizes 80 MobileNet classes including:
- People and animals (person, cat, dog, horse, etc.)
- Vehicles (car, bicycle, motorcycle, bus, etc.)
- Furniture (chair, couch, bed, dining table, etc.)
- Electronics (laptop, tv, cell phone, etc.)
- Kitchen items (microwave, refrigerator, oven, etc.)
- And many more...

## Example Interaction

```
🚀 Voice Navigation System Started
Setup complete. Press 'm' to start voice navigation, 'q' to quit.

[User presses 'm']
==================================================
🔊 Voice navigation activated!

🎤 Listening... Speak now!
[User says: "take me to a chair"]
🔄 Processing speech...
📝 You said: 'take me to a chair'
🔍 Extracted keywords: ['chair']
✅ ACKNOWLEDGED: Navigation request to 'chair' detected!
🎯 Taking you to a chair...
```

## Troubleshooting

### Microphone Issues
- Ensure your microphone is working and not muted
- Check system permissions for microphone access
- Try adjusting microphone sensitivity

### Speech Recognition Issues
- Speak clearly and at a normal pace
- Ensure stable internet connection
- Try speaking closer to the microphone

### Installation Issues
- Make sure you have the latest pip: `pip install --upgrade pip`
- For PyAudio issues on Windows, try: `pip install pipwin` then `pipwin install pyaudio`

## Dependencies

- `speechrecognition`: For speech-to-text conversion
- `pyaudio`: For microphone access
- `pynput`: For keyboard input detection 