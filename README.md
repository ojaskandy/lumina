# Lumina

Camera-based AI-powered smart compass that aids visually impaired individuals

## Overview

Lumina is an innovative assistive technology project that combines computer vision, AI, and hardware components to help visually impaired individuals navigate and understand their surroundings. The system uses a Raspberry Pi with camera capabilities, AI-powered object detection, and tactile feedback to provide real-time assistance.

## Features

- **Real-time Object Detection**: Uses MobileNet SSD v2 with Edge TPU for efficient object detection
- **Voice Commands**: Speech recognition for hands-free operation
- **Tactile Feedback**: Servo motor provides directional guidance
- **AI-Powered Analysis**: Integration with Google Gemini AI for detailed scene understanding
- **Audio Output**: Text-to-speech conversion for verbal feedback
- **Hardware Controls**: Physical buttons for easy operation

## Project Structure

```
lumina/
├── main.py                 # FastAPI backend for Gemini AI integration
├── requirements.txt        # Python dependencies
├── env_example.txt         # Environment variables template
├── raspberry_pi/          # Raspberry Pi implementation
│   ├── edge_compute.py    # Main edge computing logic
│   ├── camera_interface.py # Camera handling
│   ├── executor.py        # Execution management
│   ├── helper/            # Utility modules
│   │   ├── gpio_check.py  # GPIO testing utilities
│   │   └── servo_control.py # Servo motor control
│   ├── safe_code/         # Alternative implementations
│   │   ├── mobile_net_v*.py # Different MobileNet versions
│   │   ├── audio_recognition.py # Audio processing
│   │   └── single_imx708.py # Camera interface
│   └── mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite # AI model
└── README.md              # This file
```

## Prerequisites

### Hardware Requirements
- Raspberry Pi 5 (recommended)
- Raspberry Pi Camera Module (OV5647 or compatible)
- Servo Motor (for tactile feedback)
- Tactile Switches/Buttons
- Edge TPU USB Accelerator (optional, for enhanced performance)
- Speakers or Headphones with microphone

### Software Requirements
- Python 3.8+
- Raspberry Pi OS (latest)
- Google Gemini API key
- LMNT API key (for text-to-speech)

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd lumina
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install Additional System Dependencies
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install required system packages
sudo apt install -y python3-pip python3-venv
sudo apt install -y libatlas-base-dev libhdf5-dev libhdf5-serial-dev
sudo apt install -y libjasper-dev libqtcore4 libqtgui4 libqt4-test
sudo apt install -y libportaudio2 portaudio19-dev
sudo apt install -y libespeak-ng1 espeak-ng-data
sudo apt install -y libgstreamer1.0-0 gstreamer1.0-plugins-base
sudo apt install -y gstreamer1.0-plugins-good gstreamer1.0-plugins-bad
sudo apt install -y gstreamer1.0-plugins-ugly gstreamer1.0-libav
sudo apt install -y gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa
sudo apt install -y gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5
sudo apt install -y gstreamer1.0-pulseaudio

# Install Edge TPU runtime (if using Edge TPU)
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt update
sudo apt install libedgetpu1-std
```

### 4. Configure Environment Variables
```bash
# Copy the example environment file
cp env_example.txt .env

# Edit the .env file with your API keys
nano .env
```

Add your API keys:
```
GOOGLE_API_KEY=your_google_gemini_api_key_here
LMNT_API_KEY=your_lmnt_api_key_here
```

### 5. Hardware Setup

#### GPIO Pin Configuration
- **Servo Motor**: SG 90
- **Tracking Button**: GPIO 17
- **Reset Button**: GPIO 27
- **Gemini API Button**: GPIO 22

#### Camera Setup
```bash
# Enable camera interface
sudo raspi-config
# Navigate to Interface Options > Camera > Enable

# Test camera
vcgencmd get_camera
```

## Usage

### Starting the Backend Server
```bash
# Start the FastAPI server
python main.py
```

The server will be available at `http://localhost:8000`

### Running the Raspberry Pi Application
```bash
cd raspberry_pi
python edge_compute.py
```

### Button Controls
- **Tracking Button (GPIO 17)**: Start object tracking mode
- **Reset Button (GPIO 27)**: Reset the system to waiting state
- **Gemini Button (GPIO 22)**: Capture image and analyze with Gemini AI

### Voice Commands
The system supports voice commands for hands-free operation:
- "Start tracking"
- "Stop tracking"
- "Reset"
- "Analyze scene"

## API Endpoints

### Backend API (FastAPI)
- `GET /`: API information and status
- `GET /health`: Health check
- `POST /analyze`: Analyze uploaded image with Gemini AI

### Example API Usage
```bash
# Analyze an image
curl -X POST "http://localhost:8000/analyze" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@image.jpg" \
     -F "prompt=Describe this scene for a visually impaired person"
```

## Configuration

### Model Configuration
The system uses MobileNet SSD v2 with COCO labels for object detection. The model file is located at:
```
raspberry_pi/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite
```

### Servo Configuration
- **Min Angle**: 0°
- **Max Angle**: 180°
- **Center Position**: 90°


## Troubleshooting

### Common Issues

1. **Camera not detected**
   ```bash
   # Check camera status
   vcgencmd get_camera
   
   # Restart camera service
   sudo systemctl restart camera.service
   ```

2. **GPIO permissions**
   ```bash
   # Add user to gpio group
   sudo usermod -a -G gpio $USER
   
   # Reboot or log out/in
   sudo reboot
   ```

3. **Audio issues**
   ```bash
   # Check audio devices
   aplay -l
   
   # Test audio
   speaker-test -t wav -c 2
   ```

4. **Edge TPU not detected**
   ```bash
   # Check USB devices
   lsusb | grep Google
   
   # Install Edge TPU runtime
   sudo apt install libedgetpu1-std
   ```



**Note**: This project is designed to assist visually impaired individuals. Please ensure proper testing and safety measures when deploying in real-world environments. 