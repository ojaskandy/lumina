import speech_recognition as sr

# COCO labels with their class IDs
coco_labels = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus", 6: "train", 7: "truck", 8: "boat",
    9: "traffic light", 10: "fire hydrant", 12: "stop sign", 13: "parking meter", 14: "bench", 15: "bird", 16: "cat",
    17: "dog", 18: "horse", 19: "sheep", 20: "cow", 21: "elephant", 22: "bear", 23: "zebra", 24: "giraffe",
    26: "backpack", 27: "umbrella", 30: "handbag", 31: "tie", 32: "suitcase", 33: "frisbee", 34: "skis",
    35: "snowboard", 36: "sports ball", 37: "kite", 38: "baseball bat", 39: "baseball glove", 40: "skateboard",
    41: "surfboard", 42: "tennis racket", 43: "bottle", 45: "wine glass", 46: "cup", 47: "fork", 48: "knife",
    49: "spoon", 50: "bowl", 51: "banana", 52: "apple", 53: "sandwich", 54: "orange", 55: "broccoli", 56: "carrot",
    57: "hot dog", 58: "pizza", 59: "donut", 60: "cake", 61: "chair", 62: "couch", 63: "potted plant", 64: "bed",
    66: "dining table", 69: "toilet", 71: "tv", 72: "laptop", 73: "mouse", 74: "remote", 75: "keyboard",
    76: "cell phone", 77: "microwave", 78: "oven", 79: "toaster", 80: "sink", 81: "refrigerator", 83: "book",
    84: "clock", 85: "vase", 86: "scissors", 87: "teddy bear", 88: "hair drier", 89: "toothbrush"
}

# Convert label list to lowercase label → ID map for fast lookup
label_to_id = {v.lower(): k for k, v in coco_labels.items()}

def listen_for_commands():
    recognizer = sr.Recognizer()

    try:
        with sr.Microphone() as source:
            print("Listening... (press Ctrl+C to stop)")
            recognizer.adjust_for_ambient_noise(source, duration=1)

            while True:
                try:
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=3)
                    command = recognizer.recognize_google(audio).lower()
                    print(f"You said: {command}")

                    words = command.split()
                    found = False

                    for word in words:
                        if word in label_to_id:
                            print(f"Found label: '{word}' (ID: {label_to_id[word]})")
                            found = True

                    if not found:
                        print("No matching COCO label found.")

                except sr.WaitTimeoutError:
                    continue
                except sr.UnknownValueError:
                    print("Could not understand audio.")
                except sr.RequestError:
                    print("Speech recognition service error.")
    except KeyboardInterrupt:
        print("\nStopped by user.")

if __name__ == "__main__":
    listen_for_commands()
