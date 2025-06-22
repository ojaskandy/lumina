from gpiozero import Button
from signal import pause

# Define buttons with internal pull-up resistors enabled
button1 = Button(17, pull_up=True)
button2 = Button(22, pull_up=True)
button3 = Button(27, pull_up=True)

# Define event handlers for button presses
def on_button1_pressed():
    print("Button on GPIO17 pressed")

def on_button2_pressed():
    print("Button on GPIO22 pressed")

def on_button3_pressed():
    print("Button on GPIO27 pressed")

# Attach handlers
button1.when_pressed = on_button1_pressed
button2.when_pressed = on_button2_pressed
button3.when_pressed = on_button3_pressed

print("Press any button (GPIO 17, 22, or 27)...")

# Keep the script running
pause()
