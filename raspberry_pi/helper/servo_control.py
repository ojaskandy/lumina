import curses
from gpiozero import Servo
from time import sleep

# === SG90 Servo Setup ===
GPIO_PIN = 14
MIN_PW = 0.0005
MAX_PW = 0.0024

servo = Servo(GPIO_PIN, min_pulse_width=MIN_PW, max_pulse_width=MAX_PW)

# === Angle Management ===
ANGLE_STEP = 5         # Degrees per keypress
MIN_ANGLE = 0
MAX_ANGLE = 180
current_angle = 90     # Start centered

def angle_to_value(angle):
    """Convert 0–180° to -1.0 to +1.0 range for gpiozero."""
    angle = max(MIN_ANGLE, min(MAX_ANGLE, angle))
    return (angle - 90) / 90  # Maps: 0°→-1.0, 90°→0.0, 180°→+1.0

def update_servo(angle):
    pos = angle_to_value(angle)
    servo.value = pos
    print(f"Angle: {angle}°, Servo Value: {pos:.2f}")

# === Curses Keyboard Loop ===
def main(stdscr):
    global current_angle

    stdscr.nodelay(True)
    stdscr.clear()
    stdscr.addstr(0, 0, "Press 'a'/'d' to rotate, 'q' to quit")

    update_servo(current_angle)

    while True:
        key = stdscr.getch()
        if key == ord('a'):
            current_angle = max(MIN_ANGLE, current_angle - ANGLE_STEP)
            update_servo(current_angle)
        elif key == ord('d'):
            current_angle = min(MAX_ANGLE, current_angle + ANGLE_STEP)
            update_servo(current_angle)
        elif key == ord('q'):
            break

        stdscr.addstr(2, 0, f"Current Angle: {current_angle:3d}°   ")
        stdscr.refresh()
        sleep(0.08)

try:
    curses.wrapper(main)
except KeyboardInterrupt:
    print("\nInterrupted")
finally:
    servo.detach()
    print("Servo released. Program exited.")
