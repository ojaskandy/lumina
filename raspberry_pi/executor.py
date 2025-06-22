# wrapper.py
import subprocess
import threading
import time

def run_script1():
    # Producer script - creates shared memory (using python3.11)
    subprocess.run(['python3.11', '/home/asus/Desktop/hackathon/camera_interface.py'])

def run_script2():
    # Consumer script - uses shared memory data (using python)
    subprocess.run(['python', '/home/asus/Desktop/hackathon/edge_compute.py'])

if __name__ == "__main__":
    thread1 = threading.Thread(target=run_script1)
    thread2 = threading.Thread(target=run_script2)
    
    # Start producer first
    thread1.start()
    
    # Wait for producer to initialize and create shared memory
    time.sleep(5)  # Wait for shared memory setup
    
    # Start consumer
    thread2.start()
    
    thread1.join()
    thread2.join()
