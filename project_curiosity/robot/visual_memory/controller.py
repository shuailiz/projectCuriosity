import cv2
import time
import numpy as np
import sys
import os

# Ensure we can import from sibling directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from project_curiosity.robot.servo_control.pc_controller import FastStableController, select_port
from . import config as C

class RobotInterface:
    """
    Interface for controlling the 2-servo robot and capturing visual data.
    """
    def __init__(self, port=None, camera_index=0):
        # Initialize Servo Controller
        if port is None:
            print("Searching for servo controller port...")
            port = select_port()
            if not port:
                raise Exception("No servo controller port found or selected.")
        
        print(f"Connecting to servo controller on {port}...")
        self.controller = FastStableController(port)
        
        # Initialize Camera
        print(f"Opening camera {camera_index}...")
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise Exception(f"Could not open camera {camera_index}")
            
        # Set camera resolution (optional, to speed up or match encoder)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Track current servo positions
        self.current_angles = [0.0, 0.0]
        self._update_angles_from_robot()
        
        # Wait for camera to warm up
        time.sleep(1.0)
        print("Robot Interface Initialized.")

    def _update_angles_from_robot(self):
        """Read actual angles from the robot."""
        a1, a2 = self.controller.get_angles()
        if a1 is not None and a2 is not None:
            self.current_angles = [a1, a2]
        else:
            print("Warning: Could not read initial angles, assuming [0, 0]")
            self.current_angles = [0.0, 0.0]

    def get_frame(self):
        """Capture a frame from the webcam."""
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            return None
        # Convert BGR (OpenCV) to RGB (PIL/Torch)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb

    def move_servos(self, delta_s1, delta_s2):
        """
        Apply delta changes to servos.
        Args:
            delta_s1: Change in angle for Servo 1 (degrees)
            delta_s2: Change in angle for Servo 2 (degrees)
        Returns:
            tuple: (applied_delta_s1, applied_delta_s2) - actual deltas applied after clipping
        """
        # Read latest angles first to be safe? 
        # Or trust our internal state to avoid latency?
        # Let's trust internal state for speed, but maybe sync occasionally.
        
        new_s1 = self.current_angles[0] + delta_s1
        new_s2 = self.current_angles[1] + delta_s2
        
        # Clip to hardware limits defined in config
        # S1 Limits
        s1_min, s1_max = C.SERVO_LIMITS[0]
        new_s1 = max(s1_min, min(s1_max, new_s1))
        
        # S2 Limits
        s2_min, s2_max = C.SERVO_LIMITS[1]
        new_s2 = max(s2_min, min(s2_max, new_s2))
        
        # Calculate actual deltas
        actual_delta_s1 = new_s1 - self.current_angles[0]
        actual_delta_s2 = new_s2 - self.current_angles[1]
        
        # Send command
        # "servos <a1> <a2>"
        command = f"servos {new_s1:.2f} {new_s2:.2f}"
        if self.controller.send(command):
            self.current_angles = [new_s1, new_s2]
            return actual_delta_s1, actual_delta_s2
        else:
            print("Error sending servo command")
            return 0.0, 0.0

    def step(self, action):
        """
        Execute an action and return the new state.
        Args:
            action: list/array of [delta_s1, delta_s2]
        Returns:
            next_frame: RGB image
            actual_action: [actual_delta_s1, actual_delta_s2]
        """
        # Scale action from model output (if normalized) or use directly
        # For now assume action is in degrees directly or we apply a scale factor
        # If the model outputs raw values, we might want to clamp them to ACTION_SCALE
        
        # Clip action to max step size
        d1 = max(-C.ACTION_SCALE, min(C.ACTION_SCALE, action[0]))
        d2 = max(-C.ACTION_SCALE, min(C.ACTION_SCALE, action[1]))
        
        # Move robot
        real_d1, real_d2 = self.move_servos(d1, d2)
        
        # Wait for movement to settle?
        # Servos are fast, but maybe a small sleep is needed for the image to be stable
        time.sleep(0.1) 
        
        # Capture new frame
        next_frame = self.get_frame()
        
        return next_frame, [real_d1, real_d2]

    def close(self):
        self.cap.release()
        self.controller.close()
