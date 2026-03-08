import cv2
import time
import threading
import collections
import numpy as np

from .servo_control.pc_controller import FastStableController, select_port
from .. import config as C

class RobotInterface:
    """
    Interface for controlling the 2-servo robot and capturing visual data.

    The camera runs in a background thread, continuously filling a ring buffer
    with the latest frames. This decouples the display refresh rate from the
    action loop frequency, giving a live feed at all times.
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
        
        # Track current servo positions
        self.current_angles = [0.0, 0.0]
        self._update_angles_from_robot()
        
        # --- Background camera thread ---
        # Ring buffer holds the last N frames (RGB numpy arrays).
        # Sized to hold ~2 seconds of frames at expected FPS.
        buf_size = max(C.CLIP_FRAMES * 4, 64)
        self._frame_buffer = collections.deque(maxlen=buf_size)
        self._frame_lock = threading.Lock()
        self._cam_running = True
        self._cam_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._cam_thread.start()
        
        # Wait for camera to warm up and buffer to fill
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

    def _camera_loop(self):
        """Background thread: continuously read frames into the ring buffer."""
        while self._cam_running:
            ret, frame = self.cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with self._frame_lock:
                    self._frame_buffer.append(frame_rgb)

    def get_latest_frame(self):
        """Return the most recent frame from the ring buffer (RGB numpy array)."""
        with self._frame_lock:
            if self._frame_buffer:
                return self._frame_buffer[-1].copy()
        return None

    def get_frame(self):
        """Alias for get_latest_frame for backward compatibility."""
        return self.get_latest_frame()

    def get_clip(self, n_frames=None):
        """
        Return the last n_frames frames from the ring buffer as a list.

        Args:
            n_frames: Number of frames to return. Defaults to C.CLIP_FRAMES.
        Returns:
            list of numpy arrays (H, W, 3) RGB, length == n_frames (padded if needed).
        """
        if n_frames is None:
            n_frames = C.CLIP_FRAMES
        with self._frame_lock:
            buf = list(self._frame_buffer)
        if not buf:
            return None
        # Take the last n_frames; pad with the first frame if buffer is short
        if len(buf) >= n_frames:
            clip = buf[-n_frames:]
        else:
            pad = [buf[0]] * (n_frames - len(buf))
            clip = pad + buf
        return [f.copy() for f in clip]

    def move_servos(self, delta_s1, delta_s2):
        """
        Apply delta changes to servos.
        Args:
            delta_s1: Change in angle for Servo 1 (degrees)
            delta_s2: Change in angle for Servo 2 (degrees)
        Returns:
            tuple: (applied_delta_s1, applied_delta_s2) - actual deltas applied after clipping
        """
        new_s1 = self.current_angles[0] + delta_s1
        new_s2 = self.current_angles[1] + delta_s2
        
        s1_min, s1_max = C.SERVO_LIMITS[0]
        new_s1 = max(s1_min, min(s1_max, new_s1))
        
        s2_min, s2_max = C.SERVO_LIMITS[1]
        new_s2 = max(s2_min, min(s2_max, new_s2))
        
        actual_delta_s1 = new_s1 - self.current_angles[0]
        actual_delta_s2 = new_s2 - self.current_angles[1]
        
        command = f"servos {new_s1:.2f} {new_s2:.2f}"
        if self.controller.send(command):
            self.current_angles = [new_s1, new_s2]
            return actual_delta_s1, actual_delta_s2
        else:
            print("Error sending servo command")
            return 0.0, 0.0

    def step(self, action):
        """
        Execute an action, wait for settle, then collect a clip of frames.

        For 2D encoder: returns (latest_frame, actual_action)
        For 3D encoder: returns (clip_list, actual_action) where clip_list is
                        a list of CLIP_FRAMES RGB numpy arrays captured after settle.

        Args:
            action: list/array of [delta_s1, delta_s2]
        Returns:
            next_obs: single frame (2D) or list of frames (3D)
            actual_action: [actual_delta_s1, actual_delta_s2]
        """
        d1 = max(-C.ACTION_SCALE, min(C.ACTION_SCALE, action[0]))
        d2 = max(-C.ACTION_SCALE, min(C.ACTION_SCALE, action[1]))
        
        real_d1, real_d2 = self.move_servos(d1, d2)
        
        # Wait for servo vibration to settle
        time.sleep(C.SERVO_SETTLE_DELAY)
        
        if C.ENCODER_TYPE == '3d':
            # Collect CLIP_FRAMES frames from the ring buffer after settle
            next_obs = self.get_clip()
        else:
            next_obs = self.get_latest_frame()
        
        return next_obs, [real_d1, real_d2]

    def close(self):
        self._cam_running = False
        self._cam_thread.join(timeout=2.0)
        self.cap.release()
        self.controller.close()
