#!/usr/bin/env python3
import cv2
import argparse

from .controller import RobotInterface
from .. import config as C

def main():
    parser = argparse.ArgumentParser(description="Manual Control & Stream")
    parser.add_argument("--port", type=str, help="Serial port for servo controller")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--step", type=float, default=5.0, help="Step size in degrees")
    args = parser.parse_args()

    print("=== Robot Control & Stream ===")
    
    try:
        robot = RobotInterface(port=args.port, camera_index=args.camera)
    except Exception as e:
        print(f"Initialization Error: {e}")
        return

    print("\nControls:")
    print("  W / S : Servo 2 (Up / Down)")
    print("  A / D : Servo 1 (Left / Right)")
    print("  Q     : Quit")
    
    try:
        while True:
            # Capture frame
            frame_rgb = robot.get_frame()
            if frame_rgb is None:
                print("Failed to get frame")
                break
                
            # Convert to BGR for OpenCV display
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            # Add overlay text
            s1, s2 = robot.current_angles
            text = f"S1(L/R): {s1:.1f}  S2(U/D): {s2:.1f}"
            cv2.putText(frame_bgr, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Robot Stream", frame_bgr)
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            
            d1, d2 = 0.0, 0.0
            
            if key == ord('w'):
                d2 = args.step
            elif key == ord('s'):
                d2 = -args.step
            elif key == ord('a'):
                d1 = args.step
            elif key == ord('d'):
                d1 = -args.step
                
            if d1 != 0 or d2 != 0:
                robot.move_servos(d1, d2)
                
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        robot.close()
        cv2.destroyAllWindows()
        print("Closed.")

if __name__ == "__main__":
    main()
