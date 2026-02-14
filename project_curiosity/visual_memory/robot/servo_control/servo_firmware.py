import gc
import time
import sys
import select
from pimoroni import Button
from servo import ServoCluster, servo2040

"""
Stable Servo Control - Robust version without crash-prone status
"""

# Configuration
SERVO_EXTENT = 90.0
SERVO1_CHANNEL = servo2040.SERVO_1  # Channel 1
SERVO2_CHANNEL = servo2040.SERVO_3  # Channel 3
SERVO_CHANNELS = [SERVO1_CHANNEL, SERVO2_CHANNEL]

# Free up hardware resources
gc.collect()

# Create servo cluster
servos = ServoCluster(pio=0, sm=0, pins=SERVO_CHANNELS)
user_sw = Button(servo2040.USER_SW)

# Track servo positions more safely
current_angles = [0.0, 0.0]

def parse_command(command_str):
    try:
        parts = command_str.strip().split()
        if len(parts) == 0:
            return None, None
        
        cmd = parts[0].lower()
        
        if cmd == "stop":
            return "stop", None
        elif cmd == "angles":
            return "angles", None
        elif cmd in ["servo1", "servo2", "both"] and len(parts) == 2:
            angle = float(parts[1])
            angle = max(-SERVO_EXTENT, min(SERVO_EXTENT, angle))
            return cmd, angle
        elif cmd == "servos" and len(parts) == 3:
            angle1 = float(parts[1])
            angle2 = float(parts[2])
            angle1 = max(-SERVO_EXTENT, min(SERVO_EXTENT, angle1))
            angle2 = max(-SERVO_EXTENT, min(SERVO_EXTENT, angle2))
            return "servos", (angle1, angle2)
        else:
            return "unknown", None
    except:
        return "error", None

def set_servo_safe(servo_index, angle):
    try:
        if 0 <= servo_index < len(SERVO_CHANNELS):
            servos.value(servo_index, angle, load=False)
            current_angles[servo_index] = angle
            return True
    except:
        pass
    return False

print("Stable Servo Control Ready")
print(f"Servo 1 on channel {SERVO1_CHANNEL}, Servo 2 on channel {SERVO2_CHANNEL}")
print("Commands: servo1 <angle>, servo2 <angle>, both <angle>, servos <angle1> <angle2>, angles, stop")

while not user_sw.raw():
    try:
        # Non-blocking input check
        if select.select([sys.stdin], [], [], 0)[0]:
            command = sys.stdin.readline().strip()
            if not command:
                continue
                
            cmd_type, value = parse_command(command)
            
            if cmd_type == "servo1":
                if set_servo_safe(0, value):
                    servos.load()
                    print(f"S1:{value}")
                else:
                    print("Error")
            
            elif cmd_type == "servo2":
                if set_servo_safe(1, value):
                    servos.load()
                    print(f"S2:{value}")
                else:
                    print("Error")
            
            elif cmd_type == "both":
                success1 = set_servo_safe(0, value)
                success2 = set_servo_safe(1, value)
                if success1 and success2:
                    servos.load()
                    print(f"Both:{value}")
                else:
                    print("Error")
            
            elif cmd_type == "servos":
                angle1, angle2 = value
                success1 = set_servo_safe(0, angle1)
                success2 = set_servo_safe(1, angle2)
                if success1 and success2:
                    servos.load()
                    print(f"S1:{angle1} S2:{angle2}")
                else:
                    print("Error")
            
            elif cmd_type == "angles":
                try:
                    # Simple, safe angle reporting
                    print(f"{current_angles[0]},{current_angles[1]}")
                except:
                    print("0.0,0.0")
            
            elif cmd_type == "stop":
                try:
                    servos.disable_all()
                    print("Stopped")
                except:
                    print("Stop error")
            
            else:
                print("Unknown command")
        else:
            # No input available, small delay to prevent busy waiting
            time.sleep_ms(10)
            
    except Exception as e:
        print(f"Input error: {e}")
        # Don't crash, just continue

# Cleanup
try:
    servos.disable_all()
    print("Cleanup done")
except:
    pass
