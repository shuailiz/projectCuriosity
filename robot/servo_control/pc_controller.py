import serial
import serial.tools.list_ports
import time
import sys

"""
Fast Stable Controller - No status command to avoid crashes
"""

class FastStableController:
    def __init__(self, port, baudrate=115200):
        try:
            self.serial = serial.Serial(port, baudrate, timeout=0.1)
            time.sleep(1)
            print(f"Connected to {port}")
            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()
        except serial.SerialException as e:
            print(f"Connection error: {e}")
            sys.exit(1)
    
    def send(self, command):
        """Send command immediately"""
        try:
            self.serial.write(f"{command}\n".encode())
            self.serial.flush()
            return True
        except:
            return False
    
    def get_angles(self):
        """Get current servo angles safely"""
        try:
            # Send angles command
            self.serial.write(b"angles\n")
            self.serial.flush()
            
            # Wait a bit for processing
            time.sleep(0.2)
            
            # Read multiple lines to find the one with angles
            # The board sends debug output first, so we need to filter through it
            for attempt in range(10):
                self.serial.timeout = 0.5
                raw_data = self.serial.readline()
                
                response = raw_data.decode('utf-8', errors='ignore').strip()
                if response:
                    # Check if this line looks like angle data (two floats separated by comma)
                    if ',' in response:
                        parts = response.split(',')
                        if len(parts) == 2:
                            try:
                                return float(parts[0]), float(parts[1])
                            except ValueError:
                                # Not floats, maybe debug text containing a comma
                                pass
                
                time.sleep(0.05)
            
            return None, None
        except Exception as e:
            print(f"Angle read error: {e}")
            return None, None
    
    def close(self):
        if hasattr(self, 'serial') and self.serial.is_open:
            self.serial.close()

def list_ports():
    ports = serial.tools.list_ports.comports()
    if not ports:
        return []
    return ports

def select_port():
    ports = list_ports()
    if not ports:
        print("No serial ports found!")
        return None
    
    print("\nAvailable ports:")
    for i, p in enumerate(ports):
        print(f"{i+1}: {p.device} - {p.description}")
    
    while True:
        try:
            selection = input("\nSelect port (number) or enter path manually: ").strip()
            if not selection:
                continue
                
            if selection.isdigit():
                idx = int(selection) - 1
                if 0 <= idx < len(ports):
                    return ports[idx].device
            else:
                return selection
        except ValueError:
            pass

def main():
    print("=== Fast Stable Servo Controller ===")
    
    if len(sys.argv) > 1:
        port = sys.argv[1]
    else:
        port = select_port()
        if not port:
            sys.exit(1)
    
    controller = FastStableController(port)
    
    try:
        print("Commands: servo1 <angle>, servo2 <angle>, both <angle>, servos <a1> <a2>, angles, stop, quit")
        print("Use 'angles' to read current servo positions")
        
        while True:
            command = input("\nFast> ").strip()
            
            if command.lower() == 'quit':
                break
            elif command == '':
                continue
            elif command.lower() == 'angles':
                angle1, angle2 = controller.get_angles()
                if angle1 is not None and angle2 is not None:
                    print(f"Servo 1: {angle1}°, Servo 2: {angle2}°")
                else:
                    print("Could not read angles")
                continue
            elif command.lower().startswith('status'):
                print("Use 'angles' command instead")
                continue
            
            if controller.send(command):
                print(f"✓ {command}")
            else:
                print(f"✗ {command}")
                
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        controller.close()

if __name__ == "__main__":
    main()
