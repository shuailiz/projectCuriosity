#!/usr/bin/env python3
import argparse
import time
import sys
import random
import cv2
import numpy as np
import torch

from project_curiosity.robot.visual_memory.trainer import VisualTrainer
from project_curiosity.robot.visual_memory.controller import RobotInterface
from project_curiosity.robot.visual_memory import config as C

# Exploration modes
MODE_MANUAL = 'manual'
MODE_RANDOM = 'random'
MODE_CURIOSITY = 'curiosity'

def main():
    parser = argparse.ArgumentParser(description="Visual-Motor Continuous Learning")
    parser.add_argument("--port", type=str, help="Serial port for servo controller")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--mode", type=str, default="manual",
                        choices=["manual", "random", "curiosity"],
                        help="Exploration mode: manual, random, or curiosity")
    parser.add_argument("--model", type=str, default="default",
                        help="Model name (creates new or resumes existing)")
    parser.add_argument("--list-models", action="store_true",
                        help="List available models and exit")
    args = parser.parse_args()

    # List models and exit
    if args.list_models:
        models = VisualTrainer.list_models()
        if models:
            print("Available models:")
            for m in models:
                print(f"  - {m}")
        else:
            print("No models found.")
        return

    print("=== Visual-Motor Continuous Learning System ===")
    
    # Initialize components
    try:
        robot = RobotInterface(port=args.port, camera_index=args.camera)
        trainer = VisualTrainer(model_name=args.model)
    except Exception as e:
        print(f"Initialization Error: {e}")
        return

    print("\nControls:")
    print("  'q'   - Quit (auto-saves checkpoint)")
    print("  'x'   - Force Sleep (Consolidation)")
    print("  'c'   - Save Model")
    print("  '1'   - Manual Mode")
    print("  '2'   - Random Exploration Mode")
    print("  '3'   - Curiosity Mode")
    print("  'w/s' - Servo 2 Up/Down (Manual)")
    print("  'a/d' - Servo 1 Left/Right (Manual)")
    print("  SPACE - No Op (Manual)")
    
    mode = args.mode
    
    # Initial frame
    current_frame = robot.get_frame()
    if current_frame is None:
        print("Failed to get initial frame.")
        return

    print(f"\nStarting in {mode.upper()} mode.")
    print(f"Wake/Sleep cycle: {C.WAKE_STEPS_PER_CYCLE} wake steps → SWS({C.SWS_STEPS}) → REM({C.REM_STEPS})")

    try:
        while True:
            # Display current view
            frame_bgr = cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR)
            
            # Overlay: mode, step, and cycle info
            s1, s2 = robot.current_angles
            overlay = (f"[{mode.upper()}] Step:{trainer.step_count} "
                       f"Wake:{trainer.wake_steps_in_cycle}/{C.WAKE_STEPS_PER_CYCLE} "
                       f"S1:{s1:.1f} S2:{s2:.1f}")
            cv2.putText(frame_bgr, overlay, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.imshow("Robot View", frame_bgr)
            
            # Key handling
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('x'):
                print("\nForcing sleep cycle...")
                stats = trainer.sleep_cycle()
                print(f"Cycle {stats['cycle']} complete.")
                continue
            elif key == ord('c'):
                print("\nSaving model...")
                trainer.save()
                continue
            elif key == ord('1'):
                mode = MODE_MANUAL
                print(f"\nSwitched to MANUAL mode.")
                continue
            elif key == ord('2'):
                mode = MODE_RANDOM
                print(f"\nSwitched to RANDOM mode.")
                continue
            elif key == ord('3'):
                mode = MODE_CURIOSITY
                print(f"\nSwitched to CURIOSITY mode.")
                continue
                
            # Determine action based on mode
            action = [0.0, 0.0]
            
            if mode == MODE_CURIOSITY:
                action = trainer.get_curiosity_action(current_frame, explore=True)
                time.sleep(0.1)
                
            elif mode == MODE_RANDOM:
                d1 = random.uniform(-C.ACTION_SCALE, C.ACTION_SCALE)
                d2 = random.uniform(-C.ACTION_SCALE, C.ACTION_SCALE)
                action = [d1, d2]
                time.sleep(0.1)
                
            else:  # MODE_MANUAL
                if key == ord('w'):
                    action[1] = 5.0   # S2 Up
                elif key == ord('s'):
                    action[1] = -5.0  # S2 Down
                elif key == ord('a'):
                    action[0] = 5.0   # S1 Left
                elif key == ord('d'):
                    action[0] = -5.0  # S1 Right
                elif key == 32:  # Space
                    pass  # No Op
                else:
                    if key == 255:  # No key pressed
                        continue

            # Execute Action
            next_frame, actual_action = robot.step(action)
            
            if next_frame is None:
                print("\nError capturing next frame")
                continue
                
            # Train (Wake phase)
            train_stats = trainer.train_step(current_frame, actual_action, next_frame)
            
            # Log
            log_parts = [f"Step {train_stats['step']}",
                         f"Fast:{train_stats['fast_loss']:.4f}"]
            if train_stats['slow_wake_loss'] is not None:
                sw = train_stats['slow_wake_loss']
                log_parts.append(f"SlowD:{sw['distill']:.4f} SlowR:{sw['raw']:.4f}")
            log_parts.append(f"Cur:{train_stats['curiosity_reward']:.4f}")
            if train_stats['policy_stats']:
                log_parts.append(f"PolR:{train_stats['policy_stats']['curiosity_reward']:.4f}")
            print(f"  {' | '.join(log_parts)}", end="\r")
            
            # Update state
            current_frame = next_frame
            
            # Auto-Sleep: WAKE → SWS → REM → WAKE
            if train_stats["should_sleep"]:
                print(f"\nAuto-Sleep (wake={train_stats['wake_steps']} steps)...")
                stats = trainer.sleep_cycle()
                print(f"Cycle {stats['cycle']} complete.")

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        # Auto-save on exit
        if trainer.step_count > 0:
            print("\nSaving model before exit...")
            trainer.save()
        robot.close()
        cv2.destroyAllWindows()
        print("System Shutdown.")

if __name__ == "__main__":
    main()
