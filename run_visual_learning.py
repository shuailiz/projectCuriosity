#!/usr/bin/env python3
import argparse
import time
import threading
import queue
import random
import cv2
import numpy as np
import torch

from project_curiosity.visual_memory.trainer import VisualTrainer
from project_curiosity.visual_memory.robot.controller import RobotInterface
from project_curiosity.visual_memory import config as C

# Exploration modes
MODE_MANUAL = 'manual'
MODE_RANDOM = 'random'
MODE_CURIOSITY = 'curiosity'

# Target action frequency (Hz)
ACTION_HZ = 2.0
ACTION_PERIOD = 1.0 / ACTION_HZ


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

class SharedState:
    """Thread-safe shared state between action loop, worker, and display."""
    def __init__(self):
        self.lock = threading.Lock()
        self.quit = False
        self.last_key = None
        # Training stats updated by worker thread, read by display thread
        self.step_count = 0
        self.wake_steps = 0
        self.actual_hz = 0.0
        self.train_log = ""
        self.mode = MODE_RANDOM
        # Latest embedding from worker (used by curiosity action in action loop)
        self.latest_emb = None          # Tensor (ENCODED_DIM,) on CPU
        self.latest_joints = [0.0, 0.0]


# ---------------------------------------------------------------------------
# Display thread — pulls live frames directly from robot ring buffer
# ---------------------------------------------------------------------------

def display_thread_fn(robot: RobotInterface, state: SharedState):
    """Refreshes the OpenCV window at ~30Hz from the robot's live ring buffer."""
    cv2.namedWindow("Robot View", cv2.WINDOW_NORMAL)
    while not state.quit:
        frame = robot.get_latest_frame()
        if frame is not None:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            with state.lock:
                mode = state.mode
                step = state.step_count
                wake = state.wake_steps
                hz = state.actual_hz
                log = state.train_log
                s1, s2 = state.latest_joints

            cv2.putText(frame_bgr,
                        f"[{mode.upper()}] Step:{step} Wake:{wake}/{C.WAKE_STEPS_PER_CYCLE}  {hz:.2f}Hz",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.putText(frame_bgr,
                        f"S1:{s1:.1f} S2:{s2:.1f}  ENC:{C.ENCODER_TYPE.upper()}",
                        (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            if log:
                cv2.putText(frame_bgr, log,
                            (10, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
            cv2.imshow("Robot View", frame_bgr)

        key = cv2.waitKey(33) & 0xFF  # ~30Hz
        if key != 255:
            with state.lock:
                state.last_key = key
            if key == ord('q'):
                with state.lock:
                    state.quit = True
                break

    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Worker thread — encode + train (decoupled from action loop)
# ---------------------------------------------------------------------------

def worker_thread_fn(trainer: VisualTrainer, work_queue: queue.Queue,
                     state: SharedState):
    """
    Consumes (obs, action, next_obs, joints, commanded) tuples from the queue,
    runs encode + train_step, and updates shared state.

    Runs independently of the action loop so encode latency (~600ms for R3D-18)
    does not block servo commands.
    """
    step_times = []
    last_t = None

    while not state.quit:
        try:
            item = work_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        if item is None:  # sentinel — shut down
            break

        obs, actual_action, next_obs, joints_before, commanded = item

        try:
            train_stats = trainer.train_step(
                obs, actual_action, next_obs, joints_before, commanded)
        except Exception as e:
            print(f"\n[worker] train_step error: {e}")
            work_queue.task_done()
            continue

        # Update shared state
        now = time.time()
        if last_t is not None:
            step_times.append(now - last_t)
            if len(step_times) > 10:
                step_times.pop(0)
        last_t = now

        hz = 1.0 / (sum(step_times) / len(step_times)) if step_times else 0.0

        log_parts = [f"Fast:{train_stats['fast_loss']:.4f}"]
        if train_stats['slow_wake_loss'] is not None:
            sw = train_stats['slow_wake_loss']
            log_parts.append(f"SlD:{sw['distill']:.4f}")
        log_parts.append(f"Cur:{train_stats['curiosity_reward']:.4f}")
        if train_stats['policy_stats']:
            log_parts.append(f"Pol:{train_stats['policy_stats']['curiosity_reward']:.4f}")
        log_str = " | ".join(log_parts)
        print(f"  Step {train_stats['step']} | {log_str}", end="\r")

        with state.lock:
            state.step_count = train_stats['step']
            state.wake_steps = train_stats['wake_steps']
            state.actual_hz = hz
            state.latest_emb = train_stats['next_state_emb']
            state.train_log = log_str

        # Auto-sleep (blocks worker, not action loop)
        if train_stats["should_sleep"]:
            print(f"\nAuto-Sleep (wake={train_stats['wake_steps']} steps)...")
            sleep_stats = trainer.sleep_cycle()
            print(f"Cycle {sleep_stats['cycle']} complete.")

        work_queue.task_done()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_obs(robot):
    """Snapshot current observation without blocking."""
    if C.ENCODER_TYPE == '3d':
        return robot.get_clip()
    return robot.get_latest_frame()


def main():
    parser = argparse.ArgumentParser(description="Visual-Motor Continuous Learning")
    parser.add_argument("--port", type=str, help="Serial port for servo controller")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--mode", type=str, default="random",
                        choices=["manual", "random", "curiosity"],
                        help="Exploration mode")
    parser.add_argument("--model", type=str, default="default",
                        help="Model name (creates new or resumes existing)")
    parser.add_argument("--list-models", action="store_true",
                        help="List available models and exit")
    parser.add_argument("--save-frames", action="store_true",
                        help="Save raw RGB frames as JPEGs for debugging")
    args = parser.parse_args()

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
    print(f"Encoder: {C.ENCODER_TYPE.upper()}  |  Action rate: {ACTION_HZ}Hz  |  "
          f"Settle: {C.SERVO_SETTLE_DELAY}s")

    try:
        robot = RobotInterface(port=args.port, camera_index=args.camera)
        trainer = VisualTrainer(model_name=args.model)
        if args.save_frames:
            trainer.save_frames = True
            print("Frame saving ENABLED → models/{}/frames/".format(args.model))
    except Exception as e:
        print(f"Initialization Error: {e}")
        return

    print("\nControls:")
    print("  'q'         - Quit")
    print("  'x'         - Force Sleep")
    print("  'c'         - Save Model")
    print("  '1/2/3'     - Manual / Random / Curiosity mode")
    print("  'w/s/a/d'   - Manual servo control")
    print("  SPACE       - No-op step (manual)")

    state = SharedState()
    state.mode = args.mode
    state.latest_joints = list(robot.current_angles)

    # Work queue: action loop enqueues, worker dequeues
    work_queue = queue.Queue(maxsize=2)  # small cap — drop old work if falling behind

    # Start display thread
    disp_thread = threading.Thread(
        target=display_thread_fn, args=(robot, state), daemon=True)
    disp_thread.start()

    # Start worker thread
    wrk_thread = threading.Thread(
        target=worker_thread_fn, args=(trainer, work_queue, state), daemon=True)
    wrk_thread.start()

    print(f"\nStarting in {args.mode.upper()} mode.")
    print(f"Wake/Sleep cycle: {C.WAKE_STEPS_PER_CYCLE} steps → SWS({C.SWS_STEPS}) → REM({C.REM_STEPS})")

    # Action loop timing
    action_step_times = []
    last_action_t = None
    current_obs = get_obs(robot)

    try:
        while not state.quit:
            step_start = time.time()

            # --- Read key ---
            with state.lock:
                key = state.last_key
                state.last_key = None
                mode = state.mode

            # --- Control key handling ---
            if key == ord('x'):
                print("\nForcing sleep cycle...")
                # Drain queue first so worker doesn't conflict
                work_queue.join()
                sleep_stats = trainer.sleep_cycle()
                print(f"Cycle {sleep_stats['cycle']} complete.")
                continue
            elif key == ord('c'):
                work_queue.join()
                print("\nSaving model...")
                trainer.save()
                continue
            elif key == ord('1'):
                with state.lock:
                    state.mode = MODE_MANUAL
                print("\nSwitched to MANUAL.")
                continue
            elif key == ord('2'):
                with state.lock:
                    state.mode = MODE_RANDOM
                print("\nSwitched to RANDOM.")
                continue
            elif key == ord('3'):
                with state.lock:
                    state.mode = MODE_CURIOSITY
                print("\nSwitched to CURIOSITY.")
                continue

            # --- Determine action ---
            action = [0.0, 0.0]
            skip_step = False

            if mode == MODE_CURIOSITY:
                with state.lock:
                    emb = state.latest_emb
                    joints = list(state.latest_joints)
                if emb is not None:
                    action = trainer.model.policy.get_action(emb, joints, explore=True)
                else:
                    action = [random.uniform(-C.ACTION_SCALE, C.ACTION_SCALE),
                              random.uniform(-C.ACTION_SCALE, C.ACTION_SCALE)]

            elif mode == MODE_RANDOM:
                action = [random.uniform(-C.ACTION_SCALE, C.ACTION_SCALE),
                          random.uniform(-C.ACTION_SCALE, C.ACTION_SCALE)]

            else:  # MANUAL
                if key == ord('w'):
                    action[1] = 5.0
                elif key == ord('s'):
                    action[1] = -5.0
                elif key == ord('a'):
                    action[0] = 5.0
                elif key == ord('d'):
                    action[0] = -5.0
                elif key == 32:
                    pass  # no-op step
                else:
                    skip_step = True

            # --- Execute action (servo + settle + clip snapshot) ---
            if not skip_step:
                joints_before = list(robot.current_angles)
                commanded = list(action)

                next_obs, actual_action = robot.step(action)

                with state.lock:
                    state.latest_joints = list(robot.current_angles)

                if next_obs is None:
                    print("\nError capturing observation")
                    continue

                # Enqueue for async encode+train; drop if worker is backed up
                try:
                    work_queue.put_nowait(
                        (current_obs, actual_action, next_obs, joints_before, commanded))
                except queue.Full:
                    pass  # worker still busy — skip this training step

                current_obs = next_obs

                # Track action loop Hz
                now = time.time()
                if last_action_t is not None:
                    action_step_times.append(now - last_action_t)
                    if len(action_step_times) > 10:
                        action_step_times.pop(0)
                last_action_t = now

            # --- Pace to ACTION_HZ ---
            elapsed = time.time() - step_start
            remaining = ACTION_PERIOD - elapsed
            if remaining > 0 and not skip_step:
                time.sleep(remaining)
            elif skip_step:
                time.sleep(0.033)  # ~30Hz poll in manual/idle

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        state.quit = True
        # Drain queue then send sentinel so worker can exit cleanly
        while not work_queue.empty():
            try:
                work_queue.get_nowait()
                work_queue.task_done()
            except queue.Empty:
                break
        try:
            work_queue.put_nowait(None)
        except queue.Full:
            pass
        wrk_thread.join(timeout=5.0)
        disp_thread.join(timeout=2.0)
        if trainer.step_count > 0:
            print("\nSaving model before exit...")
            trainer.save()
        robot.close()
        cv2.destroyAllWindows()
        print("System Shutdown.")


if __name__ == "__main__":
    main()
