# Visual-Motor Continuous Learning

A dual-network continuous learning system for a 2-servo robot with visual input, inspired by the **Complementary Learning Systems (CLS)** theory of hippocampal-cortical memory consolidation. The system learns a visual world model — predicting how the visual scene changes in response to motor actions — through a biologically-motivated wake/sleep training cycle.

## Core Idea

The brain doesn't learn everything the same way. The hippocampus rapidly encodes new experiences, while the neocortex slowly integrates them into stable, generalizable knowledge — primarily during sleep. This system mirrors that:

- A **Fast Learner** (hippocampus) adapts instantly to new sensory-motor transitions.
- A **Slow Learner** (cortex) consolidates knowledge gradually through distillation and replay.
- A **Wake/Sleep cycle** orchestrates when and how each network learns.
- A **Curiosity Policy** drives exploration toward states where the world model is most uncertain.

## Architecture

```
Camera Frame (224x224 RGB)
        │
        ▼
┌─────────────────┐
│  Visual Encoder  │  Frozen ResNet50 backbone → 2048-d → Linear → 512-d
│  (Perception)    │  Pretrained on ImageNet, not fine-tuned
└────────┬────────┘
         │ z_t (512-d state embedding)
         │
         ├──────────────────────────────────────────┐
         ▼                                          ▼
┌──────────────────┐                    ┌──────────────────────┐
│  Fast Learner     │                    │  Slow Learner         │
│  (Hippocampus)    │                    │  (Cortex)             │
│                   │                    │                       │
│  2-layer MLP      │                    │  3-layer MLP          │
│  hidden: 256      │                    │  hidden: 1024         │
│  LR: 1e-3         │                    │  Wake LR: 1e-5        │
│                   │                    │  Sleep LR: 5e-4       │
│  Input: [z_t, a]  │                    │  Input: [z_t, a]      │
│  Output: z_{t+1}  │                    │  Output: z_{t+1}      │
└──────────────────┘                    └──────────────────────┘
         │                                          │
         └──────────┬───────────────────────────────┘
                    ▼
          ┌──────────────────┐
          │ Curiosity Policy  │  MLP: z_t → action (tanh scaled)
          │ (Exploration)     │  Trained to maximize prediction error
          └──────────────────┘
```

**World Model**: Both learners predict the next state embedding given the current state and action: `F(z_t, a_t) → z_{t+1}`. The prediction error serves as the curiosity reward signal.

## Wake/Sleep Training Cycle

The system alternates between Wake and Sleep phases in a continuous loop:

```
  ┌─────────────────────────────────────────────────────────┐
  │                                                         │
  ▼                                                         │
WAKE (500 steps)  →  SWS (200 steps)  →  REM (200 steps)  →  Tether (10 steps)
  │                    │                    │                    │
  │ Fast: raw targets  │ Fast: frozen       │ Fast: frozen       │ Fast: distill → Slow
  │ Slow: distill+raw  │ Slow: distill      │ Slow: rollout      │ Slow: frozen
  │                    │                    │                    │
  └────────────────────┴────────────────────┴────────────────────┘
```

### Wake Phase

The robot interacts with the environment. Both networks learn, but with different objectives:

**Fast Learner** trains directly on raw sensory transitions (high LR):
```
L_fast = ||F_H(z_t, a_t) - z_{t+1}||^2
```

**Slow Learner** trains on a weighted combination (very low LR):
```
L_wake_slow = w_distill * L_distill + w_raw * L_raw

L_distill = ||F_C(z_t, a_t) - stopgrad(F_H(z_t, a_t))||^2    (match Fast)
L_raw     = ||F_C(z_t, a_t) - stopgrad(z_{t+1})||^2           (grounded in reality)
```

The distillation loss is primary (weight 1.0); the raw loss is conservative (weight 0.1). The small LR provides implicit regularization, preventing catastrophic forgetting.

### SWS Phase (Slow-Wave Sleep — Consolidation)

Fast is frozen. Slow trains strongly on replay buffer data via distillation from Fast:
```
L_SWS = α * ||F_C(z_t, a_t) - stopgrad(F_H(z_t, a_t))||^2
```

This is the main knowledge transfer phase — Fast's recent experience is consolidated into Slow's stable representation.

### REM Phase (Dreaming — Stabilization)

Fast is frozen. Slow trains on multi-step rollout consistency using trajectory windows from the replay buffer:
```
L_REM = γ * Σ_{k=1}^{K} ||z_hat_k - stopgrad(z_k)||^2

where z_hat_k = F_C(z_hat_{k-1}, a_{k-1})   (autoregressive rollout)
```

This ensures Slow's predictions remain coherent over multiple time steps, not just single transitions. Rollout horizon K=3 by default.

### Post-Sleep Tether

After sleep, Fast is briefly trained to match Slow's outputs via output-level distillation. This prevents Fast from drifting too far from the consolidated knowledge:
```
L_tether = ||F_H(z_t, a_t) - stopgrad(F_C(z_t, a_t))||^2
```

Note: output-level distillation is used because Fast and Slow have different architectures (different hidden dimensions), so weight-level EMA is not possible.

## Curiosity-Driven Exploration

The Curiosity Policy is a separate MLP that maps state embeddings to motor actions. It is trained to **maximize** the world model's prediction error — seeking out states where the model is most surprised:

```
Policy: π(z_t) → a_t
Loss:   -||F_H(z_t, π(z_t)) - z_{t+1}||^2    (maximize prediction error)
```

During warmup (first 50 steps), the robot explores randomly. After that, the policy generates actions with added Gaussian noise for exploration.

## File Structure

```
project_curiosity/
├── dual_network_model.py              # Shared: ContinuousFast/SlowLearner
└── visual_memory/                     # This module
    ├── config.py                      # All hyperparameters
    ├── encoder.py                     # VisualEncoder (frozen ResNet50 → 512-d)
    ├── curiosity_policy.py            # CuriosityPolicy (state → action)
    ├── trainer.py                     # VisualTrainer (orchestrates everything)
    ├── README.md                      # This file
    └── robot/                         # Hardware interface
        ├── controller.py              # RobotInterface (camera + servos)
        ├── servo_control/             # Low-level servo communication
        └── control_stream.py          # Manual control & video stream

run_visual_learning.py                 # Main entry point

models/                                # Saved models (one folder per model)
└── <model_name>/
    ├── config.json                    # Frozen config snapshot at creation
    ├── checkpoint.pt                  # Weights + optimizer states + counters
    ├── replay.pt                      # Replay buffer (embeddings, no raw frames)
    ├── training.log                   # Append-only CSV training log
    └── frames/                        # Debug frames (--save-frames)
        ├── index.csv                  # Step, action, timestamp index
        ├── step_000000_before.jpg     # Pre-action frame
        └── step_000000_after.jpg      # Post-action frame
```

## Usage

### Prerequisites
- A 2-servo robot connected via serial (USB)
- A webcam
- `torch`, `torchvision`, `opencv-python`

### Running

```bash
# Start a new model
python run_visual_learning.py --model my_experiment --port /dev/tty.usbmodem1234

# Resume an existing model (auto-detects saved state)
python run_visual_learning.py --model my_experiment --port /dev/tty.usbmodem1234

# List all saved models
python run_visual_learning.py --list-models

# Exploration modes
python run_visual_learning.py --model my_experiment --mode curiosity   # auto-explore
python run_visual_learning.py --model my_experiment --mode random      # random actions
python run_visual_learning.py --model my_experiment --mode manual      # keyboard control

# Save raw frames for debugging (before/after each action as JPEG)
python run_visual_learning.py --model my_experiment --save-frames
```

### Controls
| Key | Action |
|-----|--------|
| `w/s` | Servo 2 Up/Down (manual mode) |
| `a/d` | Servo 1 Left/Right (manual mode) |
| `Space` | No-Op |
| `1/2/3` | Switch to Manual/Random/Curiosity mode |
| `x` | Force Sleep cycle |
| `c` | Save model |
| `q` | Quit (auto-saves) |

## Configuration

All hyperparameters are in `config.py`. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FAST_LR` | 1e-3 | Fast learner learning rate |
| `SLOW_WAKE_LR` | 1e-5 | Slow learner LR during wake |
| `SLOW_SLEEP_LR` | 5e-4 | Slow learner LR during sleep |
| `WAKE_STEPS_PER_CYCLE` | 500 | Wake steps before triggering sleep |
| `SLOW_WAKE_W_DISTILL` | 1.0 | Weight for distillation loss during wake |
| `SLOW_WAKE_W_RAW` | 0.1 | Weight for raw transition loss during wake |
| `SWS_STEPS` | 200 | Gradient steps in SWS phase |
| `REM_STEPS` | 200 | Gradient steps in REM phase |
| `REM_K` | 3 | Rollout horizon for REM dreaming |
| `TETHER_STEPS` | 10 | Post-sleep Fast tether steps |
| `REPLAY_BUFFER_SIZE` | 5000 | Max experiences stored |
| `ENCODED_DIM` | 512 | Visual embedding dimension |
| `SAVE_FRAMES` | False | Save raw RGB frames as JPEGs for debugging |
