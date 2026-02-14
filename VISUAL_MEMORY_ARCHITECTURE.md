# Visual-Motor Learning Architecture Diagram

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│              VISUAL-MOTOR CONTINUOUS LEARNING SYSTEM             │
│            (Hippocampus-Cortex + Curiosity-Driven)              │
└─────────────────────────────────────────────────────────────────┘

                    ┌───────────────┐
                    │   Camera      │
                    │   (Webcam)    │
                    └───────┬───────┘
                            │ 224×224 RGB
                            ▼
                    ┌───────────────┐
                    │ Visual Encoder│  Frozen ResNet50
                    │ (Perception)  │  2048-d → Linear → 512-d
                    └───────┬───────┘
                            │ z_t (512-d embedding)
                            │
         ┌──────────────────┼──────────────────┐
         ▼                  ▼                  ▼
┌─────────────────┐ ┌──────────────┐ ┌─────────────────┐
│  Fast Learner   │ │   Replay     │ │  Slow Learner   │
│  (Hippocampus)  │ │   Buffer     │ │  (Cortex)       │
│                 │ │  (5000 exp)  │ │                 │
│  2-layer MLP    │ │              │ │  3-layer MLP    │
│  hidden: 256    │ │  Stores:     │ │  hidden: 1024   │
│  LR: 1e-3      │ │  z_t, a_t,   │ │  Wake LR: 1e-5  │
│                 │ │  z_{t+1}     │ │  Sleep LR: 5e-4 │
│  [z_t, a_t]    │ │              │ │                 │
│     → z_{t+1}  │ │              │ │  [z_t, a_t]     │
│                 │ │              │ │     → z_{t+1}   │
└────────┬────────┘ └──────────────┘ └────────┬────────┘
         │                                    │
         └──────────────┬─────────────────────┘
                        │ prediction error
                        ▼
              ┌──────────────────┐
              │ Curiosity Policy │  MLP: z_t → [d1, d2]
              │  (Exploration)   │  tanh × 10° scale
              │  hidden: 256     │  + Gaussian noise
              └────────┬─────────┘
                       │ action
                       ▼
              ┌──────────────────┐
              │  2-Servo Robot   │  S1: Left/Right (-90°..90°)
              │  (Actuator)      │  S2: Up/Down (-40°..40°)
              └──────────────────┘
```

## Wake/Sleep Cycle

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CONTINUOUS TRAINING LOOP                         │
│                                                                         │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐        │
│   │          │    │          │    │          │    │          │        │
│   │   WAKE   │───▶│   SWS    │───▶│   REM    │───▶│  TETHER  │───┐   │
│   │  500 stp │    │  200 stp │    │  200 stp │    │  10 stp  │   │   │
│   │          │    │          │    │          │    │          │   │   │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘   │   │
│        ▲                                                          │   │
│        │                    Auto-save checkpoint                  │   │
│        └─────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Phase 1: Wake (Interaction)

```
Robot moves (manual / random / curiosity)
    │
    ▼
┌─────────────────────┐
│  Capture Frame      │
│  (Camera → RGB)     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Encode Frame       │
│  ResNet50 → z_t     │
│  (512-d embedding)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Execute Action     │
│  a_t = [d1, d2]     │
│  (servo deltas)     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Encode Next Frame  │
│  ResNet50 → z_{t+1} │
└──────────┬──────────┘
           │
           ├─────────────────────────────────┐
           ▼                                 ▼
┌─────────────────────┐          ┌─────────────────────┐
│  Train Fast Learner │          │  Store Experience    │
│  (High LR: 1e-3)   │          │  in Replay Buffer    │
│                     │          │                      │
│  L = ||F_H(z,a)    │          │  {z_t, a_t, z_{t+1}} │
│       - z_{t+1}||² │          │                      │
└──────────┬──────────┘          └──────────────────────┘
           │
           ▼
┌─────────────────────┐
│  Train Slow Learner │  (every SLOW_WAKE_UPDATE_INTERVAL steps)
│  (Low LR: 1e-5)    │
│                     │
│  L = 1.0 · L_dist  │  L_dist = ||F_C(z,a) - sg(F_H(z,a))||²
│    + 0.1 · L_raw   │  L_raw  = ||F_C(z,a) - sg(z_{t+1})||²
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Compute Curiosity  │
│  reward = ||pred    │
│         - actual||² │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Train Policy       │  (every 4 steps)
│  Maximize reward    │
│  (seek surprise)    │
└─────────────────────┘

After 500 wake steps → trigger Sleep
```

## Phase 2: SWS (Slow-Wave Sleep — Consolidation)

```
┌─────────────────────────────────────────────────────────┐
│  Fast Learner: FROZEN (no gradient updates)             │
│  Slow Learner: ACTIVE (Sleep LR: 5e-4)                 │
└─────────────────────────────────────────────────────────┘

For 200 gradient steps:
    │
    ▼
┌─────────────────────┐
│  Sample Batch       │
│  from Replay Buffer │
│  (32 transitions)   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐     ┌─────────────────────┐
│  Fast Learner       │     │  Slow Learner       │
│  F_H(z_t, a_t)     │     │  F_C(z_t, a_t)      │
│  (frozen, no grad)  │     │  (trainable)        │
└──────────┬──────────┘     └──────────┬──────────┘
           │                           │
           └───────────┬───────────────┘
                       │
                       ▼
              ┌────────────────┐
              │  Dynamics      │
              │  Distillation  │
              │                │
              │  L_dyn = α ·   │
              │  ||F_C - sg(   │
              │    F_H)||²     │
              │                │
              │  α = 1.0       │
              └────────┬───────┘
                       │
                       ▼
              ┌────────────────┐
              │  Backprop to   │
              │  Slow Learner  │
              │  (LR: 5e-4)   │
              └────────────────┘

Purpose: Transfer Fast's recent knowledge → Slow's stable memory
```

## Phase 3: REM (Dreaming — Multi-Step Stabilization)

```
┌─────────────────────────────────────────────────────────┐
│  Fast Learner: FROZEN                                   │
│  Slow Learner: ACTIVE (Sleep LR: 5e-4)                 │
└─────────────────────────────────────────────────────────┘

For 200 gradient steps:
    │
    ▼
┌─────────────────────┐
│  Sample Trajectory  │
│  Window (K+1 = 4    │
│  consecutive steps) │
│  from Replay Buffer │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│  Multi-Step Rollout (Autoregressive)                    │
│                                                         │
│  z_hat_0 = z_0  (ground truth start)                    │
│      │                                                  │
│      ▼                                                  │
│  z_hat_1 = F_C(z_hat_0, a_0)  ──compare──▶ z_1         │
│      │                                                  │
│      ▼                                                  │
│  z_hat_2 = F_C(z_hat_1, a_1)  ──compare──▶ z_2         │
│      │                                                  │
│      ▼                                                  │
│  z_hat_3 = F_C(z_hat_2, a_2)  ──compare──▶ z_3         │
│                                                         │
│  L_ms = γ · Σ ||z_hat_k - sg(z_k)||²                   │
│  γ = 0.5                                                │
└─────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────┐
│  Backprop to        │
│  Slow Learner       │
│  (through rollout   │
│   chain)            │
└─────────────────────┘

Purpose: Ensure Slow's predictions stay coherent over time,
         not just single-step accuracy
```

## Phase 4: Post-Sleep Tether

```
┌─────────────────────────────────────────────────────────┐
│  Fast Learner: ACTIVE (Tether LR: 1e-3)                │
│  Slow Learner: FROZEN                                   │
└─────────────────────────────────────────────────────────┘

For 10 gradient steps:
    │
    ▼
┌─────────────────────┐
│  Sample Batch       │
│  from Replay Buffer │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐     ┌─────────────────────┐
│  Slow Learner       │     │  Fast Learner       │
│  F_C(z_t, a_t)      │     │  F_H(z_t, a_t)      │
│  (frozen, target)   │     │  (trainable)        │
└──────────┬──────────┘     └──────────┬──────────┘
           │                           │
           └───────────┬───────────────┘
                       │
                       ▼
              ┌────────────────┐
              │  Output-Level  │
              │  Distillation  │
              │                │
              │  L_tether =    │
              │  ||F_H - sg(   │
              │    F_C)||²     │
              └────────┬───────┘
                       │
                       ▼
              ┌────────────────┐
              │  Backprop to   │
              │  Fast Learner  │
              └────────────────┘

Purpose: Prevent Fast from drifting too far from consolidated Slow.
Note: Output-level distillation because architectures differ
      (Fast: 256 hidden, Slow: 1024 hidden — no weight EMA possible)
```

## Network Architecture Details

### Visual Encoder (Frozen)

```
Input: 224×224×3 RGB Image
    │
    ▼
┌─────────────────────┐
│  ResNet50 Backbone   │  Pretrained on ImageNet
│  (frozen weights)    │  No fine-tuning
│                      │
│  Conv layers...      │
│  → AvgPool           │
│  → 2048-d features   │
└──────────┬───────────┘
           │
           ▼
┌─────────────────────┐
│  Linear Projection  │
│  2048 → 512         │
└──────────┬──────────┘
           │
           ▼
    z_t ∈ ℝ^512
```

### Fast Learner (Hippocampus)

```
Input: [z_t, a_t] = [512 + 2] = 514-d
    │
    ▼
┌─────────────────────┐
│  FC1: 514 → 256     │
│  + ReLU             │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  FC2: 256 → 512     │
│  (predicted z_{t+1})│
└─────────────────────┘

Parameters: ~197K
Learning Rate: 1e-3
Role: Rapid adaptation to new transitions
```

### Slow Learner (Cortex)

```
Input: [z_t, a_t] = [512 + 2] = 514-d
    │
    ▼
┌─────────────────────┐
│  FC1: 514 → 1024    │
│  + ReLU             │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  FC2: 1024 → 1024   │
│  + ReLU             │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  FC3: 1024 → 512    │
│  (predicted z_{t+1})│
└─────────────────────┘

Parameters: ~1.6M
Learning Rate: 1e-5 (wake) / 5e-4 (sleep)
Role: Stable, consolidated world model
```

### Curiosity Policy

```
Input: z_t = 512-d
    │
    ▼
┌─────────────────────┐
│  FC1: 512 → 256     │
│  + ReLU             │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  FC2: 256 → 256     │
│  + ReLU             │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  FC_out: 256 → 2    │
│  + tanh × 10.0      │
└──────────┬──────────┘
           │
           ▼
    a_t = [d1, d2] ∈ [-10°, 10°]
    + Gaussian noise (σ = 0.3 × 10°)

Parameters: ~197K
Trained to MAXIMIZE prediction error (seek novelty)
```

## Loss Functions Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                         LOSS LANDSCAPE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  WAKE PHASE                                                     │
│  ──────────                                                     │
│  L_fast     = ||F_H(z,a) - z_{t+1}||²          (raw targets)   │
│  L_slow     = 1.0·||F_C - sg(F_H)||²           (distill)       │
│             + 0.1·||F_C - sg(z_{t+1})||²        (raw)           │
│  L_policy   = -||F_H(z,π(z)) - z_{t+1}||²      (max surprise)  │
│                                                                 │
│  SWS PHASE                                                      │
│  ─────────                                                      │
│  L_SWS      = 1.0·||F_C(z,a) - sg(F_H(z,a))||² (distill)      │
│                                                                 │
│  REM PHASE                                                      │
│  ─────────                                                      │
│  L_REM      = 0.5·Σ_k ||z_hat_k - sg(z_k)||²   (rollout)      │
│               z_hat_k = F_C(z_hat_{k-1}, a_{k-1})              │
│                                                                 │
│  TETHER PHASE                                                   │
│  ────────────                                                   │
│  L_tether   = ||F_H(z,a) - sg(F_C(z,a))||²     (Fast → Slow)  │
│                                                                 │
│  sg() = stop gradient (treat as fixed target)                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Key Parameters

| Parameter | Fast Learner | Slow Learner | Ratio |
|-----------|-------------|-------------|-------|
| Hidden Dim | 256 | 1024 | 0.25x |
| Layers | 2 | 3 | 0.67x |
| Parameters | ~197K | ~1.6M | 0.12x |
| Wake LR | 1e-3 | 1e-5 | 100x |
| Sleep LR | — (frozen) | 5e-4 | — |
| Update Freq | Every step | Every step (wake) | 1x |

| Phase | Steps | Who Learns | What |
|-------|-------|-----------|------|
| Wake | 500 | Fast + Slow | Raw transitions + distillation |
| SWS | 200 | Slow only | Distill from Fast (replay) |
| REM | 200 | Slow only | Multi-step rollout (replay) |
| Tether | 10 | Fast only | Distill from Slow (replay) |

## Biological Analogy

```
┌──────────────────────────────────────────────────────────┐
│                    BIOLOGICAL BRAIN                       │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Visual Cortex (V1-V4)        Hippocampus               │
│  ├─ Feature extraction         ├─ Fast plasticity        │
│  ├─ Pretrained (development)   ├─ Episodic memory        │
│  └─ Stable representations     └─ Rapid encoding         │
│                                                          │
│  Neocortex                     Curiosity / Dopamine      │
│  ├─ Slow plasticity            ├─ Novelty detection      │
│  ├─ Semantic memory            ├─ Exploration drive      │
│  └─ Gradual integration        └─ Reward prediction error│
│                                                          │
│  During Sleep:                                           │
│  ├─ SWS: Sharp-wave ripples (hippocampal replay)        │
│  ├─ REM: Dream sequences (trajectory simulation)        │
│  └─ Both: Systems consolidation (hippo → cortex)        │
│                                                          │
└──────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│                 VISUAL-MOTOR MODEL                        │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Visual Encoder                Fast Learner              │
│  ├─ ResNet50 (frozen)          ├─ High LR (1e-3)        │
│  ├─ Pretrained on ImageNet     ├─ Small network (256h)   │
│  └─ 512-d embeddings           └─ Quick adaptation       │
│                                                          │
│  Slow Learner                  Curiosity Policy          │
│  ├─ Low LR (1e-5 / 5e-4)      ├─ Prediction error       │
│  ├─ Large network (1024h)      ├─ Drives exploration     │
│  └─ Consolidated world model   └─ Seeks novel states     │
│                                                          │
│  During Sleep:                                           │
│  ├─ SWS: Distill Fast → Slow (replay buffer)           │
│  ├─ REM: Multi-step rollout consistency                 │
│  └─ Tether: Slow → Fast (prevent drift)                │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

## Data Flow Example

```
Step 1-499: Wake Phase
─────────────────────────

Camera captures scene → z_42 = Encoder(frame)
    │
    ▼
Curiosity Policy: π(z_42) → a = [+3.2°, -1.5°]
    │
    ▼
Robot moves servos: S1 += 3.2°, S2 -= 1.5°
    │
    ▼
Camera captures new scene → z_43 = Encoder(frame)
    │
    ▼
Fast Learner predicts: z_hat = F_H(z_42, [3.2, -1.5])
    │
    ▼
Prediction error: ||z_hat - z_43||² = 0.0847
    │
    ├─▶ Update Fast (LR=1e-3): reduce prediction error
    ├─▶ Update Slow (LR=1e-5): distill + raw
    ├─▶ Store {z_42, [3.2,-1.5], z_43} in replay buffer
    └─▶ Curiosity reward = 0.0847 → train policy to seek more


Step 500: Sleep Cycle
──────────────────────

SWS (200 steps):
    Sample 32 transitions from replay
    Slow learns to match Fast's predictions
    avg L_dyn: 0.0234 → 0.0089

REM (200 steps):
    Sample trajectory windows (4 consecutive steps)
    Slow predicts 3 steps ahead autoregressively
    avg L_ms: 0.1523 → 0.0612

Tether (10 steps):
    Fast adjusts toward Slow's outputs
    avg L_tether: 0.0156

Auto-save → models/my_experiment/checkpoint.pt
    │
    ▼
Resume Wake Phase (step 501)
```

## File Structure

```
project_curiosity/
├── dual_network_model.py              # Shared: ContinuousFast/SlowLearner
└── visual_memory/
    ├── config.py                      # All hyperparameters
    ├── encoder.py                     # VisualEncoder (frozen ResNet50 → 512-d)
    ├── curiosity_policy.py            # CuriosityPolicy (state → action)
    ├── trainer.py                     # VisualTrainer (orchestrates everything)
    ├── README.md                      # Detailed documentation
    └── robot/                         # Hardware interface
        ├── controller.py              # RobotInterface (serial + camera)
        ├── servo_control/             # Low-level servo communication
        └── control_stream.py          # Manual control & video stream

run_visual_learning.py                 # Main entry point

models/                                # Saved models (one folder per model)
└── <model_name>/
    ├── config.json                    # Frozen config snapshot at creation
    ├── checkpoint.pt                  # Weights + optimizer states + counters
    ├── replay.pt                      # Replay buffer (embeddings, no raw frames)
    └── training.log                   # Append-only CSV training log
```

## Model Persistence

```
models/
└── my_experiment/
    ├── config.json          Created once. Never overwritten.
    │   {                    Frozen snapshot of all hyperparameters.
    │     "fast_lr": 0.001,
    │     "slow_wake_lr": 1e-5,
    │     "wake_steps_per_cycle": 500,
    │     "created_at": "2026-02-14 09:36:37",
    │     ...
    │   }
    │
    ├── checkpoint.pt        Updated every save.
    │   - Fast Learner weights + optimizer
    │   - Slow Learner weights + optimizer (wake + sleep)
    │   - Policy weights + optimizer
    │   - step_count, cycle_count, wake_steps_in_cycle
    │
    ├── replay.pt            Updated every save.
    │   - List of {z_t, a_t, z_{t+1}, timestamp}
    │   - Embeddings only (no raw frames → small file)
    │
    └── training.log         Append-only CSV.
        timestamp, event, step, cycle, fast_loss, slow_distill,
        slow_raw, curiosity_reward, sws_L_dyn, rem_L_ms, ...

Resume: python run_visual_learning.py --model my_experiment
        (auto-detects checkpoint.pt and loads everything)
```

## Benefits Visualization

```
                    Catastrophic Forgetting
                            │
                            │  ┌──────────────────┐
Single Network:             │  │  Old environment │
                            │  │  forgotten!      │
                            ▼  └──────────────────┘
    ┌───────────────────────────────────────┐
    │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░ │
    └───────────────────────────────────────┘
      New scene only        Old scenes lost


Dual Network + Sleep:    Continual Learning
                                │
    Fast:  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  (adapts to new)
    Slow:  ░░░░░░░░░░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  (retains all)
                      │
                      ▼
           All environments preserved!


    Why it works:
    ┌────────────────────────────────────────────────────┐
    │  Wake:   Fast learns new → Slow absorbs slowly    │
    │  SWS:    Fast's knowledge → distilled into Slow   │
    │  REM:    Slow's predictions → stabilized over time│
    │  Tether: Slow's stability → anchors Fast          │
    └────────────────────────────────────────────────────┘
```
