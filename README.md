# Project Curiosity

An exploration of **curiosity-driven continual learning**, inspired by the brain's Complementary Learning Systems (CLS) theory. The project implements two parallel learning systems — one for language concepts and one for visual-motor control — both built on a shared **dual-network architecture** that mirrors hippocampal-cortical memory consolidation.

## Core Idea

The brain uses two complementary systems to learn without forgetting:
- The **hippocampus** rapidly encodes new experiences (Fast Learner)
- The **neocortex** slowly consolidates them into stable knowledge (Slow Learner)
- **Sleep** orchestrates the transfer between the two

This project applies that principle to two domains:

| | Language | Visual-Motor |
|---|---|---|
| **Domain** | Concept relationships (discrete) | Visual scene prediction (continuous) |
| **Input** | Word embeddings | Camera frames (ResNet50 → 512-d) |
| **Task** | Predict concept operations & relations | Predict next visual state from action |
| **Feedback** | LLM validates predictions | Prediction error (self-supervised) |
| **Exploration** | Random concept sampling | Curiosity-driven servo movement |
| **Sleep** | Replay + distillation | SWS → REM → Tether cycle |

## Repository Structure

```
project_curiosity/
├── __init__.py
├── dual_network_model.py              # Shared: ContinuousFast/SlowLearner
│
├── language/                           # Language concept learning
│   ├── config.py                       # Language hyperparameters
│   ├── model.py                        # ConceptActionModel (single network)
│   ├── dual_network_model_language.py  # Language-specific Fast/Slow learners
│   ├── trainer.py                      # Single-network trainer
│   ├── dual_trainer.py                 # Dual-network trainer with sleep
│   ├── vocabulary.py                   # Dynamic vocabulary management
│   ├── embeddings.py                   # GloVe embedding loader
│   ├── actions.py                      # Relation & operation action types
│   ├── questions.py                    # LLM prompt generation
│   ├── llm.py                          # LLM API interface
│   ├── training_logger.py              # Training log management
│   ├── cooccurrence_stats.py           # Co-occurrence analysis
│   └── mock_test.py                    # Offline testing with mock LLM
│
├── visual_memory/                      # Visual-motor continuous learning
│   ├── config.py                       # Visual system hyperparameters
│   ├── encoder.py                      # VisualEncoder (frozen ResNet50)
│   ├── curiosity_policy.py             # Curiosity-driven exploration policy
│   ├── trainer.py                      # VisualTrainer (wake/sleep cycle)
│   ├── README.md                       # Detailed visual system documentation
│   └── robot/                          # Hardware interface
│       ├── controller.py               # RobotInterface (camera + servos)
│       ├── servo_control/              # Low-level servo communication
│       └── control_stream.py           # Manual control & video stream
│
├── models/                             # Saved visual models (per-model folders)

# Root-level scripts
├── run_visual_learning.py              # Visual-motor training (main entry)
├── train.py                            # Language training (automated, LLM)
├── train_dual.py                       # Language dual-network training
├── interactive_train.py                # Language training (human-in-the-loop)
├── init_model.py                       # Initialize a new language model
├── demo_dreaming.py                    # Demo: language dreaming via inverse pass
└── demo_dual_network.py                # Demo: dual-network language learning
```

## System 1: Language Concept Learning

Teaches a model conceptual relationships and operations through interactive training with LLM feedback.

### Action Types

- **Relations**: `oppose`, `similar`, `include` — verify relationships between concepts
- **Operations**: `combine`, `add`, `subtract`, `intersect` — generate new concepts

### Training Flow

1. Model proposes an action between two concepts
2. Model predicts the result
3. An LLM validates the prediction
4. If incorrect, the LLM provides a correction
5. Vocabulary dynamically expands with new concepts
6. During sleep, the Slow Learner consolidates via replay + distillation

### Quick Start

```bash
# Download GloVe embeddings (one-time)
python download_embeddings.py

# Initialize a model
python init_model.py --words words.txt --model-dir my_language_model --dual

# Train with LLM feedback
python train_dual.py --model-dir my_language_model

# Or train interactively
python interactive_train.py --model-dir my_language_model
```

## System 2: Visual-Motor Learning

A 2-servo robot learns a visual world model — predicting how the scene changes in response to motor actions — through a biologically-motivated wake/sleep cycle.

### Architecture

```
Camera → ResNet50 (frozen) → 512-d embedding → Fast/Slow Learners → predicted next state
                                                                   ↓
                                              Curiosity Policy ← prediction error
                                                    ↓
                                              servo actions → Robot
```

### Wake/Sleep Cycle

```
WAKE (500 steps)  →  SWS (200 steps)  →  REM (200 steps)  →  Tether (10 steps)  →  repeat
 Fast: learns          Fast: frozen        Fast: frozen        Fast: distill→Slow
 Slow: distill+raw     Slow: distill       Slow: rollout       Slow: frozen
```

### Quick Start

```bash
# Start a new model (creates models/my_robot/)
python run_visual_learning.py --model my_robot --port /dev/tty.usbmodem1234 --mode curiosity

# Resume training (auto-detects saved state)
python run_visual_learning.py --model my_robot --port /dev/tty.usbmodem1234

# List saved models
python run_visual_learning.py --list-models
```

Each model is self-contained in `models/<name>/` with weights, config snapshot, replay buffer, and training logs. See `project_curiosity/visual_memory/README.md` for detailed architecture documentation.

## Shared Foundation: Dual-Network Model

Both systems share the same core principle defined in `dual_network_model.py`:

| | Fast Learner (Hippocampus) | Slow Learner (Cortex) |
|---|---|---|
| **Size** | Small (256 hidden) | Large (1024 hidden) |
| **Learning Rate** | High (1e-3) | Low (1e-5 wake, 5e-4 sleep) |
| **Role** | Rapid adaptation | Stable consolidation |
| **When** | Always active during wake | Gradual wake + strong sleep |

## Prerequisites

- Python 3.10+
- `torch`, `torchvision`
- `opencv-python` (visual system)
- `gensim` (language system, for GloVe embeddings)
- `serial` (visual system, for servo control)

```bash
pip install -r requirements.txt
```