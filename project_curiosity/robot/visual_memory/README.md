# Visual-Motor Continuous Learning

This component implements a dual-network continuous learning system for a 2-servo robot with visual input. It adapts the "Hippocampus-Cortex" memory consolidation theory to the visual-motor domain.

## Architecture

1.  **Visual Encoder**: ResNet18 (pretrained) extracts 512-dim embeddings from camera frames.
2.  **Fast Learner (Hippocampus)**: Shallow MLP that learns rapid associations between (State, Action) -> Next State. High learning rate.
3.  **Slow Learner (Cortex)**: Deep MLP that consolidates knowledge for long-term stability. Low learning rate.
4.  **Replay Buffer**: Stores `(State_Emb, Action, Next_State_Emb)` tuples.

## Usage

### Prerequisites
- A 2-servo robot connected via serial (USB).
- A webcam.
- `torch`, `torchvision`, `opencv-python`.

### Running the System
Run the training script from the project root:

```bash
python run_visual_learning.py --port /dev/tty.usbmodem1234 --camera 0
```

### Controls
- **W/S**: Move Servo 1 (+/-)
- **A/D**: Move Servo 2 (+/-)
- **Space**: No-Op (Capture & Train on stationarity)
- **A**: Toggle Auto-Exploration Mode
- **S**: Force Sleep (Consolidation) Phase
- **Q**: Quit

## Configuration
Hyperparameters are defined in `robot/visual_memory/config.py`:
- `FAST_LEARNING_RATE` / `SLOW_LEARNING_RATE`
- `CONSOLIDATION_INTERVAL`: How often (in steps) to sleep.
- `ACTION_SCALE`: Max degree change per step.

## Theory of Operation
1.  **Awake (Interaction)**:
    -   Robot moves (randomly or manually).
    -   Encodes current and next frames.
    -   Fast Learner updates *immediately* on the single transition.
    -   Transition stored in Replay Buffer.
2.  **Sleep (Consolidation)**:
    -   Triggered every N steps.
    -   Slow Learner trains on batches from Replay Buffer (Ground Truth).
    -   **Distillation**: Slow Learner also minimizes distance to Fast Learner's predictions, transferring the "recent" intuition to long-term memory.
