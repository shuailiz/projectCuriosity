# Project Curiosity

## Overview

Project Curiosity is an LLM Knowledge Trainer designed to teach language models conceptual relationships and operations. The system uses a novel approach to expand a model's understanding of concepts and their relationships through interactive training with feedback from a more capable LLM.

## Core Components

### Action Types

The system handles two primary categories of actions:

1. **Relation Actions**: Verify relationships between concepts
   - `oppose`: Checks if concepts are opposites
   - `similar`: Checks if concepts are similar
   - `include`: Checks if one concept includes another

2. **Operation Actions**: Generate new concepts from existing ones
   - `combine`: Creates a new concept by combining two concepts
   - `add`: Adds qualities of one concept to another
   - `subtract`: Removes qualities of one concept from another
   - `intersect`: Finds the common elements between two concepts

### Key Modules

- **Model**: Neural network architecture that learns to predict concept relationships and perform concept operations
- **Trainer**: Manages the training loop and optimization process
- **Actions**: Handles different action types and computes appropriate losses
- **Questions**: Generates validation questions and correction prompts for LLM feedback
- **LLM Integration**: Uses a more capable LLM to validate predictions and provide corrections

## Training Process

1. The model proposes an action between two concepts
2. For relation actions, it validates if the relation holds
3. For operation actions, it predicts a resulting concept
4. The system queries an external LLM to validate the prediction
5. If incorrect, the LLM provides a correction
6. The vocabulary is dynamically expanded to include new concepts
7. Loss is computed based on concept prediction and action prediction accuracy

## Features

- **Dynamic Vocabulary Expansion**: Automatically adds new concepts suggested by the LLM
- **Adaptive Loss Calculation**: Penalizes incorrect predictions with increased loss
- **Invalid Question Handling**: Uses KL divergence with uniform distribution over other actions to discourage invalid action proposals
- **Mock Testing**: Supports offline testing with mock LLM responses
- **Modular Design**: Clear separation between relation and operation actions

## Usage

### Training

```python
from project_curiosity.trainer import Trainer
from project_curiosity.model import ConceptModel

# Initialize model and trainer
model = ConceptModel(vocab_size=1000, embedding_dim=256)
trainer = Trainer(model)

# Start training
trainer.train(num_epochs=100)
```

### Mock Testing

```python
python -m project_curiosity.mock_test
```

## Development

The codebase is designed with modularity and testability in mind. Key design principles include:

- Dependency injection for mock functions to enable offline testing
- Clear separation of concerns between different action types
- Comprehensive docstrings and type hints
- Efficient loss computation and vocabulary management