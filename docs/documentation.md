# MLP Question Generator with ChatGPT Verification

This document provides detailed documentation for the MLP Question Generator project.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Components](#components)
   - [MLP Model](#mlp-model)
   - [Token Processing](#token-processing)
   - [Question Generation](#question-generation)
   - [ChatGPT Integration](#chatgpt-integration)
   - [Verification Workflow](#verification-workflow)
5. [Usage Examples](#usage-examples)
6. [API Reference](#api-reference)
7. [Configuration](#configuration)
8. [Development Guide](#development-guide)
9. [Testing](#testing)
10. [Troubleshooting](#troubleshooting)

## Project Overview

The MLP Question Generator is a machine learning system that:

1. Takes a set of tokens as input
2. Reasons over these tokens using a Multi-Layer Perceptron (MLP) architecture
3. Generates questions based on the reasoning
4. Sends these questions to ChatGPT for verification
5. Processes and evaluates the responses

This system can be used for various applications including:
- Educational content generation
- Interview question preparation
- Research question formulation
- Content ideation and brainstorming
- Testing and evaluation of language models

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.9.0 or higher
- OpenAI API key for ChatGPT integration

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mlp_question_generator.git
cd mlp_question_generator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the project root with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Project Structure

```
mlp_question_generator/
├── config/               # Configuration files
├── data/                 # Data directory
│   ├── raw/              # Raw input data
│   └── processed/        # Processed data ready for training
├── docs/                 # Documentation
├── notebooks/            # Jupyter notebooks for exploration and demonstration
├── src/                  # Source code
│   ├── model/            # MLP model implementation
│   ├── data/             # Data processing utilities
│   ├── utils/            # Utility functions
│   ├── question_generator/  # Question generation logic
│   └── chatgpt_integration/ # Integration with ChatGPT API
└── tests/                # Unit and integration tests
```

## Components

### MLP Model

The core of the system is a Multi-Layer Perceptron (MLP) model that processes token embeddings and generates representations for question generation. The model architecture consists of:

1. **MLPModel**: Base MLP architecture with configurable hidden layers
2. **TokenReasoner**: Processes token embeddings using self-attention and MLP layers
3. **QuestionGenerator**: Generates questions from reasoning representations
4. **MLPQuestionGenerationSystem**: Combines all components into a complete system

The model is implemented in `src/model/mlp_model.py`.

#### Key Features

- Configurable architecture with adjustable hidden dimensions
- Self-attention mechanism for capturing relationships between tokens
- LSTM-based sequence generation for questions
- Temperature-controlled sampling for diverse question generation

### Token Processing

Token processing is handled by the `TokenProcessor` class in `src/utils/token_processor.py`. This component:

1. Converts raw tokens to embeddings using pretrained models
2. Processes token sets for input to the MLP model
3. Decodes generated token indices back into natural language questions

#### Key Features

- Integration with Hugging Face's transformers library
- Support for various pretrained models (default: BERT)
- Attention mask generation for handling variable-length inputs
- Efficient batch processing of tokens

### Question Generation

Question generation is implemented in `src/question_generator/generator.py` with two approaches:

1. **QuestionGenerationEngine**: Uses the trained MLP model for generating questions
2. **RandomQuestionGenerator**: Rule-based approach for generating questions when a trained model is not available

#### Key Features

- Model-based generation with temperature control
- Rule-based generation with templates for different question types
- Token relationship reasoning for contextually relevant questions
- Support for generating multiple questions from a single token set

### ChatGPT Integration

The ChatGPT integration is implemented in `src/chatgpt_integration/chatgpt_client.py`. This component:

1. Connects to the OpenAI API
2. Verifies generated questions
3. Gets answers to questions
4. Evaluates question-answer pairs

#### Key Features

- Comprehensive question verification with quality scoring
- Detailed feedback on question quality and relevance
- Question-answer pair evaluation for coherence and quality
- Robust error handling for API interactions

### Verification Workflow

The verification workflow ties all components together and is implemented in `src/chatgpt_integration/verification_workflow.py`. It provides:

1. **VerificationWorkflow**: End-to-end workflow for generating and verifying questions
2. **BatchVerificationWorkflow**: Batch processing for multiple token sets

#### Key Features

- Seamless integration of all system components
- Support for both model-based and rule-based question generation
- Comprehensive result tracking and storage
- Batch processing capabilities for efficiency

## Usage Examples

### Basic Usage

```python
from src.chatgpt_integration.verification_workflow import VerificationWorkflow

# Initialize the workflow with random generator (no model required)
workflow = VerificationWorkflow(use_random_generator=True)

# Define a set of tokens
tokens = ["artificial intelligence", "ethics", "regulation", "innovation"]

# Generate and verify questions
results = workflow.generate_and_verify(tokens, num_questions=3)

# Print generated questions
for question in results["questions"]:
    print(question)
```

### Using a Trained Model

```python
from src.chatgpt_integration.verification_workflow import VerificationWorkflow

# Initialize the workflow with a trained model
workflow = VerificationWorkflow(model_path="path/to/trained_model.pt")

# Define a set of tokens
tokens = ["climate change", "renewable energy", "policy", "economics"]

# Generate and verify questions
results = workflow.generate_and_verify(tokens, num_questions=5)

# Print questions and their verification scores
for i, question in enumerate(results["questions"]):
    score = results["verifications"][i]["score"]
    print(f"Question: {question}")
    print(f"Score: {score}/10")
    print()
```

### Batch Processing

```python
from src.chatgpt_integration.verification_workflow import VerificationWorkflow, BatchVerificationWorkflow

# Initialize the verification workflow
verification_workflow = VerificationWorkflow(use_random_generator=True)

# Initialize the batch workflow
batch_workflow = BatchVerificationWorkflow(verification_workflow)

# Define multiple token sets
token_sets = [
    ["machine learning", "data", "algorithms", "bias"],
    ["quantum computing", "superposition", "entanglement", "qubits"],
    ["blockchain", "cryptocurrency", "decentralization", "smart contracts"]
]

# Process all token sets
all_results = batch_workflow.process_token_sets(token_sets, num_questions_per_set=2)

# Print results for each token set
for i, results in enumerate(all_results):
    print(f"Token Set {i+1}: {results['tokens']}")
    for question in results["questions"]:
        print(f"- {question}")
    print()
```

## API Reference

### MLPModel

```python
MLPModel(input_dim, hidden_dims, output_dim, dropout_rate=0.2)
```

**Parameters:**
- `input_dim` (int): Dimension of input token embeddings
- `hidden_dims` (list): List of hidden layer dimensions
- `output_dim` (int): Dimension of output representation
- `dropout_rate` (float): Dropout probability for regularization

### TokenReasoner

```python
TokenReasoner(token_dim, hidden_dims, reasoning_dim, num_heads=4, dropout_rate=0.2)
```

**Parameters:**
- `token_dim` (int): Dimension of token embeddings
- `hidden_dims` (list): List of hidden layer dimensions for the MLP
- `reasoning_dim` (int): Dimension of the reasoning output
- `num_heads` (int): Number of attention heads
- `dropout_rate` (float): Dropout probability for regularization

### QuestionGenerator

```python
QuestionGenerator(reasoning_dim, hidden_dims, vocab_size, max_question_length=20, dropout_rate=0.2)
```

**Parameters:**
- `reasoning_dim` (int): Dimension of the reasoning representation
- `hidden_dims` (list): List of hidden layer dimensions
- `vocab_size` (int): Size of the vocabulary for question generation
- `max_question_length` (int): Maximum length of generated questions
- `dropout_rate` (float): Dropout probability for regularization

### TokenProcessor

```python
TokenProcessor(model_name="bert-base-uncased", device=None)
```

**Parameters:**
- `model_name` (str): Name of the pretrained model to use for embeddings
- `device` (str): Device to use for processing ('cuda' or 'cpu')

### QuestionGenerationEngine

```python
QuestionGenerationEngine(model_config, token_processor=None, device=None)
```

**Parameters:**
- `model_config` (dict): Configuration for the MLP model
- `token_processor` (TokenProcessor, optional): Token processor instance
- `device` (str, optional): Device to use ('cuda' or 'cpu')

### RandomQuestionGenerator

```python
RandomQuestionGenerator()
```

### ChatGPTIntegration

```python
ChatGPTIntegration(api_key=None)
```

**Parameters:**
- `api_key` (str, optional): OpenAI API key. If not provided, will attempt to load from environment.

### VerificationWorkflow

```python
VerificationWorkflow(model_path=None, use_random_generator=False, chatgpt_api_key=None)
```

**Parameters:**
- `model_path` (str, optional): Path to a pretrained MLP model
- `use_random_generator` (bool): Whether to use the random generator instead of the MLP model
- `chatgpt_api_key` (str, optional): OpenAI API key for ChatGPT

## Configuration

The system can be configured through various parameters:

### Model Configuration

```python
model_config = {
    'token_dim': 768,  # Dimension of token embeddings
    'reasoner_hidden_dims': [512, 256],  # Hidden dimensions for token reasoner
    'reasoning_dim': 128,  # Dimension of reasoning representation
    'generator_hidden_dims': [256, 512],  # Hidden dimensions for question generator
    'vocab_size': 30522,  # Vocabulary size (default BERT vocab size)
    'max_question_length': 20,  # Maximum length of generated questions
    'num_heads': 4,  # Number of attention heads
    'dropout_rate': 0.2  # Dropout rate for regularization
}
```

### Environment Variables

- `OPENAI_API_KEY`: API key for OpenAI services

## Development Guide

### Adding New Question Templates

To add new question templates to the `RandomQuestionGenerator`:

1. Open `src/question_generator/generator.py`
2. Locate the `__init__` method of the `RandomQuestionGenerator` class
3. Add new templates to the `self.question_templates` list

### Training the MLP Model

The repository includes the model architecture, but training code is not included. To train the model:

1. Prepare a dataset of tokens and corresponding questions
2. Implement a training loop using PyTorch
3. Save the trained model using `torch.save(model.state_dict(), 'path/to/model.pt')`

### Extending the System

The modular design allows for easy extension:

- To support new token embedding models, extend the `TokenProcessor` class
- To implement new question generation strategies, create a new class similar to `RandomQuestionGenerator`
- To integrate with other verification services, create a new class similar to `ChatGPTIntegration`

## Testing

The project includes a test directory for unit and integration tests. To run tests:

```bash
pytest tests/
```

### Test Coverage

- Model tests: Verify the architecture and forward pass
- Token processing tests: Ensure correct embedding generation
- Question generation tests: Validate question quality and diversity
- Integration tests: Verify end-to-end workflow

## Troubleshooting

### Common Issues

1. **OpenAI API Key Not Found**
   - Ensure the API key is set in the `.env` file or provided directly to the `ChatGPTIntegration` class

2. **CUDA Out of Memory**
   - Reduce batch size or model dimensions
   - Use CPU instead of GPU for smaller workloads

3. **Slow Question Generation**
   - Use the `RandomQuestionGenerator` for faster generation
   - Reduce the number of attention heads or model dimensions

4. **Poor Question Quality**
   - Increase the temperature parameter for more diverse questions
   - Use a larger or more specialized pretrained model for token embeddings
   - Add more specific templates to the `RandomQuestionGenerator`
