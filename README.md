# projectCuriosity

A machine learning project that uses a Multi-Layer Perceptron (MLP) model to reason over tokens and generate questions, with ChatGPT integration for verification.

## Project Overview

This project implements a neural network-based system that:
1. Takes a set of tokens as input
2. Reasons over these tokens using an MLP architecture
3. Generates questions based on the reasoning
4. Sends these questions to ChatGPT for verification
5. Processes and evaluates the responses

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

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mlp_question_generator.git
cd mlp_question_generator

# Install dependencies
pip install -r requirements.txt
```

## Usage

[Usage instructions will be added here]

## License

[License information will be added here]

## Contributing

[Contribution guidelines will be added here]
