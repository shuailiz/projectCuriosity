#!/bin/bash

# Run all tests
python -m unittest discover -s tests

# Example usage of the MLP Question Generator
echo "Running example with random question generator..."
python examples/demo.py --tokens artificial intelligence ethics regulation --num_questions 2 --use_random

echo "Done!"
