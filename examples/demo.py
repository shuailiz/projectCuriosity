"""
Example script demonstrating the MLP Question Generator with ChatGPT verification.
"""

import os
import sys
import argparse
from dotenv import load_dotenv

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chatgpt_integration.verification_workflow import VerificationWorkflow


def main():
    """
    Main function to demonstrate the MLP Question Generator.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='MLP Question Generator Demo')
    parser.add_argument('--tokens', nargs='+', required=True, help='List of tokens to reason over')
    parser.add_argument('--num_questions', type=int, default=3, help='Number of questions to generate')
    parser.add_argument('--use_random', action='store_true', help='Use random generator instead of MLP model')
    parser.add_argument('--model_path', type=str, help='Path to pretrained model')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results')
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Initialize the verification workflow
    workflow = VerificationWorkflow(
        model_path=args.model_path,
        use_random_generator=args.use_random
    )
    
    print(f"Generating {args.num_questions} questions from tokens: {args.tokens}")
    
    # Generate and verify questions
    results = workflow.generate_and_verify(
        tokens=args.tokens,
        num_questions=args.num_questions,
        save_results=True,
        output_dir=args.output_dir
    )
    
    # Print generated questions
    print("\nGenerated Questions:")
    for i, question in enumerate(results["questions"]):
        print(f"{i+1}. {question}")
    
    # Print verification results if available
    if results.get("verifications"):
        print("\nVerification Results:")
        for i, verification in enumerate(results["verifications"]):
            print(f"Question {i+1} Score: {verification['score']}/10")
    
    # Print question-answer pairs if available
    if results.get("question_answer_pairs"):
        print("\nQuestion-Answer Pairs:")
        for i, pair in enumerate(results["question_answer_pairs"]):
            print(f"Q{i+1}: {pair['question']}")
            print(f"A{i+1}: {pair['answer'][:100]}...")  # Print first 100 chars of answer
            print(f"Score: {pair['score']}/10")
            print()
    
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
