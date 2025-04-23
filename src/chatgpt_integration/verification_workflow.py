"""
Verification workflow for the MLP question generation system.
"""

import os
import json
import torch
import random
from datetime import datetime
from ..question_generator.generator import QuestionGenerationEngine, RandomQuestionGenerator
from ..chatgpt_integration.chatgpt_client import ChatGPTIntegration
from ..utils.token_processor import TokenProcessor


class VerificationWorkflow:
    """
    Workflow for generating questions from tokens and verifying them with ChatGPT.
    """
    
    def __init__(self, model_path=None, use_random_generator=False, chatgpt_api_key=None):
        """
        Initialize the verification workflow.
        
        Args:
            model_path (str, optional): Path to a pretrained MLP model
            use_random_generator (bool): Whether to use the random generator instead of the MLP model
            chatgpt_api_key (str, optional): OpenAI API key for ChatGPT
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize token processor
        self.token_processor = TokenProcessor(device=self.device)
        
        # Initialize question generator
        if use_random_generator:
            self.question_generator = RandomQuestionGenerator()
        else:
            # Default model configuration
            model_config = {
                'token_dim': 768,
                'reasoner_hidden_dims': [512, 256],
                'reasoning_dim': 128,
                'generator_hidden_dims': [256, 512],
                'vocab_size': 30522,
                'max_question_length': 20,
                'num_heads': 4,
                'dropout_rate': 0.2
            }
            
            self.question_generator = QuestionGenerationEngine(
                model_config=model_config,
                token_processor=self.token_processor,
                device=self.device
            )
            
            # Load pretrained model if provided
            if model_path and os.path.exists(model_path):
                self.question_generator.load_model(model_path)
        
        # Initialize ChatGPT integration
        try:
            self.chatgpt_integration = ChatGPTIntegration(api_key=chatgpt_api_key)
        except ValueError as e:
            print(f"Warning: {str(e)}")
            print("ChatGPT verification will not be available.")
            self.chatgpt_integration = None
    
    def generate_and_verify(self, tokens, num_questions=3, save_results=True, output_dir=None):
        """
        Generate questions from tokens and verify them with ChatGPT.
        
        Args:
            tokens (list): List of token strings
            num_questions (int): Number of questions to generate
            save_results (bool): Whether to save results to a file
            output_dir (str, optional): Directory to save results
            
        Returns:
            dict: Results including generated questions, verification, and answers
        """
        # Generate questions
        if isinstance(self.question_generator, RandomQuestionGenerator):
            questions = self.question_generator.generate_questions(tokens, num_questions)
        else:
            questions = self.question_generator.generate_questions(tokens, num_questions)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "tokens": tokens,
            "questions": questions,
            "verifications": [],
            "question_answer_pairs": []
        }
        
        # Verify questions with ChatGPT if available
        if self.chatgpt_integration:
            # Verify questions
            verifications = self.chatgpt_integration.verify_questions(questions, tokens)
            results["verifications"] = verifications
            
            # Get answers and evaluate question-answer pairs
            question_answer_pairs = []
            for question in questions:
                # Get answer
                answer = self.chatgpt_integration.answer_question(question)
                
                # Evaluate pair
                evaluation = self.chatgpt_integration.evaluate_question_answer_pair(
                    question, answer, tokens
                )
                
                question_answer_pairs.append(evaluation)
            
            results["question_answer_pairs"] = question_answer_pairs
        
        # Save results if requested
        if save_results:
            self._save_results(results, output_dir)
        
        return results
    
    def _save_results(self, results, output_dir=None):
        """
        Save results to a JSON file.
        
        Args:
            results (dict): Results to save
            output_dir (str, optional): Directory to save results
        """
        # Create output directory if not provided
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), "results")
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename based on timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"verification_results_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Save results
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {filepath}")


class BatchVerificationWorkflow:
    """
    Workflow for batch processing multiple token sets.
    """
    
    def __init__(self, verification_workflow):
        """
        Initialize the batch verification workflow.
        
        Args:
            verification_workflow (VerificationWorkflow): Verification workflow instance
        """
        self.verification_workflow = verification_workflow
    
    def process_token_sets(self, token_sets, num_questions_per_set=3, save_results=True, output_dir=None):
        """
        Process multiple token sets.
        
        Args:
            token_sets (list): List of token sets, where each set is a list of token strings
            num_questions_per_set (int): Number of questions to generate per token set
            save_results (bool): Whether to save results to a file
            output_dir (str, optional): Directory to save results
            
        Returns:
            list: List of results for each token set
        """
        all_results = []
        
        for token_set in token_sets:
            # Generate and verify questions for this token set
            results = self.verification_workflow.generate_and_verify(
                token_set,
                num_questions=num_questions_per_set,
                save_results=False  # Don't save individual results
            )
            
            all_results.append(results)
        
        # Save combined results if requested
        if save_results:
            self._save_batch_results(all_results, output_dir)
        
        return all_results
    
    def _save_batch_results(self, all_results, output_dir=None):
        """
        Save batch results to a JSON file.
        
        Args:
            all_results (list): List of results for each token set
            output_dir (str, optional): Directory to save results
        """
        # Create output directory if not provided
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), "results")
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename based on timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"batch_verification_results_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Save results
        with open(filepath, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"Batch results saved to {filepath}")
