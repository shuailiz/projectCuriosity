"""
Question generation logic based on token reasoning.
"""

import torch
import random
import numpy as np
from ..model.mlp_model import MLPQuestionGenerationSystem
from ..utils.token_processor import TokenProcessor


class QuestionGenerationEngine:
    """
    Engine for generating questions based on token reasoning.
    """
    
    def __init__(self, model_config, token_processor=None, device=None):
        """
        Initialize the question generation engine.
        
        Args:
            model_config (dict): Configuration for the MLP model
            token_processor (TokenProcessor, optional): Token processor instance
            device (str, optional): Device to use ('cuda' or 'cpu')
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize token processor if not provided
        if token_processor is None:
            self.token_processor = TokenProcessor(device=self.device)
        else:
            self.token_processor = token_processor
        
        # Initialize model
        self.model = MLPQuestionGenerationSystem(
            token_dim=model_config.get('token_dim', 768),
            reasoner_hidden_dims=model_config.get('reasoner_hidden_dims', [512, 256]),
            reasoning_dim=model_config.get('reasoning_dim', 128),
            generator_hidden_dims=model_config.get('generator_hidden_dims', [256, 512]),
            vocab_size=model_config.get('vocab_size', 30522),  # Default BERT vocab size
            max_question_length=model_config.get('max_question_length', 20),
            num_heads=model_config.get('num_heads', 4),
            dropout_rate=model_config.get('dropout_rate', 0.2)
        ).to(self.device)
    
    def load_model(self, model_path):
        """
        Load a pretrained model.
        
        Args:
            model_path (str): Path to the model checkpoint
        """
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
    
    def save_model(self, model_path):
        """
        Save the current model.
        
        Args:
            model_path (str): Path to save the model checkpoint
        """
        torch.save(self.model.state_dict(), model_path)
    
    def generate_questions(self, tokens, num_questions=1, temperature=1.0):
        """
        Generate questions based on a set of tokens.
        
        Args:
            tokens (list): List of token strings
            num_questions (int): Number of questions to generate
            temperature (float): Sampling temperature for generation
            
        Returns:
            list: List of generated questions
        """
        # Process tokens
        token_embeddings, attention_mask = self.token_processor.process_token_set(tokens)
        
        # Generate questions
        all_questions = []
        for _ in range(num_questions):
            # Generate question tokens
            with torch.no_grad():
                generated_tokens = self.model.generate_questions(
                    token_embeddings, 
                    mask=attention_mask,
                    temperature=temperature
                )
            
            # Decode questions
            questions = self.token_processor.decode_generated_questions(generated_tokens)
            all_questions.extend(questions)
        
        return all_questions


class RandomQuestionGenerator:
    """
    Generator for creating random questions based on tokens.
    
    This class implements a rule-based approach for generating questions
    when a trained model is not available.
    """
    
    def __init__(self):
        """
        Initialize the random question generator.
        """
        # Templates for question generation
        self.question_templates = [
            "What is the relationship between {token1} and {token2}?",
            "How does {token1} affect {token2}?",
            "Why is {token1} important for {token2}?",
            "Can you explain how {token1} relates to {token2}?",
            "What are the key differences between {token1} and {token2}?",
            "In what context would {token1} be used with {token2}?",
            "What is the significance of {token1} in the context of {token2}?",
            "How would you describe {token1} to someone familiar with {token2}?",
            "What are the implications of {token1} for {token2}?",
            "Can {token1} be considered a subset of {token2}?",
            "What is {token1}?",
            "How would you define {token1}?",
            "What are the characteristics of {token1}?",
            "Why is {token1} significant?",
            "What examples illustrate {token1}?",
            "How has {token1} evolved over time?",
            "What are the applications of {token1}?",
            "What challenges are associated with {token1}?",
            "How does {token1} compare to similar concepts?",
            "What future developments might we see with {token1}?"
        ]
    
    def reason_over_tokens(self, tokens):
        """
        Simulate reasoning over tokens by identifying potential relationships.
        
        Args:
            tokens (list): List of token strings
            
        Returns:
            list: List of token pairs with potential relationships
        """
        # Filter out very short tokens (likely not meaningful)
        filtered_tokens = [token for token in tokens if len(token) > 2]
        
        # If not enough tokens, return empty list
        if len(filtered_tokens) < 2:
            return []
        
        # Generate all possible pairs
        token_pairs = []
        for i in range(len(filtered_tokens)):
            for j in range(i+1, len(filtered_tokens)):
                token_pairs.append((filtered_tokens[i], filtered_tokens[j]))
        
        # Randomly shuffle pairs
        random.shuffle(token_pairs)
        
        # Return all pairs or a subset if there are many
        max_pairs = min(len(token_pairs), 10)
        return token_pairs[:max_pairs]
    
    def generate_questions(self, tokens, num_questions=1):
        """
        Generate random questions based on tokens.
        
        Args:
            tokens (list): List of token strings
            num_questions (int): Number of questions to generate
            
        Returns:
            list: List of generated questions
        """
        # Reason over tokens to get potential relationships
        token_pairs = self.reason_over_tokens(tokens)
        
        # If no pairs, generate questions about individual tokens
        if not token_pairs:
            # Filter out very short tokens
            filtered_tokens = [token for token in tokens if len(token) > 2]
            
            # If still no tokens, return default question
            if not filtered_tokens:
                return ["Can you provide more information about these concepts?"]
            
            # Generate questions about individual tokens
            questions = []
            single_token_templates = self.question_templates[10:]  # Templates for single tokens
            
            for _ in range(min(num_questions, len(filtered_tokens))):
                token = random.choice(filtered_tokens)
                template = random.choice(single_token_templates)
                question = template.format(token1=token)
                questions.append(question)
            
            return questions
        
        # Generate questions based on token pairs
        questions = []
        pair_templates = self.question_templates[:10]  # Templates for token pairs
        
        # Ensure we don't try to generate more questions than we have pairs
        num_to_generate = min(num_questions, len(token_pairs))
        
        for i in range(num_to_generate):
            # Get token pair
            token1, token2 = token_pairs[i % len(token_pairs)]
            
            # Select random template
            template = random.choice(pair_templates)
            
            # Generate question
            question = template.format(token1=token1, token2=token2)
            questions.append(question)
        
        return questions
