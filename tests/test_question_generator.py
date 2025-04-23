"""
Test file for the question generation components.
"""

import unittest
from unittest.mock import patch, MagicMock
import torch
import random
from src.question_generator.generator import QuestionGenerationEngine, RandomQuestionGenerator


class TestRandomQuestionGenerator(unittest.TestCase):
    """Test cases for the RandomQuestionGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = RandomQuestionGenerator()
        
        # Set random seed for reproducibility
        random.seed(42)
    
    def test_initialization(self):
        """Test that the generator initializes correctly."""
        self.assertIsInstance(self.generator, RandomQuestionGenerator)
        self.assertTrue(len(self.generator.question_templates) > 0)
    
    def test_reason_over_tokens(self):
        """Test the reason_over_tokens method."""
        # Test with a set of tokens
        tokens = ["artificial", "intelligence", "ethics", "regulation"]
        token_pairs = self.generator.reason_over_tokens(tokens)
        
        # Check that we get token pairs
        self.assertTrue(len(token_pairs) > 0)
        self.assertEqual(len(token_pairs[0]), 2)
        
        # Check that all pairs contain tokens from the original list
        for pair in token_pairs:
            self.assertTrue(pair[0] in tokens)
            self.assertTrue(pair[1] in tokens)
        
        # Test with too few tokens
        short_tokens = ["a", "b"]
        short_pairs = self.generator.reason_over_tokens(short_tokens)
        self.assertEqual(len(short_pairs), 0)
    
    def test_generate_questions(self):
        """Test the generate_questions method."""
        # Test with a set of tokens
        tokens = ["artificial", "intelligence", "ethics", "regulation"]
        questions = self.generator.generate_questions(tokens, num_questions=3)
        
        # Check that we get the expected number of questions
        self.assertEqual(len(questions), 3)
        
        # Check that all questions are strings
        for question in questions:
            self.assertIsInstance(question, str)
            self.assertTrue(len(question) > 0)
        
        # Test with too few tokens
        short_tokens = ["a", "b"]
        short_questions = self.generator.generate_questions(short_tokens, num_questions=2)
        self.assertEqual(len(short_questions), 1)  # Should return default question
        
        # Test with empty tokens
        empty_questions = self.generator.generate_questions([], num_questions=2)
        self.assertEqual(len(empty_questions), 1)  # Should return default question


class TestQuestionGenerationEngine(unittest.TestCase):
    """Test cases for the QuestionGenerationEngine class."""
    
    @patch('src.question_generator.generator.MLPQuestionGenerationSystem')
    @patch('src.question_generator.generator.TokenProcessor')
    def setUp(self, mock_token_processor, mock_mlp_system):
        """Set up test fixtures with mocked dependencies."""
        # Mock the token processor
        self.mock_token_processor_instance = MagicMock()
        mock_token_processor.return_value = self.mock_token_processor_instance
        
        # Mock the MLP system
        self.mock_mlp_system_instance = MagicMock()
        mock_mlp_system.return_value = self.mock_mlp_system_instance
        self.mock_mlp_system_instance.to.return_value = self.mock_mlp_system_instance
        
        # Model config
        self.model_config = {
            'token_dim': 768,
            'reasoner_hidden_dims': [512, 256],
            'reasoning_dim': 128,
            'generator_hidden_dims': [256, 512],
            'vocab_size': 30522,
            'max_question_length': 20,
            'num_heads': 4,
            'dropout_rate': 0.2
        }
        
        # Create a question generation engine with mocked dependencies
        self.engine = QuestionGenerationEngine(
            model_config=self.model_config,
            token_processor=self.mock_token_processor_instance,
            device='cpu'
        )
    
    def test_initialization(self):
        """Test that the engine initializes correctly."""
        self.assertIsInstance(self.engine, QuestionGenerationEngine)
    
    @patch('torch.load')
    def test_load_model(self, mock_torch_load):
        """Test the load_model method."""
        # Mock the torch.load function
        mock_state_dict = {'key': 'value'}
        mock_torch_load.return_value = mock_state_dict
        
        # Call the method
        self.engine.load_model('path/to/model.pt')
        
        # Check that torch.load was called with the correct path
        mock_torch_load.assert_called_once_with('path/to/model.pt', map_location='cpu')
        
        # Check that the model's load_state_dict method was called with the state dict
        self.mock_mlp_system_instance.load_state_dict.assert_called_once_with(mock_state_dict)
        
        # Check that the model was set to eval mode
        self.mock_mlp_system_instance.eval.assert_called_once()
    
    @patch('torch.save')
    def test_save_model(self, mock_torch_save):
        """Test the save_model method."""
        # Call the method
        self.engine.save_model('path/to/model.pt')
        
        # Check that torch.save was called with the model's state dict and the correct path
        mock_torch_save.assert_called_once()
        self.assertEqual(mock_torch_save.call_args[0][1], 'path/to/model.pt')
    
    def test_generate_questions(self):
        """Test the generate_questions method."""
        # Mock input tokens
        tokens = ["artificial", "intelligence", "ethics"]
        
        # Mock token embeddings and attention mask
        mock_embeddings = torch.randn(3, 4, 768)
        mock_attention_mask = torch.ones(3, 1, 1, 4)
        self.mock_token_processor_instance.process_token_set.return_value = (mock_embeddings, mock_attention_mask)
        
        # Mock generated tokens
        mock_generated_tokens = torch.tensor([[101, 2054, 2003, 102], [101, 2129, 2013, 102]])
        self.mock_mlp_system_instance.generate_questions.return_value = mock_generated_tokens
        
        # Mock decoded questions
        mock_decoded_questions = ["What is?", "How does?"]
        self.mock_token_processor_instance.decode_generated_questions.return_value = mock_decoded_questions
        
        # Call the method
        questions = self.engine.generate_questions(tokens, num_questions=1)
        
        # Check that the token processor's process_token_set method was called with the tokens
        self.mock_token_processor_instance.process_token_set.assert_called_once_with(tokens)
        
        # Check that the model's generate_questions method was called with the embeddings and mask
        self.mock_mlp_system_instance.generate_questions.assert_called_once()
        self.assertEqual(self.mock_mlp_system_instance.generate_questions.call_args[0][0].shape, mock_embeddings.shape)
        
        # Check that the token processor's decode_generated_questions method was called with the generated tokens
        self.mock_token_processor_instance.decode_generated_questions.assert_called_once_with(mock_generated_tokens)
        
        # Check that the result matches the expected decoded questions
        self.assertEqual(questions, mock_decoded_questions)


if __name__ == '__main__':
    unittest.main()
