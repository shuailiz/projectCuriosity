"""
Test file for the token processing utilities.
"""

import unittest
import torch
from unittest.mock import patch, MagicMock
from src.utils.token_processor import TokenProcessor


class TestTokenProcessor(unittest.TestCase):
    """Test cases for the TokenProcessor class."""
    
    @patch('src.utils.token_processor.AutoTokenizer')
    @patch('src.utils.token_processor.AutoModel')
    def setUp(self, mock_model, mock_tokenizer):
        """Set up test fixtures with mocked dependencies."""
        # Mock the tokenizer
        self.mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = self.mock_tokenizer_instance
        
        # Mock the model
        self.mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = self.mock_model_instance
        self.mock_model_instance.to.return_value = self.mock_model_instance
        
        # Create a token processor with mocked dependencies
        self.token_processor = TokenProcessor(device='cpu')
    
    def test_initialization(self):
        """Test that the token processor initializes correctly."""
        self.assertIsInstance(self.token_processor, TokenProcessor)
    
    def test_get_embeddings(self):
        """Test the get_embeddings method."""
        # Mock input tokens
        tokens = ["hello", "world"]
        
        # Mock encoded input
        mock_encoded_input = {"input_ids": torch.tensor([[101, 7592, 102], [101, 2088, 102]])}
        self.mock_tokenizer_instance.return_value = mock_encoded_input
        
        # Mock model output
        mock_last_hidden_state = torch.randn(2, 3, 768)
        mock_model_output = MagicMock()
        mock_model_output.last_hidden_state = mock_last_hidden_state
        self.mock_model_instance.return_value = mock_model_output
        
        # Call the method
        with patch.object(self.token_processor.tokenizer, '__call__', return_value=mock_encoded_input):
            with patch.object(self.token_processor.model, '__call__', return_value=mock_model_output):
                embeddings = self.token_processor.get_embeddings(tokens)
        
        # Check that the result is the mock last hidden state
        self.assertEqual(embeddings, mock_last_hidden_state)
    
    def test_process_token_set(self):
        """Test the process_token_set method."""
        # Mock input tokens
        token_set = ["artificial", "intelligence", "ethics"]
        
        # Mock encoded input with attention mask
        mock_encoded_input = {
            "input_ids": torch.tensor([[101, 7592, 102], [101, 2088, 102], [101, 3046, 102]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        }
        
        # Mock model output
        mock_last_hidden_state = torch.randn(3, 3, 768)
        mock_model_output = MagicMock()
        mock_model_output.last_hidden_state = mock_last_hidden_state
        
        # Call the method
        with patch.object(self.token_processor.tokenizer, '__call__', return_value=mock_encoded_input):
            with patch.object(self.token_processor.model, '__call__', return_value=mock_model_output):
                embeddings, mask = self.token_processor.process_token_set(token_set)
        
        # Check that the embeddings are the mock last hidden state
        self.assertEqual(embeddings, mock_last_hidden_state)
        
        # Check that the mask has the expected shape
        self.assertEqual(mask.shape[0], mock_encoded_input["attention_mask"].shape[0])
    
    def test_decode_generated_questions(self):
        """Test the decode_generated_questions method."""
        # Mock token indices
        token_indices = torch.tensor([
            [101, 2054, 2003, 1996, 2265, 1029, 102, 0, 0, 0],  # "What is the question?"
            [101, 2129, 2013, 2009, 2191, 1029, 102, 0, 0, 0]   # "How does it work?"
        ])
        
        # Mock decoded questions
        decoded_questions = ["What is the question?", "How does it work?"]
        
        # Set up the mock tokenizer to return the decoded questions
        self.mock_tokenizer_instance.decode.side_effect = decoded_questions
        
        # Call the method
        questions = self.token_processor.decode_generated_questions(token_indices)
        
        # Check that the result matches the expected decoded questions
        self.assertEqual(questions, decoded_questions)
        
        # Check that the tokenizer's decode method was called twice
        self.assertEqual(self.mock_tokenizer_instance.decode.call_count, 2)


if __name__ == '__main__':
    unittest.main()
