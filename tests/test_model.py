"""
Test file for the MLP model components.
"""

import unittest
import torch
from src.model.mlp_model import MLPModel, TokenReasoner, QuestionGenerator, MLPQuestionGenerationSystem


class TestMLPModel(unittest.TestCase):
    """Test cases for the MLPModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.input_dim = 768
        self.hidden_dims = [512, 256]
        self.output_dim = 128
        self.batch_size = 4
        self.model = MLPModel(self.input_dim, self.hidden_dims, self.output_dim)
    
    def test_model_initialization(self):
        """Test that the model initializes correctly."""
        self.assertIsInstance(self.model, MLPModel)
    
    def test_forward_pass(self):
        """Test the forward pass of the model."""
        # Create random input tensor
        x = torch.randn(self.batch_size, self.input_dim)
        
        # Forward pass
        output = self.model(x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))


class TestTokenReasoner(unittest.TestCase):
    """Test cases for the TokenReasoner class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.token_dim = 768
        self.hidden_dims = [512, 256]
        self.reasoning_dim = 128
        self.num_heads = 4
        self.seq_len = 10
        self.batch_size = 4
        self.reasoner = TokenReasoner(
            self.token_dim, 
            self.hidden_dims, 
            self.reasoning_dim, 
            self.num_heads
        )
    
    def test_reasoner_initialization(self):
        """Test that the reasoner initializes correctly."""
        self.assertIsInstance(self.reasoner, TokenReasoner)
    
    def test_forward_pass(self):
        """Test the forward pass of the reasoner."""
        # Create random input tensor
        tokens = torch.randn(self.seq_len, self.batch_size, self.token_dim)
        
        # Forward pass
        reasoning = self.reasoner(tokens)
        
        # Check output shape
        self.assertEqual(reasoning.shape, (self.batch_size, self.reasoning_dim))


class TestQuestionGenerator(unittest.TestCase):
    """Test cases for the QuestionGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.reasoning_dim = 128
        self.hidden_dims = [256, 512]
        self.vocab_size = 30522
        self.max_question_length = 20
        self.batch_size = 4
        self.generator = QuestionGenerator(
            self.reasoning_dim, 
            self.hidden_dims, 
            self.vocab_size, 
            self.max_question_length
        )
    
    def test_generator_initialization(self):
        """Test that the generator initializes correctly."""
        self.assertIsInstance(self.generator, QuestionGenerator)
    
    def test_forward_pass(self):
        """Test the forward pass of the generator."""
        # Create random input tensor
        reasoning = torch.randn(self.batch_size, self.reasoning_dim)
        
        # Forward pass
        outputs = self.generator(reasoning)
        
        # Check output shape
        self.assertEqual(outputs.shape, (self.batch_size, self.max_question_length, self.vocab_size))
    
    def test_generate_method(self):
        """Test the generate method of the generator."""
        # Create random input tensor
        reasoning = torch.randn(self.batch_size, self.reasoning_dim)
        
        # Generate questions
        generated_tokens = self.generator.generate(reasoning)
        
        # Check output shape
        self.assertEqual(generated_tokens.shape, (self.batch_size, self.max_question_length))
        
        # Check that output contains integers (token indices)
        self.assertTrue(torch.is_tensor(generated_tokens))
        self.assertEqual(generated_tokens.dtype, torch.long)


class TestMLPQuestionGenerationSystem(unittest.TestCase):
    """Test cases for the MLPQuestionGenerationSystem class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.token_dim = 768
        self.reasoner_hidden_dims = [512, 256]
        self.reasoning_dim = 128
        self.generator_hidden_dims = [256, 512]
        self.vocab_size = 30522
        self.max_question_length = 20
        self.num_heads = 4
        self.seq_len = 10
        self.batch_size = 4
        
        self.system = MLPQuestionGenerationSystem(
            self.token_dim,
            self.reasoner_hidden_dims,
            self.reasoning_dim,
            self.generator_hidden_dims,
            self.vocab_size,
            self.max_question_length,
            self.num_heads
        )
    
    def test_system_initialization(self):
        """Test that the system initializes correctly."""
        self.assertIsInstance(self.system, MLPQuestionGenerationSystem)
        self.assertIsInstance(self.system.token_reasoner, TokenReasoner)
        self.assertIsInstance(self.system.question_generator, QuestionGenerator)
    
    def test_forward_pass(self):
        """Test the forward pass of the system."""
        # Create random input tensor
        tokens = torch.randn(self.seq_len, self.batch_size, self.token_dim)
        
        # Forward pass
        outputs = self.system(tokens)
        
        # Check output shape
        self.assertEqual(outputs.shape, (self.batch_size, self.max_question_length, self.vocab_size))
    
    def test_generate_questions(self):
        """Test the generate_questions method of the system."""
        # Create random input tensor
        tokens = torch.randn(self.seq_len, self.batch_size, self.token_dim)
        
        # Generate questions
        generated_tokens = self.system.generate_questions(tokens)
        
        # Check output shape
        self.assertEqual(generated_tokens.shape, (self.batch_size, self.max_question_length))
        
        # Check that output contains integers (token indices)
        self.assertTrue(torch.is_tensor(generated_tokens))
        self.assertEqual(generated_tokens.dtype, torch.long)


if __name__ == '__main__':
    unittest.main()
