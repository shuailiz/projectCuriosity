"""
MLP model implementation for token reasoning and question generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPModel(nn.Module):
    """
    Multi-Layer Perceptron model for token reasoning and question generation.
    
    This model takes token embeddings as input, processes them through multiple
    fully connected layers, and outputs representations that can be used for
    question generation.
    """
    
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.2):
        """
        Initialize the MLP model.
        
        Args:
            input_dim (int): Dimension of input token embeddings
            hidden_dims (list): List of hidden layer dimensions
            output_dim (int): Dimension of output representation
            dropout_rate (float): Dropout probability for regularization
        """
        super(MLPModel, self).__init__()
        
        # Build layers dynamically based on hidden_dims
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # Combine all layers
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through the MLP model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        return self.model(x)


class TokenReasoner(nn.Module):
    """
    Model for reasoning over a set of tokens using an MLP architecture.
    
    This model processes a set of token embeddings, applies self-attention
    to capture relationships between tokens, and outputs a reasoning representation.
    """
    
    def __init__(self, token_dim, hidden_dims, reasoning_dim, num_heads=4, dropout_rate=0.2):
        """
        Initialize the token reasoner.
        
        Args:
            token_dim (int): Dimension of token embeddings
            hidden_dims (list): List of hidden layer dimensions for the MLP
            reasoning_dim (int): Dimension of the reasoning output
            num_heads (int): Number of attention heads
            dropout_rate (float): Dropout probability for regularization
        """
        super(TokenReasoner, self).__init__()
        
        # Self-attention layer to capture token relationships
        self.self_attention = nn.MultiheadAttention(token_dim, num_heads, dropout=dropout_rate)
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(token_dim)
        self.layer_norm2 = nn.LayerNorm(token_dim)
        
        # MLP for processing attended tokens
        self.mlp = MLPModel(token_dim, hidden_dims, token_dim, dropout_rate)
        
        # Final projection to reasoning dimension
        self.projection = nn.Linear(token_dim, reasoning_dim)
    
    def forward(self, tokens, mask=None):
        """
        Forward pass through the token reasoner.
        
        Args:
            tokens (torch.Tensor): Token embeddings of shape (seq_len, batch_size, token_dim)
            mask (torch.Tensor, optional): Attention mask
            
        Returns:
            torch.Tensor: Reasoning representation of shape (batch_size, reasoning_dim)
        """
        # Self-attention
        attended_tokens, _ = self.self_attention(tokens, tokens, tokens, attn_mask=mask)
        
        # Residual connection and layer normalization
        tokens = self.layer_norm1(tokens + attended_tokens)
        
        # MLP processing
        mlp_output = self.mlp(tokens)
        
        # Residual connection and layer normalization
        tokens = self.layer_norm2(tokens + mlp_output)
        
        # Average pooling over sequence dimension
        pooled = torch.mean(tokens, dim=0)
        
        # Project to reasoning dimension
        reasoning = self.projection(pooled)
        
        return reasoning


class QuestionGenerator(nn.Module):
    """
    Model for generating questions based on token reasoning.
    
    This model takes a reasoning representation and generates question embeddings
    that can be decoded into natural language questions.
    """
    
    def __init__(self, reasoning_dim, hidden_dims, vocab_size, max_question_length=20, dropout_rate=0.2):
        """
        Initialize the question generator.
        
        Args:
            reasoning_dim (int): Dimension of the reasoning representation
            hidden_dims (list): List of hidden layer dimensions
            vocab_size (int): Size of the vocabulary for question generation
            max_question_length (int): Maximum length of generated questions
            dropout_rate (float): Dropout probability for regularization
        """
        super(QuestionGenerator, self).__init__()
        
        self.max_question_length = max_question_length
        self.vocab_size = vocab_size
        
        # MLP for processing reasoning representation
        self.mlp = MLPModel(reasoning_dim, hidden_dims, hidden_dims[-1], dropout_rate)
        
        # LSTM for sequence generation
        self.lstm = nn.LSTM(
            input_size=hidden_dims[-1],
            hidden_size=hidden_dims[-1],
            num_layers=2,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Output projection to vocabulary
        self.output_projection = nn.Linear(hidden_dims[-1], vocab_size)
    
    def forward(self, reasoning, teacher_forcing_ratio=0.5):
        """
        Forward pass through the question generator.
        
        Args:
            reasoning (torch.Tensor): Reasoning representation of shape (batch_size, reasoning_dim)
            teacher_forcing_ratio (float): Probability of using teacher forcing during training
            
        Returns:
            torch.Tensor: Question token logits of shape (batch_size, max_question_length, vocab_size)
        """
        batch_size = reasoning.size(0)
        
        # Process reasoning through MLP
        processed_reasoning = self.mlp(reasoning)
        
        # Initialize hidden state with processed reasoning
        h0 = torch.stack([processed_reasoning, processed_reasoning], dim=0)  # 2 layers
        c0 = torch.zeros_like(h0)
        
        # Start with SOS token (assuming index 1)
        input_token = torch.ones(batch_size, 1, self.vocab_size).to(reasoning.device)
        input_token[:, :, 1] = 1.0  # One-hot encoding for SOS token
        
        # Initialize output tensor
        outputs = torch.zeros(batch_size, self.max_question_length, self.vocab_size).to(reasoning.device)
        
        # Generate sequence
        for t in range(self.max_question_length):
            # Process current token
            lstm_out, (h0, c0) = self.lstm(input_token, (h0, c0))
            
            # Project to vocabulary
            output = self.output_projection(lstm_out.squeeze(1))
            outputs[:, t, :] = output
            
            # Prepare next input (either ground truth or predicted)
            input_token = F.one_hot(output.argmax(dim=1), num_classes=self.vocab_size).unsqueeze(1).float()
        
        return outputs
    
    def generate(self, reasoning, temperature=1.0):
        """
        Generate questions from reasoning representation.
        
        Args:
            reasoning (torch.Tensor): Reasoning representation
            temperature (float): Sampling temperature for generation
            
        Returns:
            torch.Tensor: Generated question token indices
        """
        batch_size = reasoning.size(0)
        
        # Process reasoning through MLP
        processed_reasoning = self.mlp(reasoning)
        
        # Initialize hidden state with processed reasoning
        h0 = torch.stack([processed_reasoning, processed_reasoning], dim=0)  # 2 layers
        c0 = torch.zeros_like(h0)
        
        # Start with SOS token (assuming index 1)
        input_token = torch.ones(batch_size, 1, processed_reasoning.size(1)).to(reasoning.device)
        input_token[:, :, 1] = 1.0  # One-hot encoding for SOS token
        
        # Initialize output tensor
        generated_tokens = torch.zeros(batch_size, self.max_question_length, dtype=torch.long).to(reasoning.device)
        
        # Generate sequence
        for t in range(self.max_question_length):
            # Process current token
            lstm_out, (h0, c0) = self.lstm(input_token, (h0, c0))
            
            # Project to vocabulary
            logits = self.output_projection(lstm_out.squeeze(1))
            
            # Apply temperature
            scaled_logits = logits / temperature
            
            # Sample from distribution
            probs = F.softmax(scaled_logits, dim=1)
            next_token = torch.multinomial(probs, 1).squeeze(1)
            
            # Store generated token
            generated_tokens[:, t] = next_token
            
            # Prepare next input
            input_token = F.one_hot(next_token, num_classes=self.vocab_size).unsqueeze(1).float()
            
            # Stop if EOS token is generated (assuming index 2)
            if (next_token == 2).all():
                break
        
        return generated_tokens


class MLPQuestionGenerationSystem(nn.Module):
    """
    Complete system for token reasoning and question generation using MLP architecture.
    
    This system combines the token reasoner and question generator into a single model.
    """
    
    def __init__(self, token_dim, reasoner_hidden_dims, reasoning_dim, 
                 generator_hidden_dims, vocab_size, max_question_length=20,
                 num_heads=4, dropout_rate=0.2):
        """
        Initialize the complete system.
        
        Args:
            token_dim (int): Dimension of token embeddings
            reasoner_hidden_dims (list): Hidden dimensions for the token reasoner
            reasoning_dim (int): Dimension of the reasoning representation
            generator_hidden_dims (list): Hidden dimensions for the question generator
            vocab_size (int): Size of the vocabulary for question generation
            max_question_length (int): Maximum length of generated questions
            num_heads (int): Number of attention heads in the token reasoner
            dropout_rate (float): Dropout probability for regularization
        """
        super(MLPQuestionGenerationSystem, self).__init__()
        
        # Token reasoner component
        self.token_reasoner = TokenReasoner(
            token_dim=token_dim,
            hidden_dims=reasoner_hidden_dims,
            reasoning_dim=reasoning_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate
        )
        
        # Question generator component
        self.question_generator = QuestionGenerator(
            reasoning_dim=reasoning_dim,
            hidden_dims=generator_hidden_dims,
            vocab_size=vocab_size,
            max_question_length=max_question_length,
            dropout_rate=dropout_rate
        )
    
    def forward(self, tokens, mask=None):
        """
        Forward pass through the complete system.
        
        Args:
            tokens (torch.Tensor): Token embeddings
            mask (torch.Tensor, optional): Attention mask
            
        Returns:
            torch.Tensor: Question token logits
        """
        # Generate reasoning representation
        reasoning = self.token_reasoner(tokens, mask)
        
        # Generate questions from reasoning
        question_logits = self.question_generator(reasoning)
        
        return question_logits
    
    def generate_questions(self, tokens, mask=None, temperature=1.0):
        """
        Generate questions from input tokens.
        
        Args:
            tokens (torch.Tensor): Token embeddings
            mask (torch.Tensor, optional): Attention mask
            temperature (float): Sampling temperature for generation
            
        Returns:
            torch.Tensor: Generated question token indices
        """
        # Generate reasoning representation
        reasoning = self.token_reasoner(tokens, mask)
        
        # Generate questions from reasoning
        generated_tokens = self.question_generator.generate(reasoning, temperature)
        
        return generated_tokens
