"""
Token processing and embedding generation utilities.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel


class TokenProcessor:
    """
    Utility class for processing tokens and generating embeddings.
    """
    
    def __init__(self, model_name="bert-base-uncased", device=None):
        """
        Initialize the token processor.
        
        Args:
            model_name (str): Name of the pretrained model to use for embeddings
            device (str): Device to use for processing ('cuda' or 'cpu')
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
    def get_embeddings(self, tokens):
        """
        Generate embeddings for a list of tokens.
        
        Args:
            tokens (list): List of token strings
            
        Returns:
            torch.Tensor: Embeddings for the tokens
        """
        # Tokenize the input tokens
        encoded_input = self.tokenizer(tokens, padding=True, truncation=True, return_tensors="pt").to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            
        # Use the last hidden state as embeddings
        embeddings = model_output.last_hidden_state
        
        return embeddings
    
    def process_token_set(self, token_set):
        """
        Process a set of tokens for input to the MLP model.
        
        Args:
            token_set (list): List of token strings
            
        Returns:
            torch.Tensor: Processed token embeddings
            torch.Tensor: Attention mask
        """
        # Tokenize the input tokens
        encoded_input = self.tokenizer(token_set, padding=True, truncation=True, return_tensors="pt").to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            
        # Use the last hidden state as embeddings
        embeddings = model_output.last_hidden_state
        
        # Get attention mask
        attention_mask = encoded_input['attention_mask']
        
        # Convert attention mask to format expected by model
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        return embeddings, extended_attention_mask
    
    def decode_generated_questions(self, token_indices):
        """
        Decode generated token indices into questions.
        
        Args:
            token_indices (torch.Tensor): Generated token indices
            
        Returns:
            list: List of decoded questions
        """
        # Convert token indices to list
        token_indices_list = token_indices.cpu().numpy().tolist()
        
        # Decode each sequence
        questions = []
        for indices in token_indices_list:
            # Find EOS token if present
            if 2 in indices:  # Assuming 2 is the EOS token
                indices = indices[:indices.index(2)]
            
            # Decode tokens
            question = self.tokenizer.decode(indices, skip_special_tokens=True)
            questions.append(question)
        
        return questions
