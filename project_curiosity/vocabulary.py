"""Vocabulary handling for project_curiosity."""
from __future__ import annotations

import random
from typing import Dict, List, Optional

from . import config as C

class Vocabulary:
    def __init__(self, initial_tokens: Optional[List[str]] = None):
        self.tokens: List[str] = []
        self.token_to_id: Dict[str, int] = {}
        self._add_token(C.UNKNOWN_TOKEN)  # ID 0
        self._add_token(C.PAD_TOKEN)      # ID 1
        self._add_token(C.FINISH_TOKEN)   # ID 2 - for sequence generation
        if initial_tokens:
            for t in initial_tokens:
                self.add(t)
        while len(self.tokens) < C.VOCAB_SIZE:
            self._add_token(f"<EMPTY_{len(self.tokens)}>")

    def _add_token(self, tok: str) -> int:
        idx = len(self.tokens)
        self.tokens.append(tok)
        self.token_to_id[tok] = idx
        return idx

    def add(self, tok: str) -> int:
        if tok in self.token_to_id:
            return self.token_to_id[tok]
        
        # Look for an <EMPTY_*> token to overwrite
        for i, existing_tok in enumerate(self.tokens):
            if existing_tok.startswith("<EMPTY_"):
                # Overwrite the empty token
                self.tokens[i] = tok
                # Update the token_to_id mapping
                del self.token_to_id[existing_tok]
                self.token_to_id[tok] = i
                return i
        
        # If no empty tokens found and vocabulary is full, return unknown token
        if len(self.tokens) >= C.VOCAB_SIZE:
            import warnings
            warnings.warn(
                f"Vocabulary is full ({C.VOCAB_SIZE} tokens). Cannot add '{tok}'. "
                f"Returning unknown token instead. Consider increasing VOCAB_SIZE "
                f"or implementing a more sophisticated token management strategy.",
                RuntimeWarning
            )
            return 0
            
        # Otherwise add as normal
        return self._add_token(tok)

    def encode(self, tok: str) -> int:
        return self.token_to_id.get(tok, 0)

    def decode(self, idx: int) -> str:
        return self.tokens[idx] if 0 <= idx < len(self.tokens) else C.UNKNOWN_TOKEN

    def random_concept(self) -> int:
        """Return a random valid concept ID, excluding special tokens.
        
        Excludes: <UNK> (0), <PAD> (1), <FINISH> (2), and <EMPTY_*> tokens.
        
        Returns:
            Random concept ID from valid tokens
            
        Raises:
            ValueError: If no valid tokens exist in vocabulary
        """
        valid = [i for i, t in enumerate(self.tokens) 
                 if i > 2 and not t.startswith("<EMPTY_") and not t.startswith("<")]
        
        if not valid:
            raise ValueError(
                "No valid concepts in vocabulary. Please add some real concepts "
                "before training. Current vocabulary only contains special tokens."
            )
        
        return random.choice(valid)
