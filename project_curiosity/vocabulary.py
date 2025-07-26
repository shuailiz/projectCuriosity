"""Vocabulary handling for project_curiosity."""
from __future__ import annotations

import random
from typing import Dict, List, Optional

from . import config as C

class Vocabulary:
    def __init__(self, initial_tokens: Optional[List[str]] = None):
        self.tokens: List[str] = []
        self.token_to_id: Dict[str, int] = {}
        self._add_token(C.UNKNOWN_TOKEN)
        # reserve <PAD> with id 1 so it is never sampled by random_concept
        self._add_token(C.PAD_TOKEN)
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
        if len(self.tokens) >= C.VOCAB_SIZE:
            return 0
        return self._add_token(tok)

    def encode(self, tok: str) -> int:
        return self.token_to_id.get(tok, 0)

    def decode(self, idx: int) -> str:
        return self.tokens[idx] if 0 <= idx < len(self.tokens) else C.UNKNOWN_TOKEN

    def random_concept(self) -> int:
        valid = [i for i, t in enumerate(self.tokens) if not t.startswith("<EMPTY") and i > 1]
        return random.choice(valid) if valid else 0
