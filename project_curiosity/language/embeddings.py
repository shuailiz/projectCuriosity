"""Utility functions for creating pretrained embeddings for project_curiosity."""
from __future__ import annotations

import numpy as np
import torch
from gensim import downloader as api
from gensim.models import KeyedVectors

from . import config as C

_PRETRAINED_NAME = "glove-wiki-gigaword-100"  # 100-d vectors (~65 MB download)


def load_pretrained(name: str = _PRETRAINED_NAME) -> KeyedVectors:
    """Download (if needed) and return gensim KeyedVectors."""
    return api.load(name)


def build_embedding_matrix(vocab_tokens: list[str], kv: KeyedVectors) -> torch.Tensor:
    """Return a (|V|, C.EMBED_DIM) embedding matrix.

    Uses pre-trained embeddings for tokens that exist in the KeyedVectors.
    Any token absent from kv is initialized from N(0, 0.6).
    
    Note: With sequence generation, multi-word concepts are generated as 
    sequences of single-word tokens, so we don't need to compose embeddings.
    """
    emb = np.random.normal(scale=0.6, size=(len(vocab_tokens), C.EMBED_DIM)).astype(np.float32)
    
    for idx, tok in enumerate(vocab_tokens):
        if tok in kv:
            emb[idx] = kv[tok]
    
    return torch.tensor(emb)
