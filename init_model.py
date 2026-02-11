#!/usr/bin/env python3
"""
Initialize a new model with vocabulary.

This script creates a new model directory with:
- Initial vocabulary from a text file or list of words
- Model configuration file
- Empty checkpoint (ready for training)
"""

import os
import sys
import argparse
import json
import pickle
from pathlib import Path

from project_curiosity import config as C
from project_curiosity.vocabulary import Vocabulary


def create_model_directory(model_dir: str, vocab_words: list, learning_rate: float = None,
                           embedding_method: str = 'glove-wiki-gigaword-100', 
                           dual: bool = False, fast_lr: float = None, slow_lr: float = None):
    """Create a new model directory with initial vocabulary and config.
    
    Args:
        model_dir: Path to model directory to create
        vocab_words: List of initial vocabulary words
        learning_rate: Learning rate for training (single network, default: from config)
        embedding_method: Embedding method to use
        dual: Whether to create a dual network model
        fast_lr: Fast learner learning rate (dual network only)
        slow_lr: Slow learner learning rate (dual network only)
    """
    # Create model directory
    os.makedirs(model_dir, exist_ok=True)
    print(f"Creating model directory: {model_dir}")
    
    # Create vocabulary
    print(f"Initializing vocabulary with {len(vocab_words)} words...")
    vocab = Vocabulary(initial_tokens=vocab_words)
    
    # Save vocabulary
    vocab_path = os.path.join(model_dir, 'vocab.pkl')
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"Vocabulary saved: {vocab_path}")
    
    # Create config
    config = {
        'epoch': 0,
        'total_steps': 0,  # Total training steps completed
        'embedding_method': embedding_method,
        'vocab_size': len(vocab.tokens),
        'embed_dim': C.EMBED_DIM,
        'hidden_dim': C.HIDDEN_DIM,
        'max_sequence_length': 3,
        'save_interval': 10,  # Save checkpoint every N steps
        'device': str(C.DEVICE),
        'model_dir': model_dir,
        'is_dual_network': dual,
    }
    
    # Add learning rates based on model type
    if dual:
        config['fast_learning_rate'] = fast_lr if fast_lr is not None else C.FAST_LEARNING_RATE
        config['slow_learning_rate'] = slow_lr if slow_lr is not None else C.SLOW_LEARNING_RATE
        config['fast_embed_dim'] = C.FAST_EMBED_DIM
        config['fast_hidden_dim'] = C.FAST_HIDDEN_DIM
        config['slow_embed_dim'] = C.SLOW_EMBED_DIM
        config['slow_hidden_dim'] = C.SLOW_HIDDEN_DIM
        config['consolidation_interval'] = C.CONSOLIDATION_INTERVAL
    else:
        config['learning_rate'] = learning_rate if learning_rate is not None else C.LEARNING_RATE
    
    config_path = os.path.join(model_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved: {config_path}")
    
    print(f"\nModel initialized successfully!")
    print(f"Model type: {'Dual Network' if dual else 'Single Network'}")
    print(f"Model directory: {model_dir}")
    print(f"  - Vocabulary: {len(vocab.tokens)} tokens")
    if dual:
        print(f"  - Fast learner LR: {config['fast_learning_rate']}")
        print(f"  - Slow learner LR: {config['slow_learning_rate']}")
    else:
        print(f"  - Learning rate: {config['learning_rate']}")
    print(f"  - Embedding method: {embedding_method}")
    print(f"\nTo start training:")
    if dual:
        print(f"  python interactive_train.py --model {model_dir} --dual")
    else:
        print(f"  python interactive_train.py --model {model_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Initialize a new model with vocabulary"
    )
    parser.add_argument(
        'model_dir',
        type=str,
        help='Path to model directory to create'
    )
    parser.add_argument(
        '--vocab-file',
        type=str,
        help='Path to text file with initial vocabulary (one word per line)'
    )
    parser.add_argument(
        '--vocab-words',
        type=str,
        nargs='+',
        help='List of initial vocabulary words'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=None,
        help='Learning rate for training (default: from config)'
    )
    parser.add_argument(
        '--embedding-method',
        type=str,
        default='glove-wiki-gigaword-100',
        help='Embedding method to use (default: glove-wiki-gigaword-100)'
    )
    parser.add_argument(
        '--dual',
        action='store_true',
        help='Create a dual network model (hippocampus-cortex system)'
    )
    parser.add_argument(
        '--fast-lr',
        type=float,
        default=None,
        help='Fast learner learning rate (dual network only, default: from config)'
    )
    parser.add_argument(
        '--slow-lr',
        type=float,
        default=None,
        help='Slow learner learning rate (dual network only, default: from config)'
    )
    
    args = parser.parse_args()
    
    # Get vocabulary words
    vocab_words = []
    
    if args.vocab_file:
        if not os.path.exists(args.vocab_file):
            print(f"Error: Vocabulary file not found: {args.vocab_file}")
            sys.exit(1)
        
        print(f"Loading vocabulary from {args.vocab_file}...")
        with open(args.vocab_file, 'r', encoding='utf-8') as f:
            vocab_words = [line.strip() for line in f if line.strip()]
    
    elif args.vocab_words:
        vocab_words = args.vocab_words
    
    else:
        print("Error: Must provide either --vocab-file or --vocab-words")
        print("\nExamples:")
        print("  python init_model.py my_model --vocab-file words.txt")
        print("  python init_model.py my_model --vocab-words apple banana orange")
        sys.exit(1)
    
    if not vocab_words:
        print("Error: No vocabulary words provided")
        sys.exit(1)
    
    # Create model directory
    create_model_directory(
        args.model_dir,
        vocab_words,
        args.learning_rate,
        args.embedding_method,
        dual=args.dual,
        fast_lr=args.fast_lr,
        slow_lr=args.slow_lr
    )


if __name__ == '__main__':
    main()
