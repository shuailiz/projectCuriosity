#!/usr/bin/env python3
"""
Interactive Human-in-the-Loop Training Script

This script provides an interface for training the model with human feedback:
1. Load model checkpoint and vocabulary
2. Interact with human feedback for validation and correction
3. Update/train the model
4. Save model checkpoint and vocabulary
"""

import os
import sys
import argparse
import torch
import pickle
from pathlib import Path
import json

from project_curiosity import config as C
from project_curiosity.vocabulary import Vocabulary
from project_curiosity.model import ConceptActionModel
from project_curiosity.trainer import Trainer
from project_curiosity.training_logger import TrainingLogger


def load_checkpoint(checkpoint_path: str, vocab_path: str):
    """Load model checkpoint, vocabulary, and config file.
    
    Args:
        checkpoint_path: Path to model checkpoint file
        vocab_path: Path to vocabulary pickle file
        
    Returns:
        Tuple of (model, vocabulary, optimizer_state, config)
    """
    # Load config file
    config_path = checkpoint_path.replace('.pt', '_config.json')
    config = None
    if os.path.exists(config_path):
        print(f"Loading config from {config_path}...")
        with open(config_path, 'r') as f:
            import json
            config = json.load(f)
        print(f"Config loaded:")
        print(f"  - Epoch: {config.get('epoch', 'unknown')}")
        print(f"  - Learning rate: {config.get('learning_rate', 'unknown')}")
        print(f"  - Embedding method: {config.get('embedding_method', 'unknown')}")
        print(f"  - Vocab size: {config.get('vocab_size', 'unknown')}")
        print(f"  - Max sequence length: {config.get('max_sequence_length', 'unknown')}")
        
        # Update global config with model-specific settings
        if 'max_sequence_length' in config:
            C.MAX_SEQUENCE_LENGTH = config['max_sequence_length']
    else:
        print(f"Warning: Config file not found: {config_path}")
        print("Loading checkpoint without config validation.")
    
    print(f"\nLoading vocabulary from {vocab_path}...")
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    print(f"Vocabulary loaded with {len(vocab.tokens)} tokens")
    
    # Validate vocab size matches config
    if config and config.get('vocab_size') != len(vocab.tokens):
        print(f"Warning: Config vocab size ({config.get('vocab_size')}) doesn't match "
              f"loaded vocabulary ({len(vocab.tokens)})")
    
    print(f"\nLoading model checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=C.DEVICE)
    
    # Load embeddings (use method from config if available)
    from project_curiosity.embeddings import load_pretrained, build_embedding_matrix
    embedding_method = config.get('embedding_method', 'glove-wiki-gigaword-100') if config else 'glove-wiki-gigaword-100'
    print(f"Loading embeddings: {embedding_method}")
    kv = load_pretrained(embedding_method)
    emb_matrix = build_embedding_matrix(vocab.tokens, kv).to(C.DEVICE)
    
    # Create model
    model = ConceptActionModel(C.VOCAB_SIZE, emb_matrix, freeze_embeddings=False).to(C.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if available
    optimizer_state = checkpoint.get('optimizer_state_dict', None)
    
    print(f"\nModel loaded successfully (epoch {checkpoint.get('epoch', 'unknown')})")
    
    return model, vocab, optimizer_state, config


def save_checkpoint(model, vocab, optimizer, epoch: int, total_steps: int, checkpoint_path: str, vocab_path: str):
    """Save model checkpoint, vocabulary, and config file.
    
    Args:
        model: The model to save
        vocab: The vocabulary to save
        optimizer: The optimizer to save
        epoch: Current epoch number
        total_steps: Total training steps completed
        checkpoint_path: Path to save model checkpoint
        vocab_path: Path to save vocabulary
    """
    print(f"\nSaving checkpoint to {checkpoint_path}...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    # Save model checkpoint
    torch.save({
        'epoch': epoch,
        'total_steps': total_steps,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    
    # Save vocabulary
    print(f"Saving vocabulary to {vocab_path}...")
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    
    # Save config file
    config_path = checkpoint_path.replace('.pt', '_config.json')
    print(f"Saving config to {config_path}...")
    
    # Get current learning rate from optimizer
    learning_rate = optimizer.param_groups[0]['lr']
    
    config = {
        'epoch': epoch,
        'total_steps': total_steps,
        'learning_rate': learning_rate,
        'embedding_method': 'glove-wiki-gigaword-100',  # From embeddings.py
        'vocab_size': len(vocab.tokens),
        'embed_dim': C.EMBED_DIM,
        'hidden_dim': C.HIDDEN_DIM,
        'max_sequence_length': C.MAX_SEQUENCE_LENGTH,
        'device': str(C.DEVICE),
        'checkpoint_path': checkpoint_path,
        'vocab_path': vocab_path,
    }
    
    with open(config_path, 'w') as f:
        import json
        json.dump(config, f, indent=2)
    
    print(f"Checkpoint saved successfully (epoch {epoch})")
    print(f"  - Model: {checkpoint_path}")
    print(f"  - Vocabulary: {vocab_path}")
    print(f"  - Config: {config_path}")


def interactive_training_loop(trainer: Trainer, num_steps: int, save_interval: int,
                              checkpoint_path: str, vocab_path: str, starting_total_steps: int = 0):
    """Run interactive training loop with human feedback.
    
    Args:
        trainer: The Trainer instance
        num_steps: Number of training steps to run
        save_interval: Save checkpoint every N steps
        checkpoint_path: Path to save checkpoints
        vocab_path: Path to save vocabulary
        starting_total_steps: Total steps completed before this session
    """
    print("\n" + "="*80)
    print("Starting Interactive Training with Human Feedback")
    print("="*80)
    print("\nYou will be asked to:")
    print("  1. Validate if questions are well-formed")
    print("  2. Confirm if the model's predictions are correct")
    print("  3. Provide corrections when predictions are wrong")
    print("\nPress Ctrl+C at any time to stop and save progress.")
    print("="*80 + "\n")
    
    epoch = 0
    total_steps = starting_total_steps
    
    try:
        for step in range(num_steps):
            total_steps += 1
            print(f"\n{'='*80}")
            print(f"Training Step {step + 1}/{num_steps} (Total: {total_steps})")
            print(f"{'='*80}")
            
            # Run one training step with human feedback
            result = trainer.train_step_with_human_feedback()
            
            # Display results
            print(f"\n--- Step Results ---")
            
            # Check if step was skipped (invalid question)
            if result.get('skip'):
                print(f"Status: SKIPPED (invalid question)")
                action_loss = result.get('action_loss', 'N/A')
                if isinstance(action_loss, (int, float)):
                    print(f"Action Loss: {action_loss:.4f}")
                else:
                    print(f"Action Loss: {action_loss}")
            else:
                # Normal step - show all results
                loss = result.get('loss', 'N/A')
                if isinstance(loss, (int, float)):
                    print(f"Loss: {loss:.4f}")
                else:
                    print(f"Loss: {loss}")
                print(f"Model Answer: {result.get('model_answer', 'N/A')}")
                print(f"Correct Answer: {result.get('correct_answer', 'N/A')}")
                print(f"Is Correct: {result.get('is_correct', 'N/A')}")
            
            # Save checkpoint at intervals
            if (step + 1) % save_interval == 0:
                epoch += 1
                save_checkpoint(
                    trainer.model,
                    trainer.vocab,
                    trainer.opt,
                    epoch,
                    total_steps,
                    checkpoint_path,
                    vocab_path
                )
                print(f"\nCheckpoint saved at step {step + 1} (total: {total_steps})")
            
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Saving final checkpoint...")
        save_checkpoint(
            trainer.model,
            trainer.vocab,
            trainer.opt,
            epoch,
            total_steps,
            checkpoint_path,
            vocab_path
        )
        print("Final checkpoint saved. Exiting.")
        sys.exit(0)
    
    # Save final checkpoint
    print("\n\nTraining completed!")
    save_checkpoint(
        trainer.model,
        trainer.vocab,
        trainer.opt,
        epoch + 1,
        total_steps,
        checkpoint_path,
        vocab_path
    )


def main():
    parser = argparse.ArgumentParser(
        description="Interactive training with human feedback"
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to model directory (contains vocab.pkl, config.json, and checkpoint.pt)'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=100,
        help='Number of training steps (default: 100)'
    )
    parser.add_argument(
        '--save-interval',
        type=int,
        default=None,
        help='Save checkpoint every N steps (overrides config value if specified)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=None,
        help='Learning rate for optimizer (overrides config value)'
    )
    
    args = parser.parse_args()
    
    # Define paths within model directory
    model_dir = args.model
    vocab_path = os.path.join(model_dir, 'vocab.pkl')
    config_path = os.path.join(model_dir, 'config.json')
    checkpoint_path = os.path.join(model_dir, 'checkpoint.pt')
    
    # Check if model directory exists and has required files
    if not os.path.exists(model_dir):
        print(f"Error: Model directory not found: {model_dir}")
        print("\nTo create a new model, use:")
        print(f"  python init_model.py {model_dir} --vocab-file words.txt")
        print("or")
        print(f"  python init_model.py {model_dir} --vocab-words apple banana orange")
        sys.exit(1)
    
    if not os.path.exists(vocab_path):
        print(f"Error: Vocabulary file not found: {vocab_path}")
        print(f"\nModel directory must contain vocab.pkl")
        print("Use init_model.py to create a new model with vocabulary.")
        sys.exit(1)
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        print(f"\nModel directory must contain config.json")
        print("Use init_model.py to create a new model with config.")
        sys.exit(1)
    
    # Check if this is a new model (no checkpoint yet) or resuming
    is_new_model = not os.path.exists(checkpoint_path)
    
    if is_new_model:
        print(f"Starting training for new model: {model_dir}")
        print("No checkpoint found - will create one after first save.")
        
        # Load vocabulary and config
        print(f"\nLoading vocabulary from {vocab_path}...")
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        print(f"Vocabulary loaded with {len(vocab.tokens)} tokens")
        
        print(f"\nLoading config from {config_path}...")
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Config loaded:")
        print(f"  - Learning rate: {config.get('learning_rate', 'unknown')}")
        print(f"  - Embedding method: {config.get('embedding_method', 'unknown')}")
        print(f"  - Max sequence length: {config.get('max_sequence_length', 'unknown')}")
        
        # Update global config with model-specific settings
        if 'max_sequence_length' in config:
            C.MAX_SEQUENCE_LENGTH = config['max_sequence_length']
        
        # Create training logger
        logger = TrainingLogger(model_dir)
        
        # Create new trainer
        trainer = Trainer(vocab, logger=logger)
        
        # Set learning rate from config
        if 'learning_rate' in config:
            for param_group in trainer.opt.param_groups:
                param_group['lr'] = config['learning_rate']
    else:
        print(f"Resuming training for model: {model_dir}")
        
        # Load checkpoint
        model, vocab, optimizer_state, config = load_checkpoint(checkpoint_path, vocab_path)
        
        # Create training logger
        logger = TrainingLogger(model_dir)
        
        # Create trainer with loaded model
        trainer = Trainer(vocab, logger=logger)
        trainer.model = model
        
        # Restore optimizer state
        if optimizer_state:
            trainer.opt.load_state_dict(optimizer_state)
            print("Optimizer state restored")
        
        # Use learning rate from config
        if config and 'learning_rate' in config:
            for param_group in trainer.opt.param_groups:
                param_group['lr'] = config['learning_rate']
    
    # Override learning rate if specified on command line
    if args.learning_rate is not None:
        print(f"\nOverriding learning rate to {args.learning_rate}")
        for param_group in trainer.opt.param_groups:
            param_group['lr'] = args.learning_rate
    
    # Get save_interval from config or command line
    save_interval = args.save_interval if args.save_interval is not None else config.get('save_interval', 10)
    print(f"Save interval: {save_interval} steps")
    
    # Get starting total steps from config
    starting_total_steps = config.get('total_steps', 0)
    print(f"Starting from total steps: {starting_total_steps}")
    
    # Run interactive training
    interactive_training_loop(
        trainer,
        args.steps,
        save_interval,
        checkpoint_path,
        vocab_path,
        starting_total_steps
    )


if __name__ == '__main__':
    main()
