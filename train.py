#!/usr/bin/env python3
"""
Automated training with LLM feedback (non-interactive).

This script trains the model using LLM feedback for validation and correction,
without requiring human interaction.
"""

import os
import sys
import argparse
import json
import pickle
import torch

from project_curiosity import config as C
from project_curiosity.vocabulary import Vocabulary
from project_curiosity.model import ConceptActionModel
from project_curiosity.trainer import Trainer
from project_curiosity.training_logger import TrainingLogger


def load_model(model_dir: str):
    """Load model from directory.
    
    Args:
        model_dir: Path to model directory
        
    Returns:
        Tuple of (trainer, config, checkpoint_path, vocab_path)
    """
    vocab_path = os.path.join(model_dir, 'vocab.pkl')
    config_path = os.path.join(model_dir, 'config.json')
    checkpoint_path = os.path.join(model_dir, 'checkpoint.pt')
    
    # Load vocabulary
    print(f"Loading vocabulary from {vocab_path}...")
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    print(f"Vocabulary loaded with {len(vocab.tokens)} tokens")
    
    # Load config
    print(f"Loading config from {config_path}...")
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(f"Config loaded:")
    print(f"  - Learning rate: {config.get('learning_rate')}")
    print(f"  - Embedding method: {config.get('embedding_method')}")
    print(f"  - Max sequence length: {config.get('max_sequence_length')}")
    
    # Update global config with model-specific settings
    if 'max_sequence_length' in config:
        C.MAX_SEQUENCE_LENGTH = config['max_sequence_length']
    
    # Create training logger
    logger = TrainingLogger(model_dir)
    
    # Create trainer
    trainer = Trainer(vocab, logger=logger)
    
    # Load checkpoint if it exists
    if os.path.exists(checkpoint_path):
        print(f"\nLoading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=C.DEVICE)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            trainer.opt.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Optimizer state restored")
        
        print(f"Checkpoint loaded (epoch {checkpoint.get('epoch', 0)})")
    else:
        print("\nNo checkpoint found - starting fresh")
    
    # Set learning rate from config
    if 'learning_rate' in config:
        for param_group in trainer.opt.param_groups:
            param_group['lr'] = config['learning_rate']
    
    return trainer, config, checkpoint_path, vocab_path


def save_model(trainer: Trainer, config: dict, epoch: int, total_steps: int, checkpoint_path: str, vocab_path: str):
    """Save model checkpoint, vocabulary, and config.
    
    Args:
        trainer: The trainer instance
        config: Config dictionary
        epoch: Current epoch number
        total_steps: Total training steps completed
        checkpoint_path: Path to save checkpoint
        vocab_path: Path to save vocabulary
    """
    print(f"\nSaving checkpoint...")
    
    # Update config
    config['epoch'] = epoch
    config['total_steps'] = total_steps
    config['vocab_size'] = len(trainer.vocab.tokens)
    
    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'total_steps': total_steps,
        'model_state_dict': trainer.model.state_dict(),
        'optimizer_state_dict': trainer.opt.state_dict(),
    }, checkpoint_path)
    
    # Save vocabulary
    with open(vocab_path, 'wb') as f:
        pickle.dump(trainer.vocab, f)
    
    # Save config
    config_path = checkpoint_path.replace('checkpoint.pt', 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Checkpoint saved (epoch {epoch})")


def train(model_dir: str, steps: int, save_interval: int, learning_rate: float = None):
    """Run automated training with LLM feedback.
    
    Args:
        model_dir: Path to model directory
        steps: Number of training steps
        save_interval: Save checkpoint every N steps
        learning_rate: Optional learning rate override
    """
    # Load model
    trainer, config, checkpoint_path, vocab_path = load_model(model_dir)
    
    # Override learning rate if specified
    if learning_rate is not None:
        print(f"\nOverriding learning rate to {learning_rate}")
        for param_group in trainer.opt.param_groups:
            param_group['lr'] = learning_rate
        config['learning_rate'] = learning_rate
    
    # Get starting epoch and total steps
    start_epoch = config.get('epoch', 0)
    starting_total_steps = config.get('total_steps', 0)
    
    print(f"\n{'='*80}")
    print(f"Starting Automated Training with LLM Feedback")
    print(f"{'='*80}")
    print(f"Model: {model_dir}")
    print(f"Steps: {steps}")
    print(f"Save interval: {save_interval}")
    print(f"Starting epoch: {start_epoch}")
    print(f"Starting total steps: {starting_total_steps}")
    print(f"{'='*80}\n")
    
    # Training loop
    total_steps = starting_total_steps
    try:
        for step in range(1, steps + 1):
            total_steps += 1
            print(f"\n--- Step {step}/{steps} (Total: {total_steps}) ---")
            
            # Run one training step with LLM feedback
            result = trainer.train_step()
            
            # Display results
            if result.get('skip'):
                print(f"Skipped (invalid question)")
                if 'action_loss' in result:
                    print(f"Action loss: {result['action_loss']:.4f}")
            else:
                print(f"Concept A: {result['concept_a']}")
                print(f"Concept B: {result['concept_b']}")
                print(f"Action: {result['action']}")
                print(f"Model Answer: {result['model_answer']}")
                print(f"Correct Answer: {result['correct_answer']}")
                print(f"Is Correct: {result['is_correct']}")
                print(f"Loss: {result['loss']:.4f}")
            
            # Save checkpoint at intervals
            if step % save_interval == 0:
                epoch = start_epoch + (step // save_interval)
                save_model(trainer, config, epoch, total_steps, checkpoint_path, vocab_path)
                print(f"Progress: {step}/{steps} steps completed (total: {total_steps})")
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Saving checkpoint...")
        epoch = start_epoch + (step // save_interval)
        save_model(trainer, config, epoch, total_steps, checkpoint_path, vocab_path)
        print("Checkpoint saved. Exiting.")
        sys.exit(0)
    
    # Save final checkpoint
    print("\n\nTraining completed!")
    final_epoch = start_epoch + (steps // save_interval)
    save_model(trainer, config, final_epoch, total_steps, checkpoint_path, vocab_path)


def main():
    parser = argparse.ArgumentParser(
        description="Automated training with LLM feedback"
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to model directory'
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
    
    # Validate model directory
    if not os.path.exists(args.model):
        print(f"Error: Model directory not found: {args.model}")
        print("\nTo create a new model, use:")
        print(f"  python init_model.py {args.model} --vocab-file words.txt")
        sys.exit(1)
    
    # Load config to get save_interval if not specified
    config_path = os.path.join(args.model, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Use command-line save_interval if specified, otherwise use config value
    save_interval = args.save_interval if args.save_interval is not None else config.get('save_interval', 10)
    
    # Run training
    train(args.model, args.steps, save_interval, args.learning_rate)


if __name__ == '__main__':
    main()
