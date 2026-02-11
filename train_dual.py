#!/usr/bin/env python3
"""
Dual-Network Training with Continual Learning (Hippocampus-Cortex System).

This script trains a dual-network model that never freezes weights:
- Fast Learner (Hippocampus): Small network, high learning rate, rapid adaptation
- Slow Learner (Cortex): Large network, low learning rate, stable knowledge
- Consolidation: Periodic replay and knowledge transfer during "sleep"
"""

import os
import sys
import argparse
import json
import pickle
import torch

from project_curiosity import config as C
from project_curiosity.vocabulary import Vocabulary
from project_curiosity.dual_trainer import DualNetworkTrainer
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
    checkpoint_path = os.path.join(model_dir, 'checkpoint_dual.pt')
    
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
    
    # Create dual-network trainer
    trainer = DualNetworkTrainer(vocab, logger=logger)
    
    # Load checkpoint if it exists
    if os.path.exists(checkpoint_path):
        print(f"\nLoading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=C.DEVICE)
        
        # Load dual-network state
        if 'fast_learner_state_dict' in checkpoint:
            trainer.model.fast_learner.load_state_dict(checkpoint['fast_learner_state_dict'])
            print("Fast learner state restored")
        if 'slow_learner_state_dict' in checkpoint:
            trainer.model.slow_learner.load_state_dict(checkpoint['slow_learner_state_dict'])
            print("Slow learner state restored")
        if 'fast_optimizer_state_dict' in checkpoint:
            trainer.fast_opt.load_state_dict(checkpoint['fast_optimizer_state_dict'])
            print("Fast optimizer state restored")
        if 'slow_optimizer_state_dict' in checkpoint:
            trainer.slow_opt.load_state_dict(checkpoint['slow_optimizer_state_dict'])
            print("Slow optimizer state restored")
        if 'consolidation_count' in checkpoint:
            trainer.model.consolidation_count = checkpoint['consolidation_count']
        if 'interaction_steps' in checkpoint:
            trainer.model.interaction_steps = checkpoint['interaction_steps']
        
        print(f"Checkpoint loaded (epoch {checkpoint.get('epoch', 0)})")
    else:
        print("\nNo checkpoint found - starting fresh")
    
    return trainer, config, checkpoint_path, vocab_path


def save_model(trainer: DualNetworkTrainer, config: dict, epoch: int, total_steps: int, 
               checkpoint_path: str, vocab_path: str):
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
    
    # Prepare checkpoint data
    checkpoint_data = {
        'epoch': epoch,
        'total_steps': total_steps,
    }
    
    # Save dual-network state
    checkpoint_data.update({
        'fast_learner_state_dict': trainer.model.fast_learner.state_dict(),
        'slow_learner_state_dict': trainer.model.slow_learner.state_dict(),
        'fast_optimizer_state_dict': trainer.fast_opt.state_dict(),
        'slow_optimizer_state_dict': trainer.slow_opt.state_dict(),
        'consolidation_count': trainer.model.consolidation_count,
        'interaction_steps': trainer.model.interaction_steps,
    })
    
    # Save network stats to config
    stats = trainer.model.get_network_stats()
    config['dual_network_stats'] = stats
    
    # Save checkpoint
    torch.save(checkpoint_data, checkpoint_path)
    
    # Save vocabulary
    with open(vocab_path, 'wb') as f:
        pickle.dump(trainer.vocab, f)
    
    # Save config
    config_path = checkpoint_path.replace('checkpoint_dual.pt', 'config.json').replace('checkpoint.pt', 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Checkpoint saved (epoch {epoch})")
    logger_stats = trainer.logger.get_stats() if trainer.logger else {}
    print(f"  - Logged experiences: {logger_stats.get('total_steps', 0)}")
    print(f"  - Consolidations: {trainer.model.consolidation_count}")


def train(model_dir: str, steps: int, save_interval: int, 
          fast_lr: float = None, slow_lr: float = None):
    """Run dual-network training with continual learning.
    
    Args:
        model_dir: Path to model directory
        steps: Number of training steps
        save_interval: Save checkpoint every N steps
        fast_lr: Optional fast learner learning rate override
        slow_lr: Optional slow learner learning rate override
    """
    # Load model
    trainer, config, checkpoint_path, vocab_path = load_model(model_dir)
    
    # Override learning rates if specified
    if fast_lr is not None:
        print(f"\nOverriding fast learner learning rate to {fast_lr}")
        for param_group in trainer.fast_opt.param_groups:
            param_group['lr'] = fast_lr
        config['fast_learning_rate'] = fast_lr
    
    if slow_lr is not None:
        print(f"Overriding slow learner learning rate to {slow_lr}")
        for param_group in trainer.slow_opt.param_groups:
            param_group['lr'] = slow_lr
        config['slow_learning_rate'] = slow_lr
    
    # Get starting epoch and total steps
    start_epoch = config.get('epoch', 0)
    starting_total_steps = config.get('total_steps', 0)
    
    print(f"\n{'='*80}")
    print(f"Starting Dual-Network Training")
    print(f"{'='*80}")
    print(f"Model: {model_dir}")
    print(f"Steps: {steps}")
    print(f"Save interval: {save_interval}")
    print(f"Starting epoch: {start_epoch}")
    print(f"Starting total steps: {starting_total_steps}")
    print(f"Consolidation interval: {C.CONSOLIDATION_INTERVAL}")
    print(f"Replay buffer size: {C.REPLAY_BUFFER_SIZE}")
    print(f"{'='*80}\n")
    
    # Training loop
    total_steps = starting_total_steps
    consolidation_count = 0
    
    try:
        for step in range(1, steps + 1):
            total_steps += 1
            print(f"\n--- Step {step}/{steps} (Total: {total_steps}) ---")
            
            # Run one training step
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
                
                # Display consolidation info if present
                if 'consolidation' in result:
                    consolidation_count += 1
                    cons = result['consolidation']
                    print(f"\n✨ Consolidation #{consolidation_count}:")
                    print(f"   Replays: {cons.get('replays', 0)}")
                    print(f"   Avg Loss: {cons.get('avg_loss', 0):.4f}")
                    print(f"   Buffer Size: {cons.get('buffer_size', 0)}")
            
            # Save checkpoint at intervals
            if step % save_interval == 0:
                epoch = start_epoch + (step // save_interval)
                save_model(trainer, config, epoch, total_steps, checkpoint_path, vocab_path)
                print(f"Progress: {step}/{steps} steps completed (total: {total_steps})")
                
                # Print network stats
                stats = trainer.get_stats()
                print(f"\nNetwork Stats:")
                print(f"  - Interaction steps: {stats.get('interaction_steps', 0)}")
                print(f"  - Consolidations: {stats.get('consolidation_count', 0)}")
                print(f"  - Logged experiences: {stats.get('total_logged_experiences', 0)}")
                print(f"  - Accuracy: {stats.get('accuracy', 0.0):.2%}")
    
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
    
    # Print final statistics
    print(f"\n{'='*80}")
    print("Final Statistics")
    print(f"{'='*80}")
    stats = trainer.get_stats()
    print(f"Fast Learner: {stats.get('fast_learner_params', 0):,} parameters")
    print(f"Slow Learner: {stats.get('slow_learner_params', 0):,} parameters")
    print(f"Interaction Steps: {stats.get('interaction_steps', 0)}")
    print(f"Consolidations: {stats.get('consolidation_count', 0)}")
    print(f"Logged Experiences: {stats.get('total_logged_experiences', 0)}")
    print(f"Accuracy: {stats.get('accuracy', 0.0):.2%}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Dual-Network Training with Continual Learning"
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
        default=None,
        help='Number of training steps (uses config value if not specified)'
    )
    parser.add_argument(
        '--save-interval',
        type=int,
        default=None,
        help='Save checkpoint every N steps (overrides config value if specified)'
    )
    parser.add_argument(
        '--fast-lr',
        type=float,
        default=None,
        help='Learning rate for fast learner (overrides config value)'
    )
    parser.add_argument(
        '--slow-lr',
        type=float,
        default=None,
        help='Learning rate for slow learner (overrides config value)'
    )
    
    args = parser.parse_args()
    
    # Validate model directory
    if not os.path.exists(args.model):
        print(f"Error: Model directory not found: {args.model}")
        print("\nTo create a new model, use:")
        print(f"  python init_model.py {args.model} --vocab-file words.txt")
        sys.exit(1)
    
    # Load config to use as defaults
    config_path = os.path.join(args.model, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Use command-line args if specified, otherwise use config values
    steps = args.steps if args.steps is not None else config.get('steps', 100)
    save_interval = args.save_interval if args.save_interval is not None else config.get('save_interval', 10)
    fast_lr = args.fast_lr if args.fast_lr is not None else config.get('fast_learning_rate')
    slow_lr = args.slow_lr if args.slow_lr is not None else config.get('slow_learning_rate')
    
    # Run training
    train(
        args.model, 
        steps, 
        save_interval, 
        fast_lr, 
        slow_lr
    )


if __name__ == '__main__':
    main()
