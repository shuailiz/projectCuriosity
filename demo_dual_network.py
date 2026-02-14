#!/usr/bin/env python3
"""
Demo script for dual-network continual learning system.

This script demonstrates the hippocampus-cortex analogy with:
- Fast learner adapting quickly during interaction
- Slow learner consolidating knowledge during sleep
- Replay buffer storing experiences
"""

import json
import time
from project_curiosity.language.vocabulary import Vocabulary
from project_curiosity.language.dual_trainer import DualNetworkTrainer


def print_header(title: str):
    """Print a formatted header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def demo_basic(steps: int = 20):
    """Basic demo of dual-network training."""
    print_header("Dual-Network Continual Learning Demo")
    
    # Initialize vocabulary
    vocab = Vocabulary([
        "apple", "banana", "orange", "fruit", "red", "yellow",
        "sweet", "sour", "tree", "plant", "food", "healthy"
    ])
    
    # Create dual-network trainer
    trainer = DualNetworkTrainer(vocab, use_dual_network=True)
    
    print(f"Running {steps} training steps...")
    print("Watch for consolidation phases (🌙) every 100 steps\n")
    
    consolidations = []
    
    for step in range(1, steps + 1):
        result = trainer.train_step()
        
        # Display step info
        if not result.get('skip'):
            print(f"\nStep {step}:")
            print(f"  Question: {result['concept_a']} {result['action']} {result['concept_b']}")
            print(f"  Model: {result['model_answer']}")
            print(f"  Correct: {result['correct_answer']}")
            print(f"  ✓" if result['is_correct'] else "  ✗")
            print(f"  Loss: {result['loss']:.4f}")
            
            # Check for consolidation
            if 'consolidation' in result:
                cons = result['consolidation']
                consolidations.append(cons)
                print(f"\n  🌙 CONSOLIDATION occurred!")
                print(f"     Replays: {cons.get('replays', 0)}")
                print(f"     Avg Loss: {cons.get('avg_loss', 0):.4f}")
        
        time.sleep(0.1)  # Slow down for readability
    
    # Final statistics
    print_header("Final Statistics")
    stats = trainer.get_stats()
    
    print(f"Network Architecture:")
    print(f"  Fast Learner: {stats['fast_learner_params']:,} parameters")
    print(f"  Slow Learner: {stats['slow_learner_params']:,} parameters")
    print(f"  Ratio: {stats['param_ratio']:.2f}x\n")
    
    print(f"Training Progress:")
    print(f"  Total Steps: {stats['step_count']}")
    print(f"  Interaction Steps: {stats['interaction_steps']}")
    print(f"  Consolidations: {stats['consolidation_count']}")
    print(f"  Replay Buffer: {stats['replay_buffer_size']}/{stats['replay_buffer_capacity']}\n")
    
    if consolidations:
        avg_cons_loss = sum(c['avg_loss'] for c in consolidations) / len(consolidations)
        print(f"Consolidation Performance:")
        print(f"  Average Loss: {avg_cons_loss:.4f}")
        print(f"  Total Consolidations: {len(consolidations)}")


def demo_comparison(steps: int = 50):
    """Compare dual-network vs single-network training."""
    print_header("Dual-Network vs Single-Network Comparison")
    
    vocab = Vocabulary([
        "apple", "banana", "orange", "fruit", "red", "yellow",
        "sweet", "sour", "tree", "plant", "food", "healthy"
    ])
    
    # Train dual-network
    print("\n📊 Training Dual-Network System...")
    dual_trainer = DualNetworkTrainer(vocab, use_dual_network=True)
    dual_losses = []
    
    for step in range(steps):
        result = dual_trainer.train_step()
        if not result.get('skip'):
            dual_losses.append(result['loss'])
    
    # Train single-network
    print("\n📊 Training Single-Network System...")
    single_trainer = DualNetworkTrainer(vocab, use_dual_network=False)
    single_losses = []
    
    for step in range(steps):
        result = single_trainer.train_step()
        if not result.get('skip'):
            single_losses.append(result['loss'])
    
    # Compare results
    print_header("Comparison Results")
    
    dual_avg = sum(dual_losses) / len(dual_losses) if dual_losses else 0
    single_avg = sum(single_losses) / len(single_losses) if single_losses else 0
    
    print(f"Dual-Network:")
    print(f"  Average Loss: {dual_avg:.4f}")
    print(f"  Final Loss: {dual_losses[-1]:.4f}" if dual_losses else "  N/A")
    
    dual_stats = dual_trainer.get_stats()
    print(f"  Consolidations: {dual_stats['consolidation_count']}")
    print(f"  Buffer Size: {dual_stats['replay_buffer_size']}\n")
    
    print(f"Single-Network:")
    print(f"  Average Loss: {single_avg:.4f}")
    print(f"  Final Loss: {single_losses[-1]:.4f}" if single_losses else "  N/A")
    
    print(f"\nDifference: {abs(dual_avg - single_avg):.4f}")


def demo_consolidation_detail():
    """Detailed demo of consolidation mechanism."""
    print_header("Consolidation Mechanism Demo")
    
    vocab = Vocabulary(["apple", "banana", "orange", "fruit"])
    trainer = DualNetworkTrainer(vocab, use_dual_network=True)
    
    print("Training for 105 steps to trigger consolidation...\n")
    
    for step in range(1, 106):
        result = trainer.train_step()
        
        if step % 25 == 0:
            stats = trainer.get_stats()
            print(f"\nStep {step}:")
            print(f"  Buffer: {stats['replay_buffer_size']}/{stats['replay_buffer_capacity']}")
            print(f"  Consolidations: {stats['consolidation_count']}")
        
        if 'consolidation' in result:
            cons = result['consolidation']
            print(f"\n{'='*60}")
            print(f"🌙 CONSOLIDATION at step {step}")
            print(f"{'='*60}")
            print(f"Status: {cons['status']}")
            print(f"Replays: {cons['replays']}")
            print(f"Average Loss: {cons.get('avg_loss', 0):.4f}")
            print(f"  - Distillation: {cons.get('avg_distill_loss', 0):.4f}")
            print(f"  - Target: {cons.get('avg_target_loss', 0):.4f}")
            print(f"Buffer Size: {cons['buffer_size']}")
            print(f"{'='*60}\n")


def main():
    """Run all demos."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dual-Network System Demo")
    parser.add_argument(
        '--demo',
        type=str,
        choices=['basic', 'comparison', 'consolidation', 'all'],
        default='basic',
        help='Which demo to run'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=20,
        help='Number of training steps for basic demo'
    )
    
    args = parser.parse_args()
    
    if args.demo == 'basic' or args.demo == 'all':
        demo_basic(args.steps)
    
    if args.demo == 'comparison' or args.demo == 'all':
        demo_comparison(50)
    
    if args.demo == 'consolidation' or args.demo == 'all':
        demo_consolidation_detail()
    
    print_header("Demo Complete")
    print("To train your own model:")
    print("  python train_dual.py --model models/my_model --steps 1000\n")


if __name__ == "__main__":
    main()
