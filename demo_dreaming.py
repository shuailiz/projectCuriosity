"""Demo: Dreaming with Inverse Pass

This script demonstrates the "dreaming" capability of the dual-network system.
By running the network backwards, we can find which concepts would produce
a given output - enabling visualization and self-consistency checks.
"""
import torch
import argparse
from pathlib import Path

from project_curiosity.language.vocabulary import Vocabulary
from project_curiosity.language.dual_network_model_language import DualNetworkModel
from project_curiosity.language.embeddings import load_pretrained, build_embedding_matrix
from project_curiosity.language import config as C


def load_model(model_dir: str):
    """Load trained dual-network model."""
    model_path = Path(model_dir)
    
    # Load vocabulary
    vocab_path = model_path / 'vocab.pkl'
    import pickle
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    # Load embeddings
    kv = load_pretrained()
    emb_matrix = build_embedding_matrix(vocab.tokens, kv).to(C.DEVICE)
    
    # Create model
    model = DualNetworkModel(C.VOCAB_SIZE, emb_matrix, freeze_embeddings=False).to(C.DEVICE)
    
    # Load checkpoint
    checkpoint_path = model_path / 'checkpoint_dual.pt'
    checkpoint = torch.load(checkpoint_path, map_location=C.DEVICE)
    
    if 'fast_learner_state_dict' in checkpoint:
        model.fast_learner.load_state_dict(checkpoint['fast_learner_state_dict'])
        model.slow_learner.load_state_dict(checkpoint['slow_learner_state_dict'])
    
    model.eval()
    return model, vocab


def forward_and_backward(model, vocab, concept_a: str, action: str, concept_b: str = None):
    """Run forward pass then backward pass to reconstruct inputs."""
    
    # Encode inputs
    a_id = vocab.encode(concept_a)
    action_id = C.ACTION_TOKENS.index(action)
    is_relation = action in C.RELATION_ACTIONS
    
    if not is_relation and concept_b:
        b_id = vocab.encode(concept_b)
    else:
        b_id = None
    
    print(f"\n{'='*80}")
    print(f"Forward and Backward Pass Demo")
    print(f"{'='*80}")
    print(f"Input: {concept_a} + {action}" + (f" + {concept_b}" if concept_b else ""))
    
    # Forward pass (both networks)
    with torch.no_grad():
        # Fast learner
        fast_logits = model.fast_learner(
            torch.tensor([a_id], device=C.DEVICE),
            torch.tensor([action_id], device=C.DEVICE),
            torch.tensor([b_id], device=C.DEVICE) if b_id is not None else None,
            is_relation=is_relation
        )
        
        # Slow learner
        slow_logits = model.slow_learner(
            torch.tensor([a_id], device=C.DEVICE),
            torch.tensor([action_id], device=C.DEVICE),
            torch.tensor([b_id], device=C.DEVICE) if b_id is not None else None,
            is_relation=is_relation
        )
        
        # Get predictions
        fast_pred_id = torch.argmax(fast_logits, dim=-1).item()
        slow_pred_id = torch.argmax(slow_logits, dim=-1).item()
        
        fast_pred = vocab.decode(fast_pred_id)
        slow_pred = vocab.decode(slow_pred_id)
        
        print(f"\nForward Pass:")
        print(f"  Fast Learner → {fast_pred}")
        print(f"  Slow Learner → {slow_pred}")
        
        # Backward pass (dreaming)
        print(f"\nBackward Pass (Dreaming):")
        
        # Fast learner backward
        fast_backward = model.fast_learner.backward_pass(fast_logits, is_relation=is_relation)
        print(f"\n  Fast Learner Reconstruction:")
        print(f"    Concept A embedding shape: {fast_backward['concept_a_emb'].shape}")
        print(f"    Action embedding shape: {fast_backward['action_emb'].shape}")
        print(f"    Concept B embedding shape: {fast_backward['concept_b_emb'].shape}")
        
        # Slow learner backward
        slow_backward = model.slow_learner.backward_pass(slow_logits, is_relation=is_relation)
        print(f"\n  Slow Learner Reconstruction:")
        print(f"    Concept A embedding shape: {slow_backward['concept_a_emb'].shape}")
        print(f"    Action embedding shape: {slow_backward['action_emb'].shape}")
        print(f"    Concept B embedding shape: {slow_backward['concept_b_emb'].shape}")
        
        # Check reconstruction quality
        original_a_emb = model.fast_learner.concept_embed(torch.tensor([a_id], device=C.DEVICE))
        fast_recon_a_emb = fast_backward['concept_a_emb']
        
        # Cosine similarity
        similarity = torch.nn.functional.cosine_similarity(original_a_emb, fast_recon_a_emb)
        print(f"\n  Reconstruction Quality (Fast Learner):")
        print(f"    Cosine similarity with original: {similarity.item():.4f}")
        
        print(f"{'='*80}\n")


def dream_from_output(model, vocab, target_concept: str, network: str = 'fast'):
    """Dream: given a target output, find which inputs would produce it."""
    
    print(f"\n{'='*80}")
    print(f"Dreaming: What inputs produce '{target_concept}'?")
    print(f"{'='*80}")
    
    # Get target concept embedding
    target_id = vocab.encode(target_concept)
    
    # Create target logits (one-hot)
    target_logits = torch.zeros(1, vocab.size, device=C.DEVICE)
    target_logits[0, target_id] = 10.0  # High confidence
    
    # Dream with selected network
    learner = model.fast_learner if network == 'fast' else model.slow_learner
    
    with torch.no_grad():
        dreams = learner.dream_concept(target_logits, top_k=10)
    
    print(f"\nTop 10 concepts that could produce '{target_concept}' ({network} learner):")
    for i, (concept_id, score) in enumerate(dreams, 1):
        concept = vocab.decode(concept_id)
        print(f"  {i}. {concept:20s} (similarity: {score:.4f})")
    
    print(f"{'='*80}\n")


def compare_networks_dreaming(model, vocab, target_concept: str):
    """Compare dreaming between fast and slow learners."""
    
    print(f"\n{'='*80}")
    print(f"Network Comparison: Dreaming about '{target_concept}'")
    print(f"{'='*80}")
    
    target_id = vocab.encode(target_concept)
    target_logits = torch.zeros(1, vocab.size, device=C.DEVICE)
    target_logits[0, target_id] = 10.0
    
    with torch.no_grad():
        fast_dreams = model.fast_learner.dream_concept(target_logits, top_k=5)
        slow_dreams = model.slow_learner.dream_concept(target_logits, top_k=5)
    
    print(f"\nFast Learner (Hippocampus) Dreams:")
    for i, (concept_id, score) in enumerate(fast_dreams, 1):
        concept = vocab.decode(concept_id)
        print(f"  {i}. {concept:20s} (similarity: {score:.4f})")
    
    print(f"\nSlow Learner (Cortex) Dreams:")
    for i, (concept_id, score) in enumerate(slow_dreams, 1):
        concept = vocab.decode(concept_id)
        print(f"  {i}. {concept:20s} (similarity: {score:.4f})")
    
    # Check overlap
    fast_concepts = {vocab.decode(cid) for cid, _ in fast_dreams}
    slow_concepts = {vocab.decode(cid) for cid, _ in slow_dreams}
    overlap = fast_concepts & slow_concepts
    
    print(f"\nOverlap: {len(overlap)}/5 concepts")
    if overlap:
        print(f"  Shared: {', '.join(overlap)}")
    
    print(f"{'='*80}\n")


def self_consistency_check(model, vocab, concept_a: str, action: str, concept_b: str = None):
    """Check self-consistency: forward → backward → forward."""
    
    print(f"\n{'='*80}")
    print(f"Self-Consistency Check")
    print(f"{'='*80}")
    print(f"Input: {concept_a} + {action}" + (f" + {concept_b}" if concept_b else ""))
    
    # Encode inputs
    a_id = vocab.encode(concept_a)
    action_id = C.ACTION_TOKENS.index(action)
    is_relation = action in C.RELATION_ACTIONS
    
    if not is_relation and concept_b:
        b_id = vocab.encode(concept_b)
    else:
        b_id = None
    
    with torch.no_grad():
        # Forward pass 1
        logits_1 = model.fast_learner(
            torch.tensor([a_id], device=C.DEVICE),
            torch.tensor([action_id], device=C.DEVICE),
            torch.tensor([b_id], device=C.DEVICE) if b_id is not None else None,
            is_relation=is_relation
        )
        pred_1 = vocab.decode(torch.argmax(logits_1, dim=-1).item())
        
        # Backward pass
        backward_result = model.fast_learner.backward_pass(logits_1, is_relation=is_relation)
        reconstructed_a_emb = backward_result['concept_a_emb']
        
        # Find nearest concept to reconstructed embedding
        all_embeddings = model.fast_learner.concept_embed.weight
        similarities = torch.nn.functional.cosine_similarity(
            reconstructed_a_emb,
            all_embeddings,
            dim=-1
        )
        reconstructed_a_id = torch.argmax(similarities).item()
        reconstructed_a = vocab.decode(reconstructed_a_id)
        
        # Forward pass 2 with reconstructed input
        logits_2 = model.fast_learner(
            torch.tensor([reconstructed_a_id], device=C.DEVICE),
            torch.tensor([action_id], device=C.DEVICE),
            torch.tensor([b_id], device=C.DEVICE) if b_id is not None else None,
            is_relation=is_relation
        )
        pred_2 = vocab.decode(torch.argmax(logits_2, dim=-1).item())
        
        # Compare
        print(f"\nForward Pass 1:")
        print(f"  Input: {concept_a}")
        print(f"  Output: {pred_1}")
        
        print(f"\nBackward Pass:")
        print(f"  Reconstructed Input: {reconstructed_a}")
        print(f"  Similarity to original: {similarities[a_id].item():.4f}")
        
        print(f"\nForward Pass 2:")
        print(f"  Input: {reconstructed_a}")
        print(f"  Output: {pred_2}")
        
        print(f"\nConsistency:")
        print(f"  Input preserved: {concept_a == reconstructed_a}")
        print(f"  Output preserved: {pred_1 == pred_2}")
        
        # Logits similarity
        logits_similarity = torch.nn.functional.cosine_similarity(logits_1, logits_2)
        print(f"  Logits similarity: {logits_similarity.item():.4f}")
    
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Demo: Dreaming with Inverse Pass")
    parser.add_argument('--model', type=str, required=True, help='Path to model directory')
    parser.add_argument('--concept-a', type=str, default='apple', help='First concept')
    parser.add_argument('--action', type=str, default='similar', help='Action')
    parser.add_argument('--concept-b', type=str, default=None, help='Second concept (for operations)')
    parser.add_argument('--target', type=str, default='fruit', help='Target concept for dreaming')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("Dreaming Demo: Inverse Pass with Leaky ReLU")
    print("="*80)
    print("\nLoading model...")
    
    model, vocab = load_model(args.model)
    
    print(f"Model loaded from {args.model}")
    print(f"Vocabulary size: {vocab.size}")
    print(f"Fast learner parameters: {sum(p.numel() for p in model.fast_learner.parameters()):,}")
    print(f"Slow learner parameters: {sum(p.numel() for p in model.slow_learner.parameters()):,}")
    
    # Demo 1: Forward and backward pass
    forward_and_backward(model, vocab, args.concept_a, args.action, args.concept_b)
    
    # Demo 2: Dream from target output
    dream_from_output(model, vocab, args.target, network='fast')
    dream_from_output(model, vocab, args.target, network='slow')
    
    # Demo 3: Compare networks
    compare_networks_dreaming(model, vocab, args.target)
    
    # Demo 4: Self-consistency check
    self_consistency_check(model, vocab, args.concept_a, args.action, args.concept_b)
    
    print("\n" + "="*80)
    print("Demo Complete!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
