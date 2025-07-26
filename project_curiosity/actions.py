"""Action-specific logic for project_curiosity."""
from __future__ import annotations

import torch
import torch.nn.functional as F
import random
from typing import Dict, List, Tuple, Any, Callable

from . import config as C
from .llm import ask
from . import questions as Q


# Define action categories
RELATION_ACTIONS = ["oppose", "similar", "include"]
OPERATION_ACTIONS = ["combine", "add", "subtract", "intersect"]


def handle_relation_action(
    model_output: Dict[str, torch.Tensor],
    concept_a: str,
    concept_b: str,
    action: str,
    vocab_decode_fn,
    vocab_add_fn,
    mock_question_is_valid=None,
    mock_ask=None,
) -> Dict[str, Any]:
    """Handle relation actions that check if two concepts are related.
    
    For relation actions like 'oppose', 'similar', 'include', 'intersect',
    the model is validating whether concept_a and concept_b have the specified relation.
    
    Args:
        model_output: Output tensors from the model
        concept_a: First concept string
        concept_b: Second concept string
        action: Action string (e.g., 'oppose', 'similar')
        vocab_decode_fn: Function to decode token IDs to strings
        vocab_add_fn: Function to add new tokens to vocabulary
        mock_question_is_valid: Optional mock function for question validation
        mock_ask: Optional mock function for LLM queries
        
    Returns:
        Dictionary with loss, predictions, and other information
    """
    concept_logits = model_output["concept_logits"]
    action_logits_prop = model_output["action_logits_prop"]
    act_id = model_output["act_id"]
    
    # Validate with LLM
    question = Q.validation_question(action, concept_a, concept_b, "")
    question_is_valid_fn = mock_question_is_valid if mock_question_is_valid else Q.question_is_valid
    if not question_is_valid_fn(question):
        # For invalid questions, penalize the action prediction using uniform distribution over other actions
        # Create a target distribution with uniform probability for all actions except the current one
        num_actions = len(C.ACTION_TOKENS)
        target_dist = torch.ones(num_actions, device=C.DEVICE) / (num_actions - 1)
        target_dist[act_id] = 0.0  # Zero probability for the current action
        
        # Compute KL divergence loss between model output and target distribution
        log_probs = F.log_softmax(action_logits_prop, dim=-1)
        action_loss = F.kl_div(log_probs, target_dist.unsqueeze(0), reduction='batchmean')
        return {
            "skip": True,
            "action_loss": C.ACTION_LOSS_WEIGHT * action_loss * 1.5  # Penalize more for invalid questions
        }
    
    ask_fn = mock_ask if mock_ask else ask
    correct = ask_fn(question).lower().startswith("y")
    
    # For relation actions, we need to compute loss between concept_logits and concept_id
    # If the relation is correct, we want to predict concept_b
    # If not, we need to get a concept that correctly has the relation with concept_a
    if correct:
        # Use concept_b as the target since the relation is correct
        # We need to convert concept_b string to its ID in the vocabulary
        target_id = None
        for i in range(C.VOCAB_SIZE):
            if vocab_decode_fn(i) == concept_b:
                target_id = i
                break
        
        if target_id is None:
            # If concept_b is not in vocabulary, add it
            target_id = vocab_add_fn(concept_b)
    else:
        # Get a concept that correctly has the relation with concept_a
        correct_concept = ask_fn(Q.correction_prompt(action, concept_a, concept_b))
        # Add to vocabulary if needed
        target_id = vocab_add_fn(correct_concept)
    
    # Compute concept prediction loss
    concept_loss = F.cross_entropy(concept_logits, torch.tensor([target_id], device=C.DEVICE))
    
    # Compute action prediction loss - for valid questions, we always use the actual action ID
    action_loss = F.cross_entropy(action_logits_prop, torch.tensor([act_id], device=C.DEVICE))
    
    # If the relation is incorrect, increase the concept loss to penalize the model
    if not correct:
        concept_loss = concept_loss * 1.5
    
    # Total loss is a combination of concept and action losses
    total_loss = concept_loss + C.ACTION_LOSS_WEIGHT * action_loss
    
    return {
        "loss": total_loss,
        "is_correct": correct,
        "relation_result": "True" if correct else "False",
        "target_ids": [target_id],
        "correct_concept": vocab_decode_fn(target_id),
    }


def handle_operation_action(
    model_output: Dict[str, torch.Tensor],
    concept_a: str,
    concept_b: str,
    action: str,
    vocab_decode_fn,
    vocab_add_fn,
    mock_question_is_valid=None,
    mock_ask=None,
) -> Dict[str, Any]:
    """Handle operation actions that produce a new concept from two inputs.
    
    For operation actions like 'combine', 'add', 'subtract',
    the model predicts a new concept resulting from applying the operation.
    
    Args:
        model_output: Output tensors from the model
        concept_a: First concept string
        concept_b: Second concept string
        action: Action string (e.g., 'combine', 'add')
        vocab_decode_fn: Function to decode token IDs to strings
        vocab_add_fn: Function to add new tokens to vocabulary
        mock_question_is_valid: Optional mock function for question validation
        mock_ask: Optional mock function for LLM queries
        
    Returns:
        Dictionary with loss, predictions, and other information
    """
    concept_logits = model_output["concept_logits"]
    action_logits_prop = model_output["action_logits_prop"]
    act_id = model_output["act_id"]
    
    # Sample one token
    probs = F.softmax(concept_logits, dim=-1)
    ans_id = torch.multinomial(probs, 1).item()
    ans_tok = vocab_decode_fn(ans_id)
    
    # Validate with LLM
    question = Q.validation_question(action, concept_a, concept_b, ans_tok)
    question_is_valid_fn = mock_question_is_valid if mock_question_is_valid else Q.question_is_valid
    if not question_is_valid_fn(question):
        # For invalid questions, penalize the action prediction using uniform distribution over other actions
        # Create a target distribution with uniform probability for all actions except the current one
        num_actions = len(C.ACTION_TOKENS)
        target_dist = torch.ones(num_actions, device=C.DEVICE) / (num_actions - 1)
        target_dist[act_id] = 0.0  # Zero probability for the current action
        
        # Compute KL divergence loss between model output and target distribution
        log_probs = F.log_softmax(action_logits_prop, dim=-1)
        action_loss = F.kl_div(log_probs, target_dist.unsqueeze(0), reduction='batchmean')
        return {
            "skip": True,
            "action_loss": C.ACTION_LOSS_WEIGHT * action_loss * 1.5  # Penalize more for invalid questions
        }
    
    ask_fn = mock_ask if mock_ask else ask
    correct = ask_fn(question).lower().startswith("y")
    
    if correct:
        # Model's answer is correct, use the sampled token ID
        target_id = ans_id
    else:
        # Get correction from LLM
        correct_concept = ask_fn(Q.correction_prompt(action, concept_a, concept_b))
        # Always add to vocabulary to ensure we have the correct concept
        target_id = vocab_add_fn(correct_concept)
    
    # Compute concept prediction loss
    concept_loss = F.cross_entropy(concept_logits, torch.tensor([target_id], device=C.DEVICE))
    if not correct and ans_id != 0:  # Not UNK
        concept_loss = concept_loss * 1.5
    
    # Compute action prediction loss - for valid questions, we always use the actual action ID
    action_loss = F.cross_entropy(action_logits_prop, torch.tensor([act_id], device=C.DEVICE))
    
    # Total loss is a combination of concept and action losses
    total_loss = concept_loss + C.ACTION_LOSS_WEIGHT * action_loss
    
    return {
        "loss": total_loss,
        "model_answer": ans_tok,
        "is_correct": correct,
        "correct_answer": vocab_decode_fn(target_id),
        "target_ids": [target_id],
    }


def get_handler(action: str, mock_question_is_valid=None, mock_ask=None) -> Callable:
    """Return the appropriate handler function for an action.
    
    Args:
        action: The action token string
        mock_question_is_valid: Optional mock function for question validation
        mock_ask: Optional mock function for LLM queries
        
    Returns:
        A handler function that will be called with the appropriate arguments
    """
    if action in RELATION_ACTIONS:
        return lambda **kwargs: handle_relation_action(
            **kwargs, 
            mock_question_is_valid=mock_question_is_valid, 
            mock_ask=mock_ask
        )
    else:  # Operation actions or any other action
        return lambda **kwargs: handle_operation_action(
            **kwargs, 
            mock_question_is_valid=mock_question_is_valid, 
            mock_ask=mock_ask
        )
