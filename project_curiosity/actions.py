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


def handle_action(
    model_output,
    concept_a,
    concept_b,
    vocab_decode_fn,
    vocab_add_fn,
    mock_question_is_valid=None,
    mock_ask=None,
    human_feedback=False,
    logger=None,
    step=None,
) -> Dict[str, Any]:
    """Unified handler for both relation and operation actions.
    
    This handler processes both relation actions (oppose, similar, include) and
    operation actions (combine, add, subtract, intersect), reducing code duplication.
    
    Args:
        model_output: Output tensors from the model
        concept_a: First concept string
        concept_b: Second concept string
        vocab_decode_fn: Function to decode token IDs to strings
        vocab_add_fn: Function to add new tokens to vocabulary
        mock_question_is_valid: Optional mock function for question validation
        mock_ask: Optional mock function for LLM queries
        human_feedback: If True, prompt for human feedback instead of using LLM
        logger: Optional TrainingLogger instance for logging interactions
        step: Optional training step number for logging
        
    Returns:
        Dictionary with loss, predictions, and other information
    """
    concept_logits = model_output["concept_logits"]
    action_logits_prop = model_output["action_logits_prop"]
    act_id = model_output["act_id"]
    
    # Decode action from act_id
    action = C.ACTION_TOKENS[act_id]
    is_relation = action in RELATION_ACTIONS
    
    # Generate sequence of tokens by sampling top N distinct tokens
    probs = F.softmax(concept_logits, dim=-1)
    
    # Look up FINISH token ID from vocabulary (do this once)
    finish_id = vocab_add_fn(C.FINISH_TOKEN)
    
    # Mask out PAD token (ID 1) - it has no semantic meaning
    masked_probs = probs.clone()
    masked_probs[0, 1] = 0.0  # <PAD>
    
    # Get top N tokens (sorted by probability)
    top_k = min(C.MAX_SEQUENCE_LENGTH + 1, probs.shape[-1])  # +1 to account for possible FINISH token
    top_probs, top_indices = torch.topk(masked_probs, top_k, dim=-1)
    
    # Sample from top tokens and build sequence
    pred_ids = []
    pred_tokens = []
    
    for idx in top_indices[0]:  # Iterate through top tokens
        pred_id = idx.item()
        
        # Stop if we hit FINISH token
        if pred_id == finish_id:
            break
        
        # Add to sequence
        pred_ids.append(pred_id)
        pred_tokens.append(vocab_decode_fn(pred_id))
        
        # Stop if we've collected enough tokens
        if len(pred_ids) >= C.MAX_SEQUENCE_LENGTH:
            break
    
    # Join tokens into concept (order doesn't matter for validation)
    pred_concept = " ".join(pred_tokens) if pred_tokens else C.UNKNOWN_TOKEN
    
    # Check if the predicted concept is a special token or empty
    # Also check if ANY token in the sequence is a special token (shouldn't happen with filtering, but safety check)
    is_unknown = (not pred_ids or 
                  pred_concept == C.UNKNOWN_TOKEN or
                  any(vocab_decode_fn(id).startswith("<") for id in pred_ids))
    
    ask_fn = mock_ask if mock_ask else ask
    correct = False
    
    # For unknown/special token predictions, use "something" in validation questions
    validation_concept = "something" if is_unknown else pred_concept
    
    # For relation actions, the predicted concept is concept_b
    # For operation actions, the predicted concept is the answer
    if is_relation:
        question = Q.validation_question(action, concept_a, validation_concept, "")
    else:
        question = Q.validation_question(action, concept_a, concept_b, validation_concept)
    
    # Check if the question is valid
    is_valid = True
    if human_feedback:
        print(f"\nHuman Question Validation Required:")
        print(f"Question: {question}")
        user_input = input("Is this a valid question? (yes/no): ").strip().lower()
        is_valid = user_input.startswith("y")
    else:
        question_is_valid_fn = mock_question_is_valid if mock_question_is_valid else Q.question_is_valid
        is_valid = question_is_valid_fn(question)
    
    if not is_valid:
        # Handle invalid questions (same for both types)
        num_actions = len(C.ACTION_TOKENS)
        target_dist = torch.ones(num_actions, device=C.DEVICE) / (num_actions - 1)
        target_dist[act_id] = 0.0  # Zero probability for the current action
        
        # Compute KL divergence loss between model output and target distribution
        log_probs = F.log_softmax(action_logits_prop, dim=-1)
        action_loss = F.kl_div(log_probs, target_dist.unsqueeze(0), reduction='batchmean')
        
        # Log invalid question if logger is provided
        if logger is not None and step is not None:
            logger.log_interaction(
                step=step,
                action=action,
                concept_a=concept_a,
                concept_b=concept_b,
                model_prediction=pred_concept,
                validation_question=question,
                validation_response=None,
                is_valid_question=False,
                correction_question=None,
                correction_response=None,
                is_correct=False,
                feedback_source="human" if human_feedback else "llm",
                concept_loss=None,
                action_loss=action_loss.item(),
            )
        
        return {
            "skip": True,
            "action_loss": C.ACTION_LOSS_WEIGHT * action_loss * 1.5  # Penalize more for invalid questions
        }
    
    # Check if the prediction is correct (skip for unknown predictions)
    if is_unknown:
        # For unknown predictions, always consider them incorrect and skip to correction
        correct = False
    else:
        # For normal predictions, validate the answer
        if human_feedback:
            print(f"\nHuman Validation Required:")
            print(question)
            user_input = input("Is this correct? (yes/no): ").strip().lower()
            correct = user_input.startswith("y")
        else:
            correct = ask_fn(question).lower().startswith("y")
    
    # Handle target concept based on action type and correctness
    if correct:
        # For both relation and operation actions, if correct, reinforce the model's prediction
        target_ids = pred_ids.copy()  # Use the full predicted sequence
    else:
        # Get correction from human or LLM
        correction_prompt = Q.correction_prompt(action, concept_a, concept_b)
        if human_feedback:
            print(f"\nHuman Correction Required:")
            print(correction_prompt)
            print("(You can provide multiple words separated by spaces)")
            correct_concept = input("Please provide the correct concept: ").strip()
        else:
            # Get correction from LLM
            correct_concept = ask_fn(correction_prompt)
        
        # Parse correction into individual words and add to vocabulary
        correct_words = correct_concept.split()
        target_ids = []
        
        for word in correct_words:
            # vocab_add_fn returns existing ID if word exists, or adds it if new
            word_id = vocab_add_fn(word)
            target_ids.append(word_id)
        
        # Add FINISH token to indicate end of sequence (reuse finish_id from earlier)
        target_ids.append(finish_id)
    
    # Compute concept loss
    # For sequences, create a target probability distribution over all tokens
    # This is more accurate than averaging individual losses
    
    if len(target_ids) > 1:
        # Multi-token sequence: create uniform distribution over target tokens
        target_probs = torch.zeros((1, C.VOCAB_SIZE), device=C.DEVICE)
        for target_id in target_ids:
            target_probs[0, target_id] = 1.0 / len(target_ids)  # Uniform distribution
        
        # Compute cross-entropy with target distribution
        concept_loss = F.cross_entropy(concept_logits, target_probs)
    else:
        # Single token: standard loss
        target_id = target_ids[0] if target_ids else 0
        concept_loss = F.cross_entropy(concept_logits, torch.tensor([target_id], device=C.DEVICE))
    
    # Action loss (always single token)
    action_loss = F.cross_entropy(action_logits_prop, torch.tensor([act_id], device=C.DEVICE))
    
    # Apply penalty for incorrect predictions
    if not correct:
        # For relation actions, always apply penalty
        # For operation actions, only apply if not UNK
        if is_relation or (pred_ids and pred_ids[0] != 0):
            concept_loss = concept_loss * 1.5
    
    total_loss = concept_loss + C.ACTION_LOSS_WEIGHT * action_loss
    
    # Log the interaction if logger is provided
    if logger is not None and step is not None:
        # Determine validation response
        validation_response = "yes" if correct else "no"
        
        # Get correction response if prediction was incorrect
        correction_response = None
        if not correct:
            correction_response = " ".join([vocab_decode_fn(tid) for tid in target_ids if vocab_decode_fn(tid) != C.FINISH_TOKEN])
        
        logger.log_interaction(
            step=step,
            action=action,
            concept_a=concept_a,
            concept_b=concept_b,
            model_prediction=pred_concept,
            validation_question=question if is_valid else None,
            validation_response=validation_response if is_valid else None,
            is_valid_question=is_valid,
            correction_question=correction_prompt if not correct else None,
            correction_response=correction_response,
            is_correct=correct,
            feedback_source="human" if human_feedback else "llm",
            concept_loss=concept_loss.item(),
            action_loss=action_loss.item(),
        )
    
    # Prepare return values (same for both relation and operation actions)
    result = {
        "loss": total_loss,
        "is_correct": correct,
        "target_ids": target_ids,
        "model_answer": pred_concept,
        "correct_answer": " ".join([vocab_decode_fn(tid) for tid in target_ids if vocab_decode_fn(tid) != C.FINISH_TOKEN]),
    }
    
    return result


def get_handler(action: str, mock_question_is_valid=None, mock_ask=None, human_feedback=False) -> Callable:
    """Return the unified action handler function.
    
    Args:
        action: The action token string
        mock_question_is_valid: Optional mock function for question validation
        mock_ask: Optional mock function for LLM queries
        human_feedback: If True, use human feedback instead of LLM
        
    Returns:
        A handler function that will be called with the appropriate arguments
    """
    return lambda **kwargs: handle_action(
        **kwargs,
        mock_question_is_valid=mock_question_is_valid,
        mock_ask=mock_ask,
        human_feedback=human_feedback
    )
