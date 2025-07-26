"""Mock test for project_curiosity without LLM calls."""
import torch
import torch.nn.functional as F
from project_curiosity.vocabulary import Vocabulary
from project_curiosity.model import ConceptActionModel
import project_curiosity.config as C
from project_curiosity import actions
import random

# Mock the LLM functions
def mock_ask(question: str) -> str:
    """Mock LLM response."""
    if "make logical sense" in question:
        return "yes"  # All questions are valid
    if "decompose" in question:
        if "yes or no" in question:
            return "yes"  # Always correct
        return "part1, part2"  # Decomposition answer
    if "yes or no" in question:
        return "yes"  # Always correct
    return "result"  # Generic answer

def mock_question_is_valid(question: str) -> bool:
    """Mock question validation."""
    return True  # All questions are valid

# Test model initialization and forward pass
def test_model():
    vocab = Vocabulary(["apple", "banana", "orange", "fruit"])
    model = ConceptActionModel(C.VOCAB_SIZE).to(C.DEVICE)
    
    # Test propose_action
    a_id, b_id = 2, 3  # orange, fruit
    action_logits_prop = model.propose_action(
        torch.tensor([a_id], device=C.DEVICE),
        torch.tensor([b_id], device=C.DEVICE),
    )
    print(f"Action proposal logits shape: {action_logits_prop.shape}")
    
    # Test forward pass
    act_id = 0  # oppose
    concept_logits = model(
        torch.tensor([a_id], device=C.DEVICE),
        torch.tensor([act_id], device=C.DEVICE),
        torch.tensor([b_id], device=C.DEVICE),
    )
    print(f"Concept logits shape: {concept_logits.shape}")
    
    # Sample from distributions
    action_probs = torch.nn.functional.softmax(action_logits_prop, dim=-1)
    concept_probs = torch.nn.functional.softmax(concept_logits, dim=-1)
    
    sampled_action = torch.multinomial(action_probs, 1).item()
    sampled_concept = torch.multinomial(concept_probs, 1).item()
    
    print(f"Sampled action: {C.ACTION_TOKENS[sampled_action]}")
    print(f"Sampled concept: {vocab.decode(sampled_concept)}")
    
    return "Model test passed!"

def test_training_loop():
    """Test a few training steps with mocked LLM responses."""
    # Create vocabulary and model
    vocab = Vocabulary(["apple", "banana", "orange", "fruit", "vegetable", "healthy"])
    model = ConceptActionModel(C.VOCAB_SIZE).to(C.DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=C.LEARNING_RATE)
    
    # Run a few training steps
    for step in range(3):
        # Sample concepts
        a_id, b_id = random.randrange(4), random.randrange(4)
        a_tok, b_tok = vocab.decode(a_id), vocab.decode(b_id)
        
        # Propose action
        action_logits_prop = model.propose_action(
            torch.tensor([a_id], device=C.DEVICE),
            torch.tensor([b_id], device=C.DEVICE),
        )
        act_id = torch.multinomial(F.softmax(action_logits_prop, dim=-1), 1).item()
        act_tok = C.ACTION_TOKENS[act_id]
        
        # No special handling needed for any action now that decompose is removed
        
        # Forward pass
        concept_logits = model(
            torch.tensor([a_id], device=C.DEVICE),
            torch.tensor([act_id], device=C.DEVICE),
            torch.tensor([b_id], device=C.DEVICE),
        )
        
        # Prepare model outputs for action handler
        model_output = {
            "concept_logits": concept_logits,
            "action_logits_prop": action_logits_prop,
            "act_id": act_id,
        }
        
        # Get appropriate handler for this action with mocks
        handler = actions.get_handler(
            act_tok,
            mock_question_is_valid=mock_question_is_valid,
            mock_ask=mock_ask
        )
        
        # Handle action-specific logic
        result = handler(
            model_output=model_output,
            concept_a=a_tok,
            concept_b=b_tok,
            action=act_tok,
            vocab_decode_fn=vocab.decode,
            vocab_add_fn=vocab.add,
        )
        
        # Apply gradients
        if result.get("skip", False):
            if "action_loss" in result:
                action_loss = result["action_loss"]
                opt.zero_grad()
                action_loss.backward()
                opt.step()
                print(f"Step {step+1}: Skipped with action loss {float(action_loss.item()):.4f}")
            else:
                print(f"Step {step+1}: Skipped without action loss")
            continue
        
        loss = result["loss"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        # Handle different result formats for relation vs operation actions
        if 'model_answer' in result:
            # Operation action result
            print(f"Step {step+1}: {a_tok} {act_tok} {b_tok} = {result['model_answer']}")
        else:
            # Relation action result
            print(f"Step {step+1}: {a_tok} {act_tok} {b_tok} = {result['relation_result']}")
        
        print(f"  Correct: {result['is_correct']}, Loss: {float(loss.item()):.4f}")
        
    return "Training test completed successfully!"

if __name__ == "__main__":
    print(test_model())
    print("\n" + "-"*50 + "\n")
    print(test_training_loop())
