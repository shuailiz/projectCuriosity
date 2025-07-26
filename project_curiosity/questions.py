"""Centralised question generation & sanity-checking for project_curiosity."""
from __future__ import annotations

from . import config as C
from .llm import ask

def validation_question(action: str, concept_a: str, concept_b: str, answer: str) -> str:
    """Generate validation questions specific to each action type.
    
    For relation actions, we check if the relation holds between concept_a and concept_b.
    For operation actions, we check if the answer is the correct result of the operation.
    
    Args:
        action: The action to validate
        concept_a: First concept
        concept_b: Second concept
        answer: For operation actions, the predicted result
        
    Returns:
        A question string for the LLM
    """
    # Relation actions (checking if two concepts have a relation)
    if action == "oppose":
        return f"Does '{concept_a}' oppose '{concept_b}'? Answer yes or no."
    elif action == "similar":
        return f"Is '{concept_a}' similar to '{concept_b}'? Answer yes or no."
    elif action == "include":
        return f"Does '{concept_a}' include '{concept_b}'? Answer yes or no."
    
    # Operation actions (predicting a result from applying an operation)
    elif action == "combine":
        return f"Is '{answer}' the result of combining '{concept_a}' and '{concept_b}'? Answer with 'yes' or 'no'."
    elif action == "add":
        return f"Is '{answer}' the result of adding '{concept_b}' to '{concept_a}'? Answer with 'yes' or 'no'."
    elif action == "subtract":
        return f"Is '{answer}' the result of subtracting '{concept_b}' from '{concept_a}'? Answer with 'yes' or 'no'."
    elif action == "intersect":
        return f"Is '{answer}' the intersection of '{concept_a}' and '{concept_b}'? Answer with 'yes' or 'no'."
    
    # Fallback for any other action
    return f"Is '{answer}' the correct result of {action} '{concept_a}' and '{concept_b}'? Answer with 'yes' or 'no'."

def correction_prompt(action: str, concept_a: str, concept_b: str) -> str:
    """Generate correction prompts specific to each action type.
    
    For relation actions, we ask for a concept that correctly has the relation with concept_a.
    For operation actions, we ask for the correct result of applying the operation.
    
    Args:
        action: The action to get a correction for
        concept_a: First concept
        concept_b: Second concept
        
    Returns:
        A prompt string for the LLM
    """
    # Relation actions (asking for a concept that correctly has the relation)
    if action == "oppose":
        return f"What is a concept that opposes '{concept_a}'? Answer in one word."
    elif action == "include":
        return f"What is a concept that '{concept_a}' includes? Answer in one word."
    elif action == "similar":
        return f"What is a concept that is similar to '{concept_a}'? Answer in one word."
    
    # Operation actions (asking for the correct result of the operation)
    elif action == "combine":
        return f"What is the result of combining '{concept_a}' and '{concept_b}'? Answer in one word."
    elif action == "add":
        return f"What is the result of adding '{concept_b}' to '{concept_a}'? Answer in one word."
    elif action == "subtract":
        return f"What is the result of subtracting '{concept_b}' from '{concept_a}'? Answer in one word."
    elif action == "intersect":
        return f"What is the intersection of '{concept_a}' and '{concept_b}'? Answer in one word."
    
    # Fallback for any other action
    return f"What is the result when you {action} '{concept_a}' and '{concept_b}'? Answer in one word."

def question_is_valid(question: str) -> bool:
    """Ask the LLM if the question is meaningful (quick heuristic)."""
    response = ask(
        f"Does the following question make logical sense? Answer 'yes' or 'no':\n{question}"
    ).lower()
    return response.startswith("y")
