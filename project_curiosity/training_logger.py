"""Training logger for recording questions and feedback."""

import json
import os
from datetime import datetime
from typing import Optional


class TrainingLogger:
    """Logs training interactions to a JSON file in the model directory."""
    
    def __init__(self, model_dir: str):
        """Initialize logger with model directory.
        
        Args:
            model_dir: Path to model directory
        """
        self.model_dir = model_dir
        self.log_file = os.path.join(model_dir, 'training_log.jsonl')
        
    def log_interaction(
        self,
        step: int,
        action: str,
        concept_a: str,
        concept_b: str,
        model_prediction: str,
        validation_question: Optional[str] = None,
        validation_response: Optional[str] = None,
        is_valid_question: bool = True,
        correction_question: Optional[str] = None,
        correction_response: Optional[str] = None,
        is_correct: bool = False,
        feedback_source: str = "llm",  # "llm" or "human"
        concept_loss: Optional[float] = None,
        action_loss: Optional[float] = None,
    ):
        """Log a single training interaction.
        
        Args:
            step: Training step number
            action: Action type (e.g., "oppose", "combine")
            concept_a: First concept
            concept_b: Second concept (empty for relation actions)
            model_prediction: What the model predicted
            validation_question: Question asked to validate the prediction
            validation_response: Response to validation question
            is_valid_question: Whether the question itself was valid
            correction_question: Question asked for correction (if prediction was wrong)
            correction_response: Corrected answer
            is_correct: Whether the prediction was correct
            feedback_source: "llm" or "human"
            concept_loss: Concept loss value
            action_loss: Action loss value
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "action": action,
            "concept_a": concept_a,
            "concept_b": concept_b,
            "model_prediction": model_prediction,
            "validation_question": validation_question,
            "validation_response": validation_response,
            "is_valid_question": is_valid_question,
            "correction_question": correction_question,
            "correction_response": correction_response,
            "is_correct": is_correct,
            "feedback_source": feedback_source,
            "concept_loss": concept_loss,
            "action_loss": action_loss,
        }
        
        # Append to log file (JSONL format - one JSON object per line)
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def get_stats(self) -> dict:
        """Get statistics from the training log.
        
        Returns:
            Dictionary with training statistics
        """
        if not os.path.exists(self.log_file):
            return {
                "total_steps": 0,
                "correct_predictions": 0,
                "incorrect_predictions": 0,
                "invalid_questions": 0,
            }
        
        total_steps = 0
        correct = 0
        incorrect = 0
        invalid_questions = 0
        
        with open(self.log_file, 'r') as f:
            for line in f:
                entry = json.loads(line)
                total_steps += 1
                if entry["is_correct"]:
                    correct += 1
                else:
                    incorrect += 1
                if not entry["is_valid_question"]:
                    invalid_questions += 1
        
        return {
            "total_steps": total_steps,
            "correct_predictions": correct,
            "incorrect_predictions": incorrect,
            "invalid_questions": invalid_questions,
            "accuracy": correct / total_steps if total_steps > 0 else 0.0,
        }
