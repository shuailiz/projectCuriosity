"""Training loop for project_curiosity."""
from __future__ import annotations

import json
import time
from typing import Dict

import torch
import torch.nn.functional as F

from . import config as C
from .vocabulary import Vocabulary
from .model import ConceptActionModel
from .llm import ask
from . import questions as Q
from . import actions

class Trainer:
    def __init__(self, vocab: Vocabulary):
        self.vocab = vocab
        from .embeddings import load_pretrained, build_embedding_matrix
        kv = load_pretrained()  # may download once
        emb_matrix = build_embedding_matrix(self.vocab.tokens, kv).to(C.DEVICE)
        self.model = ConceptActionModel(C.VOCAB_SIZE, emb_matrix, freeze_embeddings=False).to(C.DEVICE)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=C.LEARNING_RATE)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def _sample_pair(self):
        return self.vocab.random_concept(), self.vocab.random_concept()

    def train_step(self) -> Dict[str, str]:
        a_id, b_id = self._sample_pair()
        # propose action
        action_logits_prop = self.model.propose_action(
            torch.tensor([a_id], device=C.DEVICE),
            torch.tensor([b_id], device=C.DEVICE),
        )
        act_id = torch.multinomial(F.softmax(action_logits_prop, dim=-1), 1).item()
        act_tok = C.ACTION_TOKENS[act_id]

        # For decompose we ignore b and use PAD
        if act_tok == "decompose":
            b_id = 1  # PAD token index
        a_tok, b_tok = self.vocab.decode(a_id), self.vocab.decode(b_id)

        concept_logits, action_logits = self.model(
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
        
        # Get appropriate handler for this action
        handler = actions.get_handler(act_tok)
        
        # Handle action-specific logic
        result = handler(
            model_output=model_output,
            concept_a=a_tok,
            concept_b=b_tok,
            action=act_tok,
            vocab_decode_fn=self.vocab.decode,
            vocab_add_fn=self.vocab.add,
        )
        
        # Handle invalid questions
        if result.get("skip", False):
            # If there's an action loss, apply it
            if "action_loss" in result:
                action_loss = result["action_loss"]
                self.opt.zero_grad()
                action_loss.backward()
                self.opt.step()
                return {"skip": True, "action_loss": float(action_loss.item())}
            return {"skip": True}
        
        # Apply gradients
        loss = result["loss"]
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        
        # Return training info
        return dict(
            concept_a=a_tok,
            concept_b=b_tok,
            action=act_tok,
            model_answer=result["model_answer"],
            is_correct=result["is_correct"],
            correct_answer=result["correct_answer"],
            loss=float(loss.item()),
        )

def demo(steps=10):
    vocab = Vocabulary(["apple", "banana", "orange", "fruit"])
    tr = Trainer(vocab)
    for s in range(1, steps + 1):
        print(json.dumps({"step": s, **tr.train_step()}, ensure_ascii=False))
        time.sleep(0.5)

if __name__ == "__main__":
    demo(20)
