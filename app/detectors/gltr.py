"""GLTR: rank of each actual token under GPT-2, bucketed top-10/100/1000/rest."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .. import config
from .device import pick_device

BUCKETS = (10, 100, 1000)


class GLTR:
    def __init__(self):
        self.device = pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(config.GLTR_MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(config.GLTR_MODEL).to(self.device).eval()
        self.max_len = self.model.config.n_positions

    @torch.inference_mode()
    def analyze(self, text: str) -> dict:
        ids = self.tokenizer(text, return_tensors="pt").input_ids[:, : self.max_len].to(self.device)
        logits = self.model(ids).logits[0, :-1]
        targets = ids[0, 1:]
        target_logit = logits.gather(-1, targets[:, None])
        ranks = (logits > target_logit).sum(-1).add(1).tolist()

        tokens = [{"text": self.tokenizer.decode(ids[0, :1]), "rank": 0, "bucket": 0}]
        counts = [0, 0, 0, 0]
        for tok_id, rank in zip(targets.tolist(), ranks):
            bucket = next((i for i, b in enumerate(BUCKETS) if rank <= b), 3)
            counts[bucket] += 1
            tokens.append({"text": self.tokenizer.decode([tok_id]), "rank": rank, "bucket": bucket})
        n = max(len(ranks), 1)
        return {"tokens": tokens, "fractions": [c / n for c in counts]}
