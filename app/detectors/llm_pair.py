"""Binoculars and Fast-DetectGPT, sharing one base/instruct model pair.

Both methods need the same two forward passes: base model (M1, "observer" /
"sampling" model) and instruct model (M2, "performer" / "scoring" model).
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .. import config
from .device import pick_device, pick_dtype


@dataclass
class PairScores:
    binoculars: float
    fast_detectgpt: float
    n_tokens: int
    # per-token pieces so callers can re-aggregate over sentence spans
    token_offsets: list[tuple[int, int]]
    log_ppl: torch.Tensor   # -log p2(x_i), shape [n]
    x_ppl: torch.Tensor     # cross-entropy of p1 against log p2, shape [n]


class LLMPair:
    def __init__(self):
        self.device = pick_device()
        dtype = pick_dtype(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
        self.base = AutoModelForCausalLM.from_pretrained(config.BASE_MODEL, torch_dtype=dtype).to(self.device).eval()
        self.instruct = AutoModelForCausalLM.from_pretrained(config.INSTRUCT_MODEL, torch_dtype=dtype).to(self.device).eval()

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False).input_ids)

    @torch.inference_mode()
    def score(self, text: str) -> PairScores:
        enc = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=config.MAX_TOKENS,
            return_offsets_mapping=True, add_special_tokens=False,
        )
        ids = enc.input_ids.to(self.device)
        offsets = [tuple(o) for o in enc.offset_mapping[0].tolist()]

        # Logits at position i predict token i+1.
        logits1 = self.base(ids).logits[0, :-1].float()
        logits2 = self.instruct(ids).logits[0, :-1].float()
        targets = ids[0, 1:]

        logp1 = F.log_softmax(logits1, dim=-1)
        logp2 = F.log_softmax(logits2, dim=-1)
        p1 = logp1.exp()

        # --- Binoculars: log-PPL under M2 / cross-PPL(M1 -> M2) ---
        nll2 = -logp2.gather(-1, targets[:, None]).squeeze(-1)
        x_ent = -(p1 * logp2).sum(-1)
        binoculars = (nll2.mean() / x_ent.mean()).item()

        # --- Fast-DetectGPT (analytic): sampling dist M1, scoring dist M2 ---
        ll = -nll2
        mean_ref = (p1 * logp2).sum(-1)
        var_ref = (p1 * logp2.square()).sum(-1) - mean_ref.square()
        fast = ((ll.sum() - mean_ref.sum()) / var_ref.sum().clamp_min(1e-6).sqrt()).item()

        return PairScores(
            binoculars=binoculars,
            fast_detectgpt=fast,
            n_tokens=ids.shape[1],
            token_offsets=offsets[1:],
            log_ppl=nll2.cpu(),
            x_ppl=x_ent.cpu(),
        )
