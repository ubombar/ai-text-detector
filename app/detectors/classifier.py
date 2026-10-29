"""desklib/ai-text-detector-v1.01: DeBERTa-v3-large + mean pooling + linear head.

The model card ships a custom class rather than a transformers head, so it is
reproduced here.
"""
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel

from .. import config
from .device import pick_device

CHUNK = 512
STRIDE = 384


class DesklibAIDetectionModel(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, cfg):
        super().__init__(cfg)
        self.model = AutoModel.from_config(cfg)
        self.classifier = nn.Linear(cfg.hidden_size, 1)
        self.post_init()

    def forward(self, input_ids, attention_mask):
        hidden = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1e-9)
        return self.classifier(pooled)


class Classifier:
    def __init__(self):
        self.device = pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(config.CLASSIFIER_MODEL)
        # DeBERTa-v3 is unstable in fp16, keep it in fp32 (~1.7 GB).
        self.model = DesklibAIDetectionModel.from_pretrained(config.CLASSIFIER_MODEL).to(self.device).eval()

    @torch.inference_mode()
    def score(self, text: str) -> float:
        """Length-weighted mean P(AI) over overlapping 512-token chunks."""
        ids = self.tokenizer(text, add_special_tokens=False).input_ids
        starts = list(range(0, max(len(ids) - CHUNK, 0) + 1, STRIDE))
        if starts[-1] + CHUNK < len(ids):
            starts.append(len(ids) - CHUNK)
        total, weight = 0.0, 0
        for s in starts:
            chunk = ids[s : s + CHUNK - 2]
            enc = self.tokenizer.prepare_for_model(chunk, return_tensors="pt")
            logit = self.model(
                enc["input_ids"][None].to(self.device), enc["attention_mask"][None].to(self.device)
            )
            total += torch.sigmoid(logit).item() * len(chunk)
            weight += len(chunk)
        return total / max(weight, 1)
