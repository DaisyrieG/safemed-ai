"""Stage 2: cross-encoder classifier (GD / MD / HD).

The language encoder is plain MiniLM (Wang et al., 2020), a pretrained model that has never been trained to
judge (query, document) pairs. Everything that turns it into a cross-encoder is defined here: the query and the
document go in together as one input, [CLS] query [SEP] document [SEP] (Nogueira & Cho, 2019), and the final
[CLS] vector goes through dropout and one linear layer to the three labels (Devlin et al., 2019).
"""

import json
import os
from types import SimpleNamespace
from typing import Optional, Sequence

import torch
from torch import nn
from transformers import AutoConfig, AutoModel

BASE_MODEL = "microsoft/MiniLM-L12-H384-uncased"
LABELS = ["ground_truth", "mediocre", "harmful"]
WEIGHTS_FILE = "cross_encoder.pt"
HEAD_CONFIG_FILE = "cross_encoder_config.json"


class SafeMedCrossEncoder(nn.Module):
    def __init__(self, encoder: nn.Module, labels: Sequence[str] = LABELS, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.labels = list(labels)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(encoder.config.hidden_size, len(self.labels))
        # Same shape as a Hugging Face config, so the filter can read the label order from it.
        self.config = SimpleNamespace(id2label={i: name for i, name in enumerate(self.labels)})

    @classmethod
    def from_base(cls, base_model: str = BASE_MODEL, labels: Sequence[str] = LABELS) -> "SafeMedCrossEncoder":
        """A new cross-encoder: pretrained MiniLM weights, untrained classifier."""
        return cls(AutoModel.from_pretrained(base_model, add_pooling_layer=False), labels)

    def forward(self, input_ids, attention_mask, token_type_ids: Optional[torch.Tensor] = None):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                              token_type_ids=token_type_ids).last_hidden_state
        cls_vector = hidden[:, 0]  # [CLS] has attended to both the query and the document
        return SimpleNamespace(logits=self.classifier(self.dropout(cls_vector)))

    def save_pretrained(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        self.encoder.config.save_pretrained(out_dir)  # config.json: the encoder's size, to rebuild it on load
        torch.save(self.state_dict(), os.path.join(out_dir, WEIGHTS_FILE))
        with open(os.path.join(out_dir, HEAD_CONFIG_FILE), "w", encoding="utf-8") as f:
            json.dump({"labels": self.labels, "dropout": self.dropout.p}, f, indent=1)

    @classmethod
    def from_pretrained(cls, model_dir: str) -> "SafeMedCrossEncoder":
        """Loads a cross-encoder saved by save_pretrained (no download needed)."""
        with open(os.path.join(model_dir, HEAD_CONFIG_FILE), "r", encoding="utf-8") as f:
            head = json.load(f)
        encoder = AutoModel.from_config(AutoConfig.from_pretrained(model_dir), add_pooling_layer=False)
        model = cls(encoder, head["labels"], head["dropout"])
        state = torch.load(os.path.join(model_dir, WEIGHTS_FILE), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        return model


def is_own_cross_encoder(model_dir: str) -> bool:
    return os.path.isfile(os.path.join(model_dir, HEAD_CONFIG_FILE))