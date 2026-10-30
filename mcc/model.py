import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


class MultiHeadClassifier(nn.Module):
    """다국어 인코더 하나에 HMC/ATC 분류 헤드를 붙인 모델."""

    def __init__(self, encoder_path, label_maps, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_path)
        hidden = self.encoder.config.hidden_size
        self.label_maps = label_maps  # {"hmc": [code, ...], "atc": [...]}
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({k: nn.Linear(hidden, len(v)) for k, v in label_maps.items()})

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(out.dtype)
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        pooled = self.dropout(pooled)
        return {k: head(pooled) for k, head in self.heads.items()}

    def save(self, out_dir, tokenizer, extra=None):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(out_dir / "encoder")
        tokenizer.save_pretrained(out_dir / "encoder")
        torch.save(self.heads.state_dict(), out_dir / "heads.pt")
        meta = {"label_maps": self.label_maps, **(extra or {})}
        (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, model_dir, device="cpu"):
        model_dir = Path(model_dir)
        meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
        model = cls(model_dir / "encoder", meta["label_maps"])
        model.heads.load_state_dict(torch.load(model_dir / "heads.pt", map_location="cpu", weights_only=True))
        tokenizer = AutoTokenizer.from_pretrained(model_dir / "encoder")
        return model.to(device).eval(), tokenizer, meta
