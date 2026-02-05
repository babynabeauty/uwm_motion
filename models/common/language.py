import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPTokenizer

class CLIPTextEncoder(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.language_model = CLIPModel.from_pretrained('/data/shared_workspace/LLM_weights/openai/clip-vit-base-patch32')
        # Freeze pretrained model parameters
        for p in self.language_model.parameters():
            p.requires_grad = False
        self.head = nn.Linear(512, embed_dim)
        self.tokenizer = CLIPTokenizer.from_pretrained('/data/shared_workspace/LLM_weights/openai/clip-vit-base-patch32')

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        feats = self.language_model.get_text_features(
            input_ids=input_ids, attention_mask=attention_mask
        )
        feats = self.head(feats)
        return feats

    # def encode_texts(self, texts: list[str], device=None):
    #     # convenience helper: accepts list[str] or single str wrapped as list
    #     if isinstance(texts, str):
    #         texts = [texts]
    #     inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    #     if device is None:
    #         device = next(self.language_model.parameters()).device
    #     input_ids = inputs["input_ids"].to(device)
    #     attention_mask = inputs["attention_mask"].to(device)
    #     with torch.no_grad():
    #         return self.forward(input_ids, attention_mask)  # (batch, embed_dim)

    def encode(self, texts: list[str], device=None):
        # convenience helper: accepts list[str] or single str wrapped as list
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        if device is None:
            device = next(self.language_model.parameters()).device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        return input_ids, attention_mask