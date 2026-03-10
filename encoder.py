import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class SiglipTextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "google-bert/bert-base-uncased",
        embed_dim: int = 768,       # must match SiglipVisionConfig.hidden_size
        max_length: int = 64,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.max_length = max_length

        self.encoder = AutoModel.from_pretrained(model_name)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        bert_hidden_size = self.encoder.config.hidden_size  # 768 for bert-base
        self.projection = nn.Linear(bert_hidden_size, embed_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.659)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor = None) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        projected = self.projection(cls_embedding)
        projected = nn.functional.normalize(projected, dim=-1)
        return projected


class SiglipTokenizer:
    """Thin wrapper around HuggingFace tokenizer for convenience."""
    def __init__(
        self,
        model_name: str = "google-bert/bert-base-uncased",
        max_length: int = 64,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def __call__(self, texts: list[str], device: torch.device) -> dict:
        encoded = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v.to(device) for k, v in encoded.items()}


if __name__ == "__main__":
    device = 'cuda'

    tokenizer = SiglipTokenizer()
    text_encoder = SiglipTextEncoder(
        model_name="google-bert/bert-base-uncased",
        embed_dim=768,
        freeze_encoder=False,   
    )
    text_encoder = text_encoder.to(device)
    texts = ["a dog sitting on a bench", "a cat sleeping on a sofa"]
    tokens = tokenizer(texts, device)

    with torch.no_grad():
        text_embeds = text_encoder(**tokens)

    print(f"Input texts    : {texts}")
    print(f"Token shape    : {tokens['input_ids'].shape}")
    print(f"Embedding shape: {text_embeds.shape}")   # expect (2, 768)
    print(f"Logit scale    : {text_encoder.logit_scale.exp().item():.4f}")