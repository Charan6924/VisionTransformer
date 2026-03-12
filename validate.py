import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO
from dataclasses import dataclass, field

from transformer import SiglipVisionModel, SiglipVisionConfig
from encoder import SiglipTextEncoder, SiglipTokenizer
from main import SiglipModel, TrainConfig


@dataclass
class ValidateConfig:
    checkpoint_path: str = "/scratch/pioneer/users/cxv166/VisionTransformer/checkpoints/step_0078000.pt"
    test_cases: list = field(default_factory=lambda: [
        {
            "image_path": "/scratch/pioneer/users/cxv166/VisionTransformer/test_images/cute_dog.jpg",
            "captions": [
                "a dog sitting on the grass",       # correct
                "a cat sleeping on a sofa",         # wrong animal
                "a car parked on the street",       # unrelated
                "a person riding a bicycle",        # unrelated
            ],
        },
        {
            "image_path": "/scratch/pioneer/users/cxv166/VisionTransformer/test_images/ant.jpg",
            "captions": [
                "a close up of an ant on a surface",  # correct
                "a butterfly on a flower",            # wrong insect
                "a dog running in a park",            # unrelated
                "a pizza on a table",                 # unrelated
            ],
        },
    ])


def load_model(cfg: ValidateConfig, device: torch.device):
    train_cfg = TrainConfig()
    model = SiglipModel(train_cfg).to(device)
    ckpt = torch.load(cfg.checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint from step {ckpt['step']}, epoch {ckpt['epoch']}")
    return model


def load_image_from_url(url: str):
    response = requests.get(url, timeout=10)
    return Image.open(BytesIO(response.content)).convert("RGB")


def load_image_from_path(path: str):
    return Image.open(path).convert("RGB")


def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return transform(image).unsqueeze(0)  


def run_sanity_check(cfg: ValidateConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, device)
    tokenizer = SiglipTokenizer()

    for i, case in enumerate(cfg.test_cases):
        print(f"\nTest {i+1}:")

        try:
            image = load_image_from_path(case["image_path"])
        except Exception as e:
            print(f"  Could not load image: {e}")
            continue

        pixel_values = preprocess_image(image).to(device)

        with torch.no_grad():
            image_embed = model.encode_image(pixel_values)  # (1, D)

            tokens = tokenizer(case["captions"], device)
            text_embeds = model.encode_text(
                tokens["input_ids"],
                tokens["attention_mask"],
                tokens.get("token_type_ids"),
            )  # (N, D)

            scores = (image_embed @ text_embeds.T).squeeze(0)  # (N,)

        ranked = sorted(zip(scores.tolist(), case["captions"]), reverse=True)

        print(f"Image: {case['image_path'].split('/')[-1]}")
        print(f"{'Score':>8}  Caption")
        print(f"{'-'*50}")
        for score, caption in ranked:
            marker = " <-- correct" if caption == case["captions"][0] else ""
            print(f"  {score:>8.4f}  {caption}{marker}")

        correct = case["captions"][0]
        top_caption = ranked[0][1]
        passed = top_caption == correct
        rank = [c for _, c in ranked].index(correct) + 1
        print(f"\n  {'PASS' if passed else 'FAIL'} — correct caption ranked #{rank}")


if __name__ == "__main__":
    cfg = ValidateConfig(
        checkpoint_path="/scratch/pioneer/users/cxv166/VisionTransformer/checkpoints/step_0078000.pt", 
    )
    run_sanity_check(cfg)