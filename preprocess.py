import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open("image.jpg")

def preprocess_image(image, image_size=224):
    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image_tensor = preprocess(image)
    image_tensor = image_tensor.unsqueeze(0) #add batch dimension
    return image_tensor

image_tensor = preprocess_image(img)

embed_dim = 768
patch_size = 16
image_size = 224
num_patches = (image_size // patch_size) ** 2

with torch.no_grad():
    patch_embedding = nn.Conv2d(in_channels=3, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
    patches = patch_embedding(image_tensor)

print(patches.shape)

position_embedding = nn.Embedding(num_patches, embed_dim)
position_ids = torch.arange(num_patches).expand((1,-1))

print(position_ids.shape)

embeddings = patches.flatten(start_dim = 2, end_dim = -1)
embeddings = embeddings.transpose(2,1)
embeddings = embeddings + position_embedding(position_ids)

print(embeddings.shape)

patches_viz = embeddings[0].detach().numpy()  # Shape: [196, 768]

plt.figure(figsize=(15, 8))
plt.imshow(patches_viz, aspect='auto', cmap='viridis')
plt.colorbar()
plt.title('Visualization of All Patch Embeddings')
plt.xlabel('Embedding Dimension')
plt.ylabel('Patch Number')
plt.savefig('embeddings')