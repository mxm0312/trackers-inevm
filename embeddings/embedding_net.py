import torch
from torch import nn
from torchvision import models, transforms
import torchvision
import numpy as np
import cv2


class EmbeddingNet(nn.Module):
    def __init__(self, emb_size=128):
        super(EmbeddingNet, self).__init__()
        self.model = models.mobilenet_v2()
        self.model.to('cuda')
        self.resize = torchvision.transforms.Resize((1, 128))
        self.model.eval()

    def forward(self, x):
        x = self.model(x)
        x = self.resize(x.unsqueeze(0)).squeeze(0)
        return x.cpu()


def preprocess_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (224, 224))
    image_tensor = (
        torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
    )  # Преобразуем numpy массив в тензор PyTorch
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    image_tensor = normalize(image_tensor)
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor.cuda()


def get_embedding(image: np.ndarray, model):
    prepr_img = preprocess_image(image)
    embedding = model(prepr_img)
    return embedding.squeeze(0).detach().numpy()
