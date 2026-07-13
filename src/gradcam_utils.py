"""Faz 5: CLIP (ViT) ve karsilastirma modelleri (ResNet/ViT) icin ortak Grad-CAM yardimcilari.

CLIP'in goruntu encoder'i standart bir siniflandirici degil - "bu goruntu bu
metinle ne kadar benzer" diye bir skaler skor uretiyor. Grad-CAM'i buna
uygulamak icin: (1) modeli, sabit bir metin embedding'ine karsi kosinus
benzerligi donduren tek-skaler-cikisli bir wrapper'a saracak, (2) Vision
Transformer'in patch token'larini (CLS token haric) 2D bir izgaraya geri
sekillendiren bir `reshape_transform` tanimlamak gerekiyor - bu, ViT
mimarilerinde Grad-CAM icin `pytorch-grad-cam` kutuphanesinin onerdigi
standart teknik (Proje 1'de CNN encoder'i icin kullanilan Grad-CAM'in ViT'e
uyarlanmis hali).
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


class SimilarityTarget:
    """pytorch-grad-cam 'target': model ciktisini (zaten skaler benzerlik) oldugu gibi geri dondurur."""

    def __call__(self, model_output):
        return model_output


class ClipImageTextSimilarity(nn.Module):
    """pixel_values -> [batch] kosinus benzerligi (sabit bir metin embedding'ine karsi)."""

    def __init__(self, clip_model, text_embedding):
        super().__init__()
        self.clip_model = clip_model
        self.register_buffer("text_embedding", text_embedding)

    def forward(self, pixel_values):
        image_features = self.clip_model.get_image_features(pixel_values=pixel_values).pooler_output
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return (image_features * self.text_embedding).sum(dim=-1)


def vit_reshape_transform(tensor, grid_size):
    """(batch, 1+n_patches, dim) -> (batch, dim, grid, grid); CLS token atilir."""
    result = tensor[:, 1:, :].reshape(tensor.size(0), grid_size, grid_size, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def clip_gradcam(clip_model, processor, image, text, device, patch_size=32, image_size=224):
    """CLIP icin: (goruntu, metin sorgusu) -> Grad-CAM isi haritasi ve orijinal goruntu uzerine bindirilmis hali."""
    with torch.no_grad():
        text_inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
        text_embedding = clip_model.get_text_features(**text_inputs).pooler_output
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        text_embedding = text_embedding.squeeze(0)

    wrapper = ClipImageTextSimilarity(clip_model, text_embedding).to(device).eval()

    img_inputs = processor(images=[image], return_tensors="pt").to(device)
    pixel_values = img_inputs["pixel_values"]

    grid_size = image_size // patch_size
    target_layer = clip_model.vision_model.encoder.layers[-1].layer_norm1

    cam = GradCAM(
        model=wrapper,
        target_layers=[target_layer],
        reshape_transform=lambda t: vit_reshape_transform(t, grid_size),
    )
    grayscale_cam = cam(input_tensor=pixel_values, targets=[SimilarityTarget()])[0]

    resized = image.convert("RGB").resize((image_size, image_size))
    rgb_float = np.array(resized).astype(np.float32) / 255.0
    overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
    return grayscale_cam, overlay


def cnn_gradcam(model, target_layer, image_tensor, device, class_idx=None):
    """Standart CNN (ör. ResNet) icin Grad-CAM - Proje 1'deki ile ayni yaklasim."""
    from pytorch_grad_cam import GradCAM as CnnGradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    cam = CnnGradCAM(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(class_idx)] if class_idx is not None else None
    grayscale_cam = cam(input_tensor=image_tensor.unsqueeze(0).to(device), targets=targets)[0]
    return grayscale_cam


def vit_classifier_gradcam(model, target_layer, image_tensor, device, grid_size, class_idx=None):
    """timm ViT siniflandirici icin Grad-CAM (reshape_transform ile)."""
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    cam = GradCAM(
        model=model,
        target_layers=[target_layer],
        reshape_transform=lambda t: vit_reshape_transform(t, grid_size),
    )
    targets = [ClassifierOutputTarget(class_idx)] if class_idx is not None else None
    grayscale_cam = cam(input_tensor=image_tensor.unsqueeze(0).to(device), targets=targets)[0]
    return grayscale_cam


def overlay_heatmap(image, grayscale_cam, image_size=224):
    resized = image.convert("RGB").resize((image_size, image_size))
    rgb_float = np.array(resized).astype(np.float32) / 255.0
    return show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
