import torch
import torchvision
import numpy as np
from torchvision.transforms.functional import to_tensor
from pathlib import Path

def load_maskrcnn_model(device: str = "cpu", model_name: str = "maskrcnn_resnet50_fpn"):
    """
    Load a pretrained Mask R-CNN model.
    """
    model = torchvision.models.detection.__dict__[model_name](pretrained=True)
    model.to(device).eval()
    return model

def detect_maskrcnn_image(model, image: np.ndarray, conf_threshold: float = 0.5) -> list:
    """
    Run Mask R-CNN inference on a single image.
    Returns a list of dicts with keys:
      - 'mask': 2D list of 0/1
    """
    img_t = to_tensor(image).to(next(model.parameters()).device)
    outputs = model([img_t])[0]
    results = []
    for mask, score in zip(outputs["masks"], outputs["scores"]):
        if score < conf_threshold:
            continue
        # mask[0] is single-channel probability map → binarize
        m = (mask[0].cpu().mul(255).byte().numpy() > 0).astype(np.uint8)
        results.append({"mask": m.tolist()})
    return results
