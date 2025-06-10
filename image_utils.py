import cv2
import json
from pathlib import Path
import numpy as np


def annotate_image(image, detections):
    out = image.copy()
    for d in detections:
        x1,y1,x2,y2 = d["bbox"]
        cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2)
        label = f"{d['class_id']}:{d['score']:.2f}"
        cv2.putText(out, label, (x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0),1)
    return out

def save_detections_json(results: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)

def annotate_masks(image: np.ndarray, masks: list) -> np.ndarray:
    """
    Overlay each binary mask (list of lists) onto the image with alpha blending.
    """
    overlay = image.copy()
    for m in masks:
        mask = np.array(m, dtype=np.uint8)
        color = (0, 255, 0)
        colored = np.zeros_like(image, dtype=np.uint8)
        colored[mask > 0] = color
        overlay = cv2.addWeighted(overlay, 1.0, colored, 0.4, 0)
    return overlay

def save_masks_json(results: dict, path: Path):
    """
    Save the per-image masks dict (image_name → [mask1, mask2, ...]) to JSON.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)