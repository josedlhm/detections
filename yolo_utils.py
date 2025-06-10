import numpy as np
from ultralytics import YOLO

def load_yolo_model(model_path: str, device: str = "cpu") -> YOLO:
    model = YOLO(model_path)
    model.to(device)
    return model

def detect_yolo_image(model: YOLO, image: np.ndarray, conf_threshold: float = 0.5) -> list:
    results = model(image, conf=conf_threshold)[0]
    dets = []
    for *xyxy, conf, cls in results.boxes.data.tolist():
        x1, y1, x2, y2 = map(int, xyxy)
        dets.append({
            "bbox": [x1, y1, x2, y2],
            "score": float(conf),
            "class_id": int(cls)
        })
    return dets
