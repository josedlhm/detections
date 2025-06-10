import cv2
import json
from pathlib import Path

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
