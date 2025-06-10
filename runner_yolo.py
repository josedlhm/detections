import yaml
from pathlib import Path
import cv2

from yolo_utils import load_yolo_model, detect_yolo_image
from image_utils import annotate_image, save_detections_json

try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    SAHI_AVAILABLE = True
except ImportError:
    SAHI_AVAILABLE = False

def run_yolo_detection(cfg: dict):
    input_dir = Path(cfg["input_dir"])
    output_dir = Path(cfg["output_dir"])
    annotated_subdir = cfg.get("annotated_subdir", "annotated")
    detections_file = cfg.get("detections_file", "detections.json")

    output_dir.mkdir(parents=True, exist_ok=True)
    img_out = output_dir / annotated_subdir
    img_out.mkdir(exist_ok=True)

    # Load model
    model = load_yolo_model(cfg["model_path"], cfg["device"])

    # Prepare tiled inference if requested
    if cfg.get("use_tiling", False):
        if not SAHI_AVAILABLE:
            raise RuntimeError("SAHI not installed; cannot use tiled inference.")
        sahi_model = AutoDetectionModel.from_pretrained(
            model_type=cfg["model_type"],
            model_path=cfg["model_path"],
            confidence_threshold=cfg["conf_thresh"],
            device=cfg["device"],
        )

    results = {}
    for img_path in sorted(input_dir.iterdir()):
        if not img_path.is_file():
            continue

        if cfg.get("use_tiling", False):
            sahi_pred = get_sliced_prediction(
                str(img_path),
                sahi_model,
                slice_height=cfg["slice_height"],
                slice_width=cfg["slice_width"],
                overlap_height_ratio=cfg["overlap_height_ratio"],
                overlap_width_ratio=cfg["overlap_width_ratio"],
            )
            dets = [
                {
                    "bbox": [
                        int(obj.bbox.minx), int(obj.bbox.miny),
                        int(obj.bbox.maxx), int(obj.bbox.maxy)
                    ],
                    "score": float(obj.score.value),
                    "class_id": int(obj.category.id)
                }
                for obj in sahi_pred.object_prediction_list
            ]
        else:
            img = cv2.imread(str(img_path))
            dets = detect_yolo_image(model, img, cfg["conf_thresh"])

        # Annotate and save
        annotated = annotate_image(cv2.imread(str(img_path)), dets)
        cv2.imwrite(str(img_out / img_path.name), annotated)
        results[img_path.name] = dets

    # Save JSON results
    save_detections_json(results, output_dir / detections_file)

def main():
    # Load configuration and run
    config = yaml.safe_load(open("config.yaml", "r"))
    yolo_cfg = config.get("yolo", {})
    run_yolo_detection(yolo_cfg)

if __name__ == "__main__":
    main()
