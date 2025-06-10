import argparse
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


def run_yolo_detection(
    input_dir: Path,
    output_dir: Path,
    model_path: str,
    conf_thresh: float = 0.5,
    device: str = "cpu",
    use_tiling: bool = False,
    model_type: str = "yolov8",
    slice_height: int = 640,
    slice_width: int = 640,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2
):
    """
    Run YOLO inference on all images in input_dir.
    If use_tiling=True, uses SAHI to slice images.
    Saves annotated images under output_dir/annotated/ and a detections.json.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    img_out = output_dir / "annotated"
    img_out.mkdir(exist_ok=True)

    # Load YOLO model
    yolo_model = load_yolo_model(model_path, device)

    # Prepare SAHI model if tiling
    if use_tiling:
        if not SAHI_AVAILABLE:
            raise RuntimeError("SAHI not installed; cannot use tiled inference.")
        sahi_model = AutoDetectionModel.from_pretrained(
            model_type=model_type,
            model_path=model_path,
            confidence_threshold=conf_thresh,
            device=device,
        )

    results = {}
    for img_path in sorted(input_dir.iterdir()):
        if not img_path.is_file():
            continue

        if use_tiling:
            sahi_pred = get_sliced_prediction(
                str(img_path),
                sahi_model,
                slice_height=slice_height,
                slice_width=slice_width,
                overlap_height_ratio=overlap_height_ratio,
                overlap_width_ratio=overlap_width_ratio,
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
            dets = detect_yolo_image(yolo_model, img, conf_thresh)

        # Annotate & save
        img = cv2.imread(str(img_path))
        annotated = annotate_image(img, dets)
        cv2.imwrite(str(img_out / img_path.name), annotated)
        results[img_path.name] = dets

    # Save JSON
    save_detections_json(results, output_dir / "detections.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("YOLO batch runner")
    parser.add_argument("--input-dir",  type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model",      required=True)
    parser.add_argument("--conf-thresh", type=float, default=0.5)
    parser.add_argument("--device",      default="cpu")
    parser.add_argument("--use-tiling", "--use-tiling", action="store_true")    
    parser.add_argument("--model-type", type=str, default="yolov11",
                        help="Model type for SAHI (e.g. yolov8, yolov5)")
    parser.add_argument("--slice-height", type=int, default=640)
    parser.add_argument("--slice-width",  type=int, default=640)
    parser.add_argument("--overlap-height-ratio", type=float, default=0.2)
    parser.add_argument("--overlap-width-ratio",  type=float, default=0.2)
    args = parser.parse_args()

    run_yolo_detection(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_path=args.model,
        conf_thresh=args.conf_thresh,
        device=args.device,
        use_tiling=args.use_tiling,
        model_type=args.model_type,
        slice_height=args.slice_height,
        slice_width=args.slice_width,
        overlap_height_ratio=args.overlap_height_ratio,
        overlap_width_ratio=args.overlap_width_ratio
    )
