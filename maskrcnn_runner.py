import argparse
from pathlib import Path
import cv2

from maskrcnn_utils import load_maskrcnn_model, detect_maskrcnn_image
from image_utils_maskrcnn import annotate_masks, save_masks_json

try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    SAHI_AVAILABLE = True
except ImportError:
    SAHI_AVAILABLE = False

def run_maskrcnn_detection(
    input_dir: Path,
    output_dir: Path,
    conf_thresh: float = 0.5,
    device: str = "cpu",
    use_tiling: bool = False,
    model_type: str = "maskrcnn_resnet50_fpn",
    slice_height: int = 640,
    slice_width: int = 640,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2
):
    """
    Run Mask R-CNN on all images in input_dir.
    If use_tiling=True, uses SAHI to slice images.
    Saves annotated images under output_dir/annotated/ and masks.json.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    img_out = output_dir / "annotated"
    img_out.mkdir(exist_ok=True)

    # Load base Mask R-CNN
    model = load_maskrcnn_model(device, model_name=model_type)

    # Prepare SAHI if tiling
    if use_tiling:
        if not SAHI_AVAILABLE:
            raise RuntimeError("SAHI not installed; cannot use tiled inference.")
        sahi_model = AutoDetectionModel.from_pretrained(
            model_type="mmdet",
            model_path=model_type,                 # SAHI expects a detectron2/mmdet name
            detection_threshold=conf_thresh,
            device=device,
            detection_category="instance_segmentation"
        )

    all_masks = {}

    for img_path in sorted(input_dir.iterdir()):
        if not img_path.is_file():
            continue

        img = cv2.imread(str(img_path))
        if use_tiling:
            sahi_pred = get_sliced_prediction(
                str(img_path),
                sahi_model,
                slice_height=slice_height,
                slice_width=slice_width,
                overlap_height_ratio=overlap_height_ratio,
                overlap_width_ratio=overlap_width_ratio,
            )
            masks = [obj.mask for obj in sahi_pred.object_prediction_list]
        else:
            masks = detect_maskrcnn_image(model, img, conf_thresh)

        # Annotate & save
        annotated = annotate_masks(img, masks)
        cv2.imwrite(str(img_out / img_path.name), annotated)
        all_masks[img_path.name] = [m["mask"] if isinstance(m, dict) else m for m in masks]

    # Save masks.json
    save_masks_json(all_masks, output_dir / "masks.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("MaskRCNN batch runner")
    parser.add_argument("--input-dir",  type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--conf-thresh", type=float, default=0.5)
    parser.add_argument("--device",      default="cpu")
    parser.add_argument("--use-tiling", action="store_true",
                        help="Enable SAHI tiled inference")
    parser.add_argument("--model-type", type=str,
                        default="maskrcnn_resnet50_fpn",
                        help="TorchVision model name")
    parser.add_argument("--slice-height", type=int, default=640)
    parser.add_argument("--slice-width",  type=int, default=640)
    parser.add_argument("--overlap-height-ratio", type=float, default=0.2)
    parser.add_argument("--overlap-width-ratio",  type=float, default=0.2)
    args = parser.parse_args()

    run_maskrcnn_detection(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        conf_thresh=args.conf_thresh,
        device=args.device,
        use_tiling=args.use_tiling,
        model_type=args.model_type,
        slice_height=args.slice_height,
        slice_width=args.slice_width,
        overlap_height_ratio=args.overlap_height_ratio,
        overlap_width_ratio=args.overlap_width_ratio
    )
