import yaml
from pathlib import Path
import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    SAHI_AVAILABLE = True
except ImportError:
    SAHI_AVAILABLE = False

from image_utils import save_masks_json


def run_maskrcnn(cfg: dict):
    input_dir = Path(cfg["input_dir"])
    output_dir = Path(cfg["output_dir"])
    annotated_subdir = cfg.get("annotated_subdir", "annotated")
    masks_file = cfg.get("masks_file", "masks.json")

    output_dir.mkdir(parents=True, exist_ok=True)
    img_out = output_dir / annotated_subdir
    img_out.mkdir(exist_ok=True)

    # Detectron2 predictor setup
    detect_cfg = get_cfg()
    detect_cfg.merge_from_file(str(cfg["config_path"]))
    detect_cfg.MODEL.WEIGHTS = str(cfg["weights_path"])
    detect_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = cfg.get("conf_thresh", 0.5)
    detect_cfg.MODEL.DEVICE = cfg.get("device", "cpu")
    predictor = DefaultPredictor(detect_cfg)

    meta = None
    if predictor.cfg.DATASETS.TRAIN:
        meta = MetadataCatalog.get(predictor.cfg.DATASETS.TRAIN[0])

    # Prepare SAHI tiled model if needed
    use_tiling = cfg.get("use_tiling", False)
    if use_tiling:
        if not SAHI_AVAILABLE:
            raise RuntimeError("SAHI not installed; cannot use tiled inference.")
        sahi_model = AutoDetectionModel.from_pretrained(
            model_type="detectron2",
            model_path=str(cfg["weights_path"]),
            config_path=str(cfg["config_path"]),
            confidence_threshold=cfg.get("conf_thresh", 0.5),
            device=cfg.get("device", "cpu"),
        )

    all_masks = {}
    for img_path in sorted(input_dir.iterdir()):
        if not img_path.is_file():
            continue
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        if use_tiling:
            # SAHI sliced inference expects RGB
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            result = get_sliced_prediction(
                image=img_rgb,
                detection_model=sahi_model,
                slice_height=cfg.get("slice_height", 640),
                slice_width=cfg.get("slice_width", 640),
                overlap_height_ratio=cfg.get("overlap_height_ratio", 0.2),
                overlap_width_ratio=cfg.get("overlap_width_ratio", 0.2),
            )
            # Use SAHI's visualized output (RGB)
            annotated = cv2.cvtColor(result.image, cv2.COLOR_RGB2BGR)
            # Extract binary masks (0/255)
            masks = []
            for pred in result.object_prediction_list:
                mask = getattr(pred.mask, "bool_mask", pred.mask)
                masks.append((mask.astype(np.uint8) * 255))
        else:
            # Standard Detectron2 inference
            outputs = predictor(img_bgr)
            inst = outputs["instances"].to("cpu")
            if inst.has("pred_masks"):
                mask_arr = inst.pred_masks.numpy().astype(np.uint8) * 255
                masks = [m for m in mask_arr]
            else:
                masks = []
            # Render with Detectron2 Visualizer
            vis = Visualizer(img_bgr[:, :, ::-1], metadata=meta, scale=1.0)
            vis_output = vis.draw_instance_predictions(inst)
            annotated = vis_output.get_image()[:, :, ::-1]

        # Save annotated image and collect masks
        cv2.imwrite(str(img_out / img_path.name), annotated)
        all_masks[img_path.name] = [m.tolist() for m in masks]

    # Write out masks JSON
    save_masks_json(all_masks, output_dir / masks_file)


def main():
    cfg = yaml.safe_load(open("config.yaml", "r"))["maskrcnn"]
    run_maskrcnn(cfg)


if __name__ == "__main__":
    main()
