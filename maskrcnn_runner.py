import yaml
from pathlib import Path
import json
import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    SAHI_AVAILABLE = True
except ImportError:
    SAHI_AVAILABLE = False

from types import SimpleNamespace


def run_maskrcnn(cfg: dict):
    input_dir    = Path(cfg["input_dir"])
    output_dir   = Path(cfg["output_dir"])
    annotated_dir = output_dir / cfg.get("annotated_subdir", "annotated")
    json_fp      = output_dir / "detections.json"

    # make dirs
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(exist_ok=True)

    # Detectron2 predictor
    detect_cfg = get_cfg()
    detect_cfg.merge_from_file(str(cfg["config_path"]))
    detect_cfg.MODEL.WEIGHTS = str(cfg["weights_path"])
    detect_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = cfg.get("conf_thresh", 0.5)
    detect_cfg.MODEL.DEVICE = cfg.get("device", "cpu")
    predictor = DefaultPredictor(detect_cfg)

    # SAHI tiled model
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

    # helper: wrap Detectron2 outputs in a common simple‐namespace format
    def _collect_preds_from_detectron(inst):
        preds = []
        masks   = inst.pred_masks.cpu().numpy().astype(bool)
        boxes   = inst.pred_boxes.tensor.cpu().numpy()
        scores  = inst.scores.cpu().tolist()
        classes = inst.pred_classes.cpu().tolist()
        for mask, box, score, cls in zip(masks, boxes, scores, classes):
            p = SimpleNamespace(
                mask     = SimpleNamespace(bool_mask=mask),
                bbox     = SimpleNamespace(minx=box[0], miny=box[1], maxx=box[2], maxy=box[3]),
                score    = SimpleNamespace(value=score),
                category = SimpleNamespace(id=int(cls)),
            )
            preds.append(p)
        return preds

    all_detections: dict[str, list] = {}
    det_id_counter = 0

    for img_path in sorted(input_dir.iterdir()):
        if not img_path.is_file():
            continue
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        # 1) run inference
        if use_tiling:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            result = get_sliced_prediction(
                image=img_rgb,
                detection_model=sahi_model,
                slice_height=cfg.get("slice_height", 640),
                slice_width=cfg.get("slice_width", 640),
                overlap_height_ratio=cfg.get("overlap_height_ratio", 0.2),
                overlap_width_ratio=cfg.get("overlap_width_ratio", 0.2),
                postprocess_type="GREEDYNMM",
            )
            preds = result.object_prediction_list
        else:
            outputs = predictor(img_bgr)
            inst    = outputs["instances"].to("cpu")
            preds   = (_collect_preds_from_detectron(inst)
                       if inst.has("pred_masks") else [])

        # 2) manual contour viz + collect masks in JSON
        vis = img_bgr.copy()
        H, W = vis.shape[:2]
        dets_for_image: list[dict] = []

        for pred in preds:
            det_id_counter += 1

            # clamp box
            x1 = max(0, min(int(pred.bbox.minx), W - 1))
            y1 = max(0, min(int(pred.bbox.miny), H - 1))
            x2 = max(0, min(int(pred.bbox.maxx), W - 1))
            y2 = max(0, min(int(pred.bbox.maxy), H - 1))

            # full‐frame 0/255 mask
            full_mask = (pred.mask.bool_mask.astype(np.uint8) * 255)

            # draw contours
            contours, _ = cv2.findContours(full_mask,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cls_id = int(pred.category.id)
            color  = (0,255,0) if cls_id == 1 else (255,0,0)
            cv2.drawContours(vis, contours, -1, color, 1)

            # crop mask & append to JSON list
            cropped = full_mask[y1:y2+1, x1:x2+1].astype(np.uint8)
            dets_for_image.append({
                "bbox":     [x1, y1, x2, y2],
                "score":    float(pred.score.value),
                "class_id": cls_id,
                "mask":     cropped.tolist(),
            })

        # 3) save annotated image + record detections
        cv2.imwrite(str(annotated_dir / img_path.name), vis)
        all_detections[img_path.name] = dets_for_image

    # 4) write single JSON
    with open(json_fp, "w") as f:
        json.dump(all_detections, f, indent=2)

    # SUCCESS flag (optional)
    (output_dir / "SUCCESS").write_text("")


def main():
    cfg = yaml.safe_load(open("config.yaml", "r"))["maskrcnn"]
    run_maskrcnn(cfg)


if __name__ == "__main__":
    main()
