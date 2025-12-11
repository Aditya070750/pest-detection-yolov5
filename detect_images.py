# Ultralytics YOLOv5 - Image/Folder Detection (no webcam)

import os
import sys
from pathlib import Path
import cv2
import torch
from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.torch_utils import select_device, smart_inference_mode
from utils.plots import Annotator, colors

# === Setup ===
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

@smart_inference_mode()
def run(weights=ROOT / "yolov5s.pt",      # you can change to your rat model later
        source=ROOT / "data" / "images",  # folder or single image path
        imgsz=(640, 640),
        conf_thres=0.25,
        iou_thres=0.45,
        device='',
        save_dir=ROOT / "runs" / "detect_images"
        ):
    """
    Run YOLOv5 on all images in a folder (or a single image).
    No webcam, only file-based detection.
    """

    # device & model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    # collect image files
    source = str(source)
    if os.path.isdir(source):
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        files = [os.path.join(source, f) for f in os.listdir(source)
                 if f.lower().endswith(exts)]
    else:
        files = [source]

    if not files:
        print(f"No images found in {source}")
        return

    os.makedirs(save_dir, exist_ok=True)

    for path in files:
        im0 = cv2.imread(path)
        if im0 is None:
            print(f"Could not read {path}")
            continue

        # preprocess
        im = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
        h, w = im.shape[:2]
        new_w = (w // int(stride)) * int(stride)
        new_h = (h // int(stride)) * int(stride)
        if (new_w != w) or (new_h != h):
            im = cv2.resize(im, (new_w, new_h))

        im = torch.from_numpy(im).permute(2, 0, 1).unsqueeze(0).to(device)
        im = im.float() / 255.0
        if getattr(model, "fp16", False):
            im = im.half()

        # inference
        pred = model(im)
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        annotator = Annotator(im0.copy(), line_width=2, example=str(names))

        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # print detected classes
                detected_labels = [names[int(c)] for c in det[:, -1].unique()]
                print(f"\nImage: {os.path.basename(path)}")
                print("Detected:", ", ".join(detected_labels))

                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(int(cls), True))

        result = annotator.result()
        save_path = os.path.join(save_dir, os.path.basename(path))
        cv2.imwrite(save_path, result)
        print(f"Saved: {save_path}")

    print("\nDone! All results saved in:", save_dir)


if __name__ == "__main__":
    # SIMPLE USAGE: edit these two paths if needed
    weights_path = ROOT / "yolov5s.pt"               # later: change to your rat best.pt
    source_path  = ROOT / "data" / "images"          # folder with images or one image file

    run(weights=weights_path, source=source_path)
