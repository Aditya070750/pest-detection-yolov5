# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""YOLOv5 Live Webcam Detection (Optimized for FPS + Accuracy)."""

import argparse
import os
import sys
import time
from pathlib import Path
from threading import Thread

import cv2
import torch

# === Setup ===
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from models.common import DetectMultiBackend
from utils.general import check_img_size, check_requirements, non_max_suppression, scale_boxes
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, smart_inference_mode


# ===============================================================
# Threaded Webcam Stream (for higher FPS)
# ===============================================================
class WebcamStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True


# ===============================================================
# Main YOLOv5 Inference Function
# ===============================================================
@smart_inference_mode()
def run(
    weights=ROOT / "yolov5n.pt",  # lightweight nano model
    source=0,
    imgsz=(320, 320),
    conf_thres=0.5,  # increased confidence threshold for better accuracy
    iou_thres=0.4,  # slightly reduced IoU to reduce duplicate boxes
    device="",
    view_img=True,
    classes=None,
    agnostic_nms=False,
    half=False,
    dnn=False,
):
    """Run YOLOv5 live webcam detection optimized for FPS and accuracy."""
    # Initialize
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, _pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    # Open webcam with threaded stream
    print("Opening webcam...")
    cap = WebcamStream(int(source)).start()
    time.sleep(1.0)

    print("✅ Webcam started. Press 'Q' to exit.")

    # Detection loop
    while True:
        im0 = cap.read()
        if im0 is None:
            print("⚠️ Frame not received, exiting...")
            break

        start_time = time.time()

        # Preprocess
        im = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)

        h, w = im.shape[:2]
        stride_val = int(model.stride) if hasattr(model, "stride") else 32
        new_w = (w // stride_val) * stride_val
        new_h = (h // stride_val) * stride_val
        if (new_w != w) or (new_h != h):
            im = cv2.resize(im, (new_w, new_h))

        im = torch.from_numpy(im).permute(2, 0, 1).unsqueeze(0).to(device)
        im = im.float() / 255.0
        if model.fp16:
            im = im.half()

        # Inference
        pred = model(im)
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms)

        # Process predictions
        for det in pred:
            annotator = Annotator(im0, line_width=2, example=str(names))
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # ✅ Print detected object names once per frame
                detected_labels = [names[int(c)] for c in det[:, -1].unique()]
                print("Detected:", ", ".join(detected_labels))

                # Draw boxes and labels on the frame
                for *xyxy, conf, cls in reversed(det):
                    label = f"{names[int(cls)]} {conf:.2f}"
                    annotator.box_label(xyxy, label, color=colors(int(cls), True))

            im0 = annotator.result()

        # Calculate FPS
        fps = 1 / (time.time() - start_time)
        cv2.putText(im0, f"FPS: {int(fps)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        # Display
        if view_img:
            cv2.imshow("YOLOv5 Live Detection (Optimized)", im0)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.stop()
    cv2.destroyAllWindows()


# ===============================================================
# CLI Arguments
# ===============================================================
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5n.pt", help="model path")
    parser.add_argument("--source", type=str, default="0", help="source (0 for webcam)")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[320], help="image size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.4, help="NMS IoU threshold")
    parser.add_argument("--device", default="", help="cuda device or cpu")
    parser.add_argument("--view-img", action="store_true", help="show detection results")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--half", action="store_true", help="use FP16")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    return opt


# ===============================================================
# Main
# ===============================================================
def main(opt):
    check_requirements(exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
