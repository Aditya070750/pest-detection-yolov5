import sys
import os
import torch
import cv2
import time
import json
import datetime
from pathlib import Path

# === Allow imports from YOLOv5 folder ===
YOLO_PATH = os.path.join(os.getcwd(), 'yolov5')
if YOLO_PATH not in sys.path:
    sys.path.append(YOLO_PATH)

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes, check_img_size
from utils.torch_utils import select_device
from utils.plots import Annotator, colors

# === YOLOv5 Initialization ===
ROOT = Path(__file__).resolve().parents[0] / "yolov5"
weights = ROOT / "yolov5s.pt"  # ✅ use YOLOv5s for better accuracy
device = select_device("cpu")  # change to "cuda" if GPU available

print("🔍 Loading YOLOv5 model... Please wait.")
model = DetectMultiBackend(weights, device=device)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size((640, 640), s=stride)
print(f"✅ Model loaded successfully: {weights.name}")

# === Global variable to store latest detection ===
latest_detection = "None"

# === Detection log file ===
DATA_FILE = "detected_data.json"

# -----------------------------
# Save Detection to JSON
# -----------------------------
def save_detection_to_file(object_name):
    """Save each detected object with timestamp in detected_data.json."""
    if not object_name or object_name == "None":
        return

    new_data = {
        "object": object_name,
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Load previous detections
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    # Avoid duplicates
    if len(data) == 0 or data[-1]["object"] != object_name:
        data.append(new_data)
        with open(DATA_FILE, "w") as f:
            json.dump(data, f, indent=2)
        print(f"📝 Saved detection: {object_name}")

# -----------------------------
# Get Latest Detection
# -----------------------------
def get_latest_detection():
    """Return the latest detected object name."""
    global latest_detection
    return latest_detection


# -----------------------------
# Generate YOLOv5 Frames
# -----------------------------
def generate_frames():
    """Generator that yields YOLOv5 processed frames for Flask."""
    global latest_detection

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("⚠️ Error: Unable to access camera.")
        return

    print("🎥 Camera feed started successfully.")

    while True:
        success, im0 = cap.read()
        if not success:
            print("⚠️ Failed to grab frame, exiting...")
            break

        start_time = time.time()

        # Preprocess image
        im0_resized = cv2.resize(im0, (640, 480))
        im = cv2.cvtColor(im0_resized, cv2.COLOR_BGR2RGB)
        im = torch.from_numpy(im).permute(2, 0, 1).unsqueeze(0).to(device)
        im = im.float() / 255.0

        # YOLO Inference
        with torch.no_grad():
            pred = model(im)
            pred = non_max_suppression(
                pred,
                conf_thres=0.3,    # 🔧 Lower confidence threshold for more detections
                iou_thres=0.45,    # Intersection threshold
                classes=None,      # Detect all classes
                agnostic=False
            )

        # Process detections
        for det in pred:
            annotator = Annotator(im0_resized, line_width=2, example=str(names))
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0_resized.shape).round()
                detected_objects = [names[int(c)] for c in det[:, -1].unique()]

                # Combine detections and save
                latest_detection = ", ".join(detected_objects)
                save_detection_to_file(latest_detection)

                for *xyxy, conf, cls in reversed(det):
                    label = f"{names[int(cls)]} {conf:.2f}"
                    annotator.box_label(xyxy, label, color=colors(int(cls), True))

            im0_resized = annotator.result()

        # FPS display
        fps = 1 / (time.time() - start_time)
        cv2.putText(im0_resized, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode frame for Flask
        ret, buffer = cv2.imencode(".jpg", im0_resized)
        frame = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    cap.release()
    print("🛑 Camera released successfully.")
