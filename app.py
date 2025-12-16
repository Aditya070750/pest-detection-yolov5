from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
import datetime
import json
import os
from yolo_flask import generate_frames, get_latest_detection  # YOLOv5 interface

# =============================
# Flask Initialization
# =============================
app = Flask(__name__)
DATA_FILE = 'detected_data.json'
current_theme = "night"  # Default theme

# =============================
# Utility: Save Detections to JSON
# =============================
def save_detection(object_name):
    """Save latest detected object with timestamp."""
    if not object_name or object_name == "None":
        return  # Skip empty detections

    new_data = {
        "object": object_name,
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Load existing data
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            data = json.load(f)
    else:
        data = []

    # Append and write back
    data.append(new_data)
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

# =============================
# ROUTES
# =============================

@app.route('/')
def index():
    """Dashboard Home Page."""
    return render_template('index.html', theme=current_theme)


@app.route('/live')
def live():
    """Live YOLOv5 detection page."""
    latest = get_latest_detection()
    save_detection(latest)
    data = {
        "object": latest if latest else "None",
        "count": "Streaming",
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    return render_template('live.html', theme=current_theme, data=data)


# =============================
# LOG PAGE ROUTES (Auto-Refresh + Clear)
# =============================

@app.route('/log')
def log():
    """Render the Detection Log Page."""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            data = json.load(f)
    else:
        data = []
    return render_template('log.html', data=data, theme=current_theme)


@app.route('/get_log_data', methods=['GET'])
def get_log_data():
    """Return detection logs as JSON for frontend auto-refresh."""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            data = json.load(f)
    else:
        data = []
    return jsonify(data)


@app.route('/clear_log', methods=['POST'])
def clear_log():
    """Clear all detection data from JSON log."""
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)
        print("🧹 Detection log cleared by user.")
    return jsonify({"status": "Log cleared"}), 200


@app.route('/video_feed')
def video_feed():
    """Serve live YOLOv5 detection video feed."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# =============================
# ESP32 COMMUNICATION
# =============================

@app.route('/esp_data', methods=['GET'])
def send_data_to_esp():
    """
    ESP32 polls this route via HTTP GET to receive
    the latest YOLOv5 detection result.
    """
    latest = get_latest_detection()
    print(f"📤 Sent to ESP32: {latest}")
    return jsonify({"object": latest, "status": "ok"}), 200


@app.route('/esp_update', methods=['POST'])
def receive_data_from_esp():
    """
    ESP32 can POST data here (like sensor readings or status updates).
    """
    data = request.get_json()
    print(f"📥 Data received from ESP32: {data}")

    if data and "object" in data:
        save_detection(data["object"])

    return jsonify({"status": "Data received successfully"}), 200


# =============================
# THEME TOGGLER
# =============================

@app.route('/toggle_theme', methods=['POST'])
def toggle_theme():
    """Toggle between night/day theme."""
    global current_theme
    current_theme = 'day' if current_theme == 'night' else 'night'
    return redirect(request.referrer or url_for('index'))


# =============================
# MAIN ENTRY POINT
# =============================
if __name__ == '__main__':
    print("🚀 Flask YOLOv5 + ESP32 Server Running on Port 5000...")
    app.run(debug=True, host="0.0.0.0", port=5000)
