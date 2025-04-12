import os
import json
import glob
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data"
)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB max upload size

# Ensure the upload folder exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


def parse_timestamp(ts):
    """Convert various timestamp formats to Unix timestamp"""
    if isinstance(ts, (int, float)):
        return float(ts)
    try:
        # Try parsing ISO format
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except:
        return 0


@app.route("/")
def index():
    """Render the main visualization page"""
    return render_template("index.html")


@app.route("/api/runs")
def get_runs():
    """Get a list of all available runs"""
    run_dirs = glob.glob(os.path.join(app.config["UPLOAD_FOLDER"], "run_*"))
    runs = []

    for run_dir in run_dirs:
        metadata_path = os.path.join(run_dir, "run_metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)

                run_id = os.path.basename(run_dir)
                runs.append(
                    {
                        "id": run_id,
                        "timestamp": metadata.get("timestamp", ""),
                        "num_frames": metadata.get("num_frames", 0),
                        "num_objects": metadata.get("num_objects", 0),
                        "description": metadata.get("description", ""),
                    }
                )
            except Exception as e:
                print(f"Error loading metadata for {run_dir}: {e}")

    # Sort runs by timestamp (newest first)
    runs.sort(key=lambda x: parse_timestamp(x.get("timestamp", 0)), reverse=True)
    return jsonify(runs)


@app.route("/api/run/<run_id>/run_metadata.json")
def get_run_metadata(run_id):
    """Get metadata for a specific run"""
    metadata_path = os.path.join(
        app.config["UPLOAD_FOLDER"], secure_filename(run_id), "run_metadata.json"
    )
    if not os.path.exists(metadata_path):
        return jsonify({"error": "Run metadata not found"}), 404

    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        return jsonify(metadata)
    except Exception as e:
        print(f"Error loading run metadata: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/run/<run_id>/epochs")
def get_epochs(run_id):
    """Get a list of iterations for a specific run"""
    run_dir = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(run_id))
    if not os.path.exists(run_dir):
        return jsonify({"error": "Run not found"}), 404

    # Look for iteration directories
    iter_dirs = sorted(glob.glob(os.path.join(run_dir, "iter_*")))
    if not iter_dirs:
        return jsonify({"error": "No iterations found for this run"}), 404

    iterations = []
    for iter_dir in iter_dirs:
        iter_id = os.path.basename(iter_dir)
        iter_num = int(iter_id.split("_")[1])

        # Load iteration metadata
        metadata_path = os.path.join(iter_dir, "metadata.json")
        metadata = {}
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
            except Exception as e:
                print(f"Error loading metadata for {iter_dir}: {e}")

        # Get frame data
        frames = []
        frame_files = sorted(glob.glob(os.path.join(iter_dir, "frame_*.json")))
        for frame_file in frame_files:
            frame_id = os.path.basename(frame_file).split(".")[0]
            frame_num = int(frame_id.split("_")[1])
            frames.append(
                {
                    "id": frame_id,
                    "number": frame_num,
                }
            )

        iterations.append(
            {
                "id": iter_id,
                "number": iter_num,
                "frames": frames,
                "metadata": metadata,
            }
        )

    return jsonify(iterations)


@app.route("/api/run/<run_id>/iter/<iter_id>/frame_<frame_id>.json")
def get_frame_data(run_id, iter_id, frame_id):
    """Get detailed data for a specific frame"""
    frame_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        secure_filename(run_id),
        secure_filename(iter_id),
        f"frame_{frame_id}.json",
    )

    if not os.path.exists(frame_path):
        return jsonify({"error": f"Frame not found: {frame_path}"}), 404

    try:
        with open(frame_path, "r") as f:
            frame_data = json.load(f)
        return jsonify(frame_data)
    except Exception as e:
        print(f"Error loading frame data: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
