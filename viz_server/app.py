import os
import json
import glob
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data"
)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB max upload size

# Ensure the upload folder exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


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
                        "num_prototypes": metadata.get("num_prototypes", 0),
                        "num_slots": metadata.get("num_slots", 0),
                    }
                )
            except Exception as e:
                print(f"Error loading metadata for {run_dir}: {e}")

    # Sort runs by timestamp (newest first)
    runs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return jsonify(runs)


@app.route("/api/run/<run_id>/epochs")
def get_epochs(run_id):
    """Get a list of epochs for a specific run"""
    run_dir = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(run_id))
    if not os.path.exists(run_dir):
        return jsonify({"error": "Run not found"}), 404

    epochs_dir = os.path.join(run_dir, "epochs")
    if not os.path.exists(epochs_dir):
        return jsonify({"error": "No epochs found for this run"}), 404

    epoch_dirs = glob.glob(os.path.join(epochs_dir, "epoch_*"))
    epochs = []

    for epoch_dir in epoch_dirs:
        epoch_id = os.path.basename(epoch_dir)
        epoch_num = int(epoch_id.split("_")[1])

        # Find all batches in this epoch
        batch_dirs = glob.glob(os.path.join(epoch_dir, "batch_*"))
        batches = []

        for batch_dir in batch_dirs:
            batch_id = os.path.basename(batch_dir)
            batch_num = int(batch_id.split("_")[1])

            # Load batch metadata if available
            metadata_path = os.path.join(batch_dir, "metadata.json")
            metadata = {}
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                except Exception as e:
                    print(f"Error loading metadata for {batch_dir}: {e}")

            batches.append(
                {
                    "id": batch_id,
                    "number": batch_num,
                    "loss": metadata.get("loss", None),
                    "global_chamfer": metadata.get("global_chamfer", None),
                    "per_slot_chamfer": metadata.get("per_slot_chamfer", None),
                }
            )

        # Sort batches by number
        batches.sort(key=lambda x: x["number"])

        epochs.append({"id": epoch_id, "number": epoch_num, "batches": batches})

    # Sort epochs by number
    epochs.sort(key=lambda x: x["number"])
    return jsonify(epochs)


@app.route("/api/run/<run_id>/epoch/<epoch_id>/batch/<batch_id>")
def get_batch_data(run_id, epoch_id, batch_id):
    """Get detailed data for a specific batch"""
    batch_dir = os.path.join(
        app.config["UPLOAD_FOLDER"],
        secure_filename(run_id),
        "epochs",
        secure_filename(epoch_id),
        secure_filename(batch_id),
    )

    if not os.path.exists(batch_dir):
        return jsonify({"error": "Batch not found"}), 404

    # Load batch metadata
    metadata_path = os.path.join(batch_dir, "metadata.json")
    metadata = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"Error loading metadata for {batch_dir}: {e}")

    # Get point cloud data
    point_cloud_path = os.path.join(batch_dir, "point_cloud.json")
    point_cloud = None
    if os.path.exists(point_cloud_path):
        try:
            with open(point_cloud_path, "r") as f:
                point_cloud = json.load(f)
        except Exception as e:
            print(f"Error loading point cloud data for {batch_dir}: {e}")

    # Get slot data
    slots_dir = os.path.join(batch_dir, "slots")
    slots = []
    if os.path.exists(slots_dir):
        slot_files = glob.glob(os.path.join(slots_dir, "slot_*.json"))
        for slot_file in slot_files:
            slot_id = os.path.basename(slot_file).split(".")[0]
            try:
                with open(slot_file, "r") as f:
                    slot_data = json.load(f)
                    slots.append({"id": slot_id, "data": slot_data})
            except Exception as e:
                print(f"Error loading slot data for {slot_file}: {e}")

    # Get prototype data
    prototypes_dir = os.path.join(batch_dir, "prototypes")
    prototypes = None
    if os.path.exists(prototypes_dir):
        proto_file = os.path.join(prototypes_dir, "prototypes.json")
        if os.path.exists(proto_file):
            try:
                with open(proto_file, "r") as f:
                    prototypes = json.load(f)
            except Exception as e:
                print(f"Error loading prototype data for {proto_file}: {e}")

    # Compile complete data
    data = {
        "metadata": metadata,
        "point_cloud": point_cloud,
        "slots": slots,
        "prototypes": prototypes,
        "has_depth_image": os.path.exists(os.path.join(batch_dir, "depth_img.png")),
    }

    return jsonify(data)


@app.route("/api/run/<run_id>/epoch/<epoch_id>/batch/<batch_id>/depth_img.png")
def get_depth_image(run_id, epoch_id, batch_id):
    """Get the depth image for a specific batch"""
    img_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        secure_filename(run_id),
        "epochs",
        secure_filename(epoch_id),
        secure_filename(batch_id),
        "depth_img.png",
    )

    if not os.path.exists(img_path):
        return jsonify({"error": "Depth image not found"}), 404

    return send_from_directory(os.path.dirname(img_path), os.path.basename(img_path))


@app.route("/api/upload", methods=["POST"])
def upload_data():
    """Handle data uploads from the training script"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        run_id = data.get("run_id")
        if not run_id:
            return jsonify({"error": "No run_id provided"}), 400

        run_dir = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(run_id))
        os.makedirs(run_dir, exist_ok=True)

        # Save run metadata if provided
        run_metadata = data.get("run_metadata")
        if run_metadata:
            with open(os.path.join(run_dir, "run_metadata.json"), "w") as f:
                json.dump(run_metadata, f, indent=2)

        # Process epoch/batch data
        epoch_id = data.get("epoch_id")
        batch_id = data.get("batch_id")
        if epoch_id and batch_id:
            epochs_dir = os.path.join(run_dir, "epochs")
            os.makedirs(epochs_dir, exist_ok=True)

            epoch_dir = os.path.join(epochs_dir, secure_filename(epoch_id))
            os.makedirs(epoch_dir, exist_ok=True)

            batch_dir = os.path.join(epoch_dir, secure_filename(batch_id))
            os.makedirs(batch_dir, exist_ok=True)

            # Save batch metadata
            batch_metadata = data.get("batch_metadata")
            if batch_metadata:
                with open(os.path.join(batch_dir, "metadata.json"), "w") as f:
                    json.dump(batch_metadata, f, indent=2)

            # Save point cloud data
            point_cloud = data.get("point_cloud")
            if point_cloud:
                with open(os.path.join(batch_dir, "point_cloud.json"), "w") as f:
                    json.dump(point_cloud, f, indent=2)

            # Save slot data
            slots = data.get("slots")
            if slots:
                slots_dir = os.path.join(batch_dir, "slots")
                os.makedirs(slots_dir, exist_ok=True)

                for i, slot in enumerate(slots):
                    with open(os.path.join(slots_dir, f"slot_{i + 1}.json"), "w") as f:
                        json.dump(slot, f, indent=2)

            # Save prototype data
            prototypes = data.get("prototypes")
            if prototypes:
                prototypes_dir = os.path.join(batch_dir, "prototypes")
                os.makedirs(prototypes_dir, exist_ok=True)

                with open(os.path.join(prototypes_dir, "prototypes.json"), "w") as f:
                    json.dump(prototypes, f, indent=2)

            # Save depth image if provided
            depth_img = data.get("depth_img")
            if depth_img:
                import base64
                from PIL import Image
                import io

                try:
                    # Remove data URL prefix if present
                    if depth_img.startswith("data:image"):
                        depth_img = depth_img.split(",")[1]

                    # Decode base64 image
                    img_data = base64.b64decode(depth_img)
                    img = Image.open(io.BytesIO(img_data))
                    img.save(os.path.join(batch_dir, "depth_img.png"))
                except Exception as e:
                    print(f"Error saving depth image: {e}")

        return jsonify({"success": True})

    except Exception as e:
        print(f"Error processing upload: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
