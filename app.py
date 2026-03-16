import os
import gc
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_INFERENCE_ENABLED = os.environ.get("IMAGE_INFERENCE_ENABLED", "true").lower() in ("1", "true", "yes", "on")
IMAGE_MODEL_PERSISTENT = os.environ.get("IMAGE_MODEL_PERSISTENT", "false").lower() in ("1", "true", "yes", "on")
IMAGE_MODEL_PRELOAD = os.environ.get("IMAGE_MODEL_PRELOAD", "true").lower() in ("1", "true", "yes", "on")

# Lazy-load all models to keep startup memory low
dt_model = None
knn_model = None
knn_scaler = None

_kmeans_model = None
_resnet_model = None


def get_tabular_models():
    global dt_model, knn_model, knn_scaler
    if dt_model is None:
        dt_model = joblib.load(
            os.path.join(BASE_DIR, "decisiontree/decision_tree_models/decision_tree_cat_dog.pkl")
        )
    if knn_model is None:
        knn_model = joblib.load(
            os.path.join(BASE_DIR, "knn/knn_models/knn_cat_dog_model.pkl")
        )
    if knn_scaler is None:
        knn_scaler = joblib.load(
            os.path.join(BASE_DIR, "knn/knn_models/scaler.pkl")
        )
    return dt_model, knn_model, knn_scaler


def get_kmeans_model():
    global _kmeans_model
    if _kmeans_model is None:
        _kmeans_model = joblib.load(
            os.path.join(BASE_DIR, "k-mean(clustering)/kmeans_models/cat_dog_kmeans.pkl")
        )
    return _kmeans_model


def get_resnet_model():
    global _kmeans_model, _resnet_model
    if _resnet_model is None:
        import tensorflow as tf
        _resnet_model = tf.keras.applications.ResNet50(
            weights="imagenet",
            include_top=False,
            pooling="avg"
        )
    return _resnet_model


def preload_image_models_once():
    """Optionally warm up image models at process start."""
    if not IMAGE_INFERENCE_ENABLED:
        return

    try:
        get_kmeans_model()
        if IMAGE_MODEL_PERSISTENT and IMAGE_MODEL_PRELOAD:
            get_resnet_model()
    except Exception as e:
        app.logger.warning("Image model preload skipped: %s", e)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/config", methods=["GET"])
def config():
    return jsonify(
        {
            "image_inference_enabled": IMAGE_INFERENCE_ENABLED,
            "image_model_persistent": IMAGE_MODEL_PERSISTENT,
        }
    )


@app.route("/predict/features", methods=["POST"])
def predict_features():
    """Predict cat or dog from physical feature measurements (DT and KNN)."""
    data = request.get_json(force=True)
    algorithm = data.get("algorithm", "").strip()

    if algorithm not in ("decision_tree", "knn"):
        return jsonify({"error": "Algorithm must be 'decision_tree' or 'knn'"}), 400

    try:
        features = np.array([[
            float(data["H"]),
            float(data["W"]),
            float(data["L"]),
            int(data["FL"]),
            int(data["PS"]),
            int(data["ES"])
        ]])
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    dt, knn, scaler = get_tabular_models()

    if algorithm == "decision_tree":
        pred = dt.predict(features)[0]
        proba = dt.predict_proba(features)[0]
    else:  # knn
        features_scaled = scaler.transform(features)
        pred = knn.predict(features_scaled)[0]
        proba = knn.predict_proba(features_scaled)[0]

    result = "Dog" if int(pred) == 1 else "Cat"
    confidence = round(float(max(proba)) * 100, 1)

    return jsonify({"result": result, "confidence": confidence})


@app.route("/predict/image", methods=["POST"])
def predict_image():
    """Predict cat or dog from an uploaded image using K-Means + ResNet50."""
    if not IMAGE_INFERENCE_ENABLED:
        return jsonify({
            "error": "Image inference is disabled on this deployment to prevent memory issues."
        }), 503

    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]

    try:
        img = Image.open(file.stream).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32)
    except Exception as e:
        return jsonify({"error": f"Could not process image: {e}"}), 400

    try:
        import tensorflow as tf
        kmeans = get_kmeans_model()
        if IMAGE_MODEL_PERSISTENT:
            resnet = get_resnet_model()
        else:
            resnet = tf.keras.applications.ResNet50(
                weights="imagenet",
                include_top=False,
                pooling="avg"
            )

        img_batch = np.expand_dims(img_array, axis=0)
        img_batch = tf.keras.applications.resnet50.preprocess_input(img_batch)
        vector = resnet.predict(img_batch, verbose=0)

        cluster = int(kmeans.predict(vector.astype(np.float32))[0])
        # Cluster mapping determined from camera_predict.py: 0 = Dog, 1 = Cat
        result = "Dog" if cluster == 0 else "Cat"

        response = jsonify({"result": result, "cluster": cluster})

        if not IMAGE_MODEL_PERSISTENT:
            del resnet
            tf.keras.backend.clear_session()
            gc.collect()

        return response
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)


preload_image_models_once()
