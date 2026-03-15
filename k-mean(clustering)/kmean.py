import tensorflow as tf
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from PIL import Image, UnidentifiedImageError
import os
import time
import joblib

start_total = time.time()

print("Starting image clustering pipeline...\n")

# ---------------- GPU CHECK ----------------
print("Checking GPU availability...")

gpus = tf.config.list_physical_devices('GPU')
print("GPUs detected:", gpus)

if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("GPU memory growth enabled\n")
else:
    print("WARNING: No GPU detected, running on CPU\n")

# ---------------- MODEL LOAD ----------------
print("Loading ResNet50 feature extractor...")

start_model = time.time()

base_model = tf.keras.applications.ResNet50(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

print(f"Model loaded in {time.time() - start_model:.2f} seconds\n")

# ---------------- IMAGE LIST ----------------
image_folder = "./t"

image_files = [
    f for f in os.listdir(image_folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

total_images = len(image_files)

print(f"Found {total_images} images\n")

if total_images == 0:
    raise RuntimeError("No images found in the folder.")

# ---------------- FEATURE EXTRACTION ----------------
print("Starting feature extraction...\n")

start_features = time.time()

batch_size = 32
vectors = []
valid_files = []

total_batches = total_images // batch_size + 1

for i in range(0, total_images, batch_size):

    batch_files = image_files[i:i + batch_size]
    batch_images = []

    print(f"Preparing batch {i//batch_size + 1}/{total_batches}")

    for file in batch_files:

        path = os.path.join(image_folder, file)

        try:
            img = Image.open(path).convert("RGB")
            img = img.resize((224, 224))

            img_array = np.array(img)

            batch_images.append(img_array)
            valid_files.append(file)

        except (UnidentifiedImageError, OSError):
            print("Skipping unreadable image:", file)

    if not batch_images:
        continue

    batch_images = np.array(batch_images)

    batch_images = tf.keras.applications.resnet50.preprocess_input(batch_images)

    print("Running ResNet inference...")

    batch_vectors = base_model.predict(batch_images, verbose=0)

    vectors.extend(batch_vectors)

    print(
        f"Batch complete. Processed {min(i + batch_size, total_images)} / {total_images} images\n"
    )

vectors = np.array(vectors)

print("Feature extraction finished.")
print(f"Time taken: {time.time() - start_features:.2f} seconds\n")

# ---------------- CLUSTERING ----------------
print("Running MiniBatchKMeans clustering...\n")

start_kmeans = time.time()

kmeans = MiniBatchKMeans(
    n_clusters=2,
    batch_size=1024,
    random_state=42
)

clusters = kmeans.fit_predict(vectors)

print(f"KMeans finished in {time.time() - start_kmeans:.2f} seconds\n")

# ---------------- RESULTS ----------------
print("Cluster assignments:\n")

for i, file in enumerate(valid_files):
    print(file, "-> Cluster", clusters[i])

print("\nPipeline complete.")
print(f"Total runtime: {time.time() - start_total:.2f} seconds")

print("Saving clustering model...")

os.makedirs("kmeans_models", exist_ok=True)

joblib.dump(kmeans, "kmeans_models/cat_dog_kmeans.pkl")

print("Model saved to kmeans_models/cat_dog_kmeans.pkl")