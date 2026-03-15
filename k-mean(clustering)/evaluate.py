import os
import cv2
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA

print("Loading models for evaluation...")

kmeans = joblib.load("kmeans_models/cat_dog_kmeans.pkl")

base_model = tf.keras.applications.ResNet50(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

base_folder = "./images/Petimages" 
categories = {"Dog": 0, "Cat": 1} 
batch_size = 32

print("\nCataloging all images...")

all_filepaths = []
all_labels = []

for category, label in categories.items():
    folder_path = os.path.join(base_folder, category)
    if not os.path.exists(folder_path):
        continue
        
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')) and not filename.startswith('._'):
            all_filepaths.append(os.path.join(folder_path, filename))
            all_labels.append(label)

total_images = len(all_filepaths)
print(f"Found {total_images} potential images to process.")

true_labels = []
predicted_labels = []
all_vectors = []

print("\nExtracting features in bulletproof batches...")

for i in range(0, total_images, batch_size):
    batch_paths = all_filepaths[i:i + batch_size]
    batch_labels_raw = all_labels[i:i + batch_size]
    
    valid_images = []
    valid_labels = []
    
    for path, label in zip(batch_paths, batch_labels_raw):
        img = cv2.imread(path) 
        
        if img is not None:
            try:
                img = cv2.resize(img, (224, 224))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                valid_images.append(img)
                valid_labels.append(label)
            except Exception as e:
                pass

    if not valid_images:
        continue

    batch_images_np = np.array(valid_images)
    processed_images = tf.keras.applications.resnet50.preprocess_input(batch_images_np)
    
    vectors = base_model.predict(processed_images, verbose=0)
    
    clusters = kmeans.predict(vectors.astype(np.float32))
    
    true_labels.extend(valid_labels)
    predicted_labels.extend(clusters)
    all_vectors.extend(vectors)
    
    if (i // batch_size) % 50 == 0:
        print(f"Processed batch {i // batch_size}... (Successfully read {len(true_labels)} images so far)")

# Convert to final numpy arrays
true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)
all_vectors = np.array(all_vectors)

# --- PRINT METRICS ---
print("\n" + "="*40)
print("EVALUATION RESULTS")
print("="*40)

accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Overall Accuracy: {accuracy * 100:.2f}%\n")

print("Detailed Classification Report:")
print(classification_report(true_labels, predicted_labels, target_names=["Dog", "Cat"]))

# --- GENERATE PNG PLOTS ---
print("\nGenerating visual reports (PNGs)...")
os.makedirs("reports", exist_ok=True)

# Plot A: Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
cm = confusion_matrix(true_labels, predicted_labels)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=["Predicted Dog", "Predicted Cat"], 
            yticklabels=["Actual Dog", "Actual Cat"])
plt.title('Cat vs Dog - Confusion Matrix')
plt.savefig('reports/confusion_matrix_full.png', bbox_inches='tight')
plt.close()

# Plot B: PCA Scatter Plot
print("Running PCA to visualize the AI's brain...")
pca = PCA(n_components=2)
vectors_2d = pca.fit_transform(all_vectors)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c=true_labels, cmap='coolwarm', alpha=0.3, edgecolors='none', s=10)
plt.title('PCA Scatter Plot: Cat vs Dog ResNet Features')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

handles, _ = scatter.legend_elements()
plt.legend(handles, ["Dogs", "Cats"])

plt.savefig('reports/pca_scatter_full.png', bbox_inches='tight')
plt.close()

print("Done! Check the 'reports' folder for your full dataset PNG images.")