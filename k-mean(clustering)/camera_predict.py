import cv2
import tensorflow as tf
import numpy as np
import joblib

print("Loading models...")

kmeans = joblib.load("kmeans_models/cat_dog_kmeans.pkl")

base_model = tf.keras.applications.ResNet50(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

print("Models loaded")

def image_to_vector(frame):

    img = cv2.resize(frame, (224,224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = tf.keras.applications.resnet50.preprocess_input(img)

    img = np.expand_dims(img, axis=0)

    vector = base_model.predict(img, verbose=0)

    return vector.flatten().astype(np.float32)


print("Starting webcam...")

cap = cv2.VideoCapture(0)

prediction_text = "Press SPACE to classify"

while True:

    ret, frame = cap.read()

    if not ret:
        break

    cv2.putText(
        frame,
        prediction_text,
        (20,50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.imshow("Cat vs Dog Classifier", frame)

    key = cv2.waitKey(1) & 0xFF

    # SPACE → classify frame
    if key == 32:

        print("Classifying frame...")

        vector = image_to_vector(frame)

        cluster = kmeans.predict(np.array([vector], dtype=np.float32))[0]

        if cluster == 0:
            prediction_text = "Prediction: Dog"
        else:
            prediction_text = "Prediction: Cat"

        print(prediction_text)

    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()