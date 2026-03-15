import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

df = pd.read_csv('../cats_and_dogs.csv')

# H (Height): Height of the animal (in cm).
# W (Weight): Weight of the animal (in kg).
# L (Length): Length of the animal (in cm).
# FL (Fur Length): Fur length categorized as 0 (Short), 1 (Medium), or 2 (Long).
# PS (Paw Size): Paw size categorized as 0 (Small), 1 (Medium), or 2 (Large).
# ES (Ear Shape): Ear shape categorized as 0 (Round), 1 (Fluffy), or 2 (Triangular).
# Animal (Target Label): The animal type, where 0 represents Cat and 1 represents Dog.

X = df.drop('Animal', axis=1) 
y = df['Animal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train_scaled, y_train)

predictions = knn.predict(X_test_scaled)

accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%\n")

print("Detailed Performance Report:")
print(classification_report(y_test, predictions, target_names=['Cats (0)', 'Dogs (1)']))

os.makedirs("knn_models", exist_ok=True)
joblib.dump(knn, "knn_models/knn_cat_dog_model.pkl")
joblib.dump(scaler, "knn_models/scaler.pkl")

print("Model and scaler saved successfully!")