import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import os
import joblib

df = pd.read_csv('../cats_and_dogs.csv')

X = df.drop('Animal', axis=1) 
y = df['Animal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)

dt_model.fit(X_train, y_train)

predictions = dt_model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"Decision Tree Accuracy: {accuracy * 100:.2f}%\n")

print("Detailed Performance Report:")
print(classification_report(y_test, predictions, target_names=['Cats (0)', 'Dogs (1)']))

print("\n" + "="*40)
print("HOW THE AI IS MAKING ITS DECISIONS:")
print("="*40)

plt.figure(figsize=(16, 10))

plot_tree(dt_model, 
          feature_names=list(X.columns), 
          class_names=['Cat', 'Dog'], # Tells it what 0 and 1 mean
          filled=True,                # Colors the boxes based on the prediction
          rounded=True,               # Makes the boxes look nice and modern
          fontsize=12)

plt.savefig('my_decision_tree.png', bbox_inches='tight')
os.makedirs("decision_tree_models", exist_ok=True)

joblib.dump(dt_model, "decision_tree_models/decision_tree_cat_dog.pkl")
