# Step 1: Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 2: Load Fashion-MNIST Dataset
print("Loading Fashion-MNIST dataset...")
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

# Select a subset (e.g., 10,000 images for training)
X_train, _, y_train, _ = train_test_split(X_train_full, y_train_full, train_size=10000, random_state=42)

# Step 3: Visualize Sample Images
plt.figure(figsize=(10,4))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(X_train[i], cmap='gray')
    plt.axis('off')
plt.show()

# Step 4: Preprocess Data
X_train = X_train.reshape(len(X_train), -1) / 255.0  # Flatten & normalize
X_test = X_test.reshape(len(X_test), -1) / 255.0

# Step 5: Train-Test Split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
print(f"Train set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")

# Step 6: Optimize max_depth Using GridSearchCV
param_grid = {'max_depth': range(5, 15)}
grid_search = GridSearchCV(DecisionTreeClassifier(criterion="gini", random_state=42),
                           param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_depth = grid_search.best_params_['max_depth']
print(f"Optimal max_depth: {best_depth}")

# Step 7: Train Decision Tree Classifier
print("Training Decision Tree...")
dt_classifier = DecisionTreeClassifier(criterion="gini", max_depth=best_depth, random_state=42)
dt_classifier.fit(X_train, y_train)
print("Model training completed!")

# Step 8: Make Predictions
y_pred = dt_classifier.predict(X_test)

# Step 9: Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 10: Confusion Matrix Visualization
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
