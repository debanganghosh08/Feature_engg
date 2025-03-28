# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 2: Load the dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Step 3: Basic Data Exploration
print(df.info())
print(df.head())

# Step 4: Handling Missing Values
df['Age'].fillna(df['Age'].median(), inplace=True)  # Fill missing ages with median
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)  # Fill missing Embarked with mode
df.drop(columns=['Cabin'], inplace=True)  # Drop Cabin (too many missing values)

# Step 5: Encode Categorical Variables
label_encoder = LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])  # Male: 1, Female: 0
df['Embarked'] = label_encoder.fit_transform(df['Embarked'])  # Convert Embarked to numeric

# Step 6: Feature Selection (Choosing important columns)
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = df[features]  # Independent variables
y = df['Survived']  # Target variable

# Step 7: Train-Test Split (70% Train, 30% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 8: Optimize max_depth using GridSearchCV
param_grid = {'max_depth': range(1, 15)}  # Testing max_depth from 1 to 14
grid_search = GridSearchCV(DecisionTreeClassifier(criterion="gini", random_state=42),
                           param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best max_depth
best_max_depth = grid_search.best_params_['max_depth']
print(f"Optimal max_depth: {best_max_depth}")

# Step 9: Train Decision Tree with optimized max_depth
dt_classifier = DecisionTreeClassifier(criterion="gini", max_depth=best_max_depth, random_state=42)
dt_classifier.fit(X_train, y_train)

# Step 10: Model Prediction
y_pred = dt_classifier.predict(X_test)

# Step 11: Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 12: Visualizing Feature Importance
feature_importances = pd.Series(dt_classifier.feature_importances_, index=features)
feature_importances.sort_values().plot(kind='barh', title="Feature Importance")
plt.show()
