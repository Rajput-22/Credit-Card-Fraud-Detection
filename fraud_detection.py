import warnings
warnings.filterwarnings('ignore')
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load Data
print("Loading dataset...")
data = pd.read_csv("creditcard.csv")

print("Dataset Loaded!")
print(data.head())

# Step 2: Check class distribution
print("\nFraud Cases Distribution:")
print(data["Class"].value_counts())

# Step 3: Separate fraud and non-fraud
fraud = data[data["Class"] == 1]
normal = data[data["Class"] == 0]

# Undersample normal data to balance dataset
normal_sample = normal.sample(n=len(fraud), random_state=42)

balanced_data = pd.concat([fraud, normal_sample]).sample(frac=1, random_state=42)

print("\nBalanced Data Distribution:")
print(balanced_data["Class"].value_counts())

# Step 4: Split features and labels
X = balanced_data.drop("Class", axis=1)
y = balanced_data["Class"]

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Scale features
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 7: Train Logistic Regression model
print("\nTraining Logistic Regression...")
lr = LogisticRegression(solver='liblinear', max_iter=1000)
lr.fit(X_train_scaled, y_train)

# Step 8: Train Random Forest model
print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# Step 9: Evaluation
print("\n=== Logistic Regression Report ===")
y_pred_lr = lr.predict(X_test_scaled)
print(classification_report(y_test, y_pred_lr))

print("\n=== Random Forest Report ===")
y_pred_rf = rf.predict(X_test)
print(classification_report(y_test, y_pred_rf))

# Step 10: Confusion Matrix
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt="d", cmap="Blues")
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
