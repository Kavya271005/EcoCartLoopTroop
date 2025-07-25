import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load training data
df = pd.read_csv('train.csv')

# Encode labels
le = LabelEncoder()
df['Segmentation'] = le.fit_transform(df['Segmentation'])
joblib.dump(le, 'label_encoder.pkl')  # Save encoder

# Map ordinal columns
df['Family Expenses'] = df['Family Expenses'].map({'Low': 0, 'Average': 1, 'High': 2})

# Convert to numeric
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['Work Experience'] = pd.to_numeric(df['Work Experience'], errors='coerce')
df['Family  Size'] = pd.to_numeric(df['Family  Size'], errors='coerce')

# Fill missing values
df.fillna(df.median(numeric_only=True), inplace=True)

# One-hot encode all categoricals
categorical_cols = ['Sex', 'Graduated', 'Career', 'Bachelor', 'Variable']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

# Drop unnecessary
df.drop(columns=['ID', 'Description'], inplace=True, errors='ignore')

# Save feature columns
feature_columns = [col for col in df.columns if col != 'Segmentation']
joblib.dump(feature_columns, 'model_columns.pkl')

# Split
X = df[feature_columns]
y = df['Segmentation']

x_train, x_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Train
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(x_train, y_train)

# Evaluate
y_pred = model.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Save model
joblib.dump(model, 'personality_predictor.pkl')

print("Class distribution:")
print(df['Segmentation'].value_counts())
X_test.to_csv("test.csv", index=False)


X = df.drop("Segmentation", axis=1)
y = df["Segmentation"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

