# persona_analysis.py

import pandas as pd
import joblib

# Load data
df = pd.read_csv("train.csv")
label_encoder = joblib.load("label_encoder.pkl")

# Preprocessing
df['Family Expenses'] = df['Family Expenses'].map({'Low': 0, 'Average': 1, 'High': 2})
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['Work Experience'] = pd.to_numeric(df['Work Experience'], errors='coerce')
df['Family  Size'] = pd.to_numeric(df['Family  Size'], errors='coerce')
df.fillna(df.median(numeric_only=True), inplace=True)

# Convert target to integers using encoder
df['Segmentation'] = label_encoder.transform(df['Segmentation'])
df['Persona'] = label_encoder.inverse_transform(df['Segmentation'])

# Grouped averages
grouped_means = df.groupby('Persona')[['Age', 'Family  Size', 'Family Expenses', 'Work Experience']].mean().round(2)

print("📊 Persona-Wise Averages:")
print(grouped_means)

# Optional: Save to CSV
grouped_means.to_csv("persona_insights.csv")
