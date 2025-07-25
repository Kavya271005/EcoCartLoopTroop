# comparative_modeling.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
    auc
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# -----------------------------------------------
# STEP 1: Load and Preprocess the Data
# -----------------------------------------------

df = pd.read_csv("train.csv")

# Label encode target
le = LabelEncoder()
df["Segmentation"] = le.fit_transform(df["Segmentation"])
joblib.dump(le, "label_encoder.pkl")

# Ordinal encode
df["Family Expenses"] = df["Family Expenses"].map({"Low": 0, "Average": 1, "High": 2})

# Convert numerics
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
df["Work Experience"] = pd.to_numeric(df["Work Experience"], errors="coerce")
df["Family  Size"] = pd.to_numeric(df["Family  Size"], errors="coerce")

# Fill missing
df.fillna(df.median(numeric_only=True), inplace=True)

# One-hot encoding
categorical = ["Sex", "Graduated", "Career", "Bachelor", "Variable"]
df = pd.get_dummies(df, columns=categorical)

# Drop irrelevant
df.drop(["ID", "Description"], axis=1, errors="ignore", inplace=True)

# Features & target
X = df.drop("Segmentation", axis=1)
y = df["Segmentation"]

# Save model input columns
joblib.dump(list(X.columns), "model_columns.pkl")

# -----------------------------------------------
# STEP 2: Define Models
# -----------------------------------------------

models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42),
    "LightGBM": LGBMClassifier(random_state=42)
}

# -----------------------------------------------
# STEP 3: Cross-Validated Model Evaluation
# -----------------------------------------------

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

print("\n--- Cross-Validation Results ---")
for name, model in models.items():
    accs, precisions, recalls, f1s = [], [], [], []
    for train_idx, val_idx in kf.split(X, y):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_val_fold)
        accs.append(accuracy_score(y_val_fold, y_pred))
        p, r, f1, _ = precision_recall_fscore_support(y_val_fold, y_pred, average="macro")
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
    results[name] = {
        "Accuracy": np.mean(accs),
        "Precision": np.mean(precisions),
        "Recall": np.mean(recalls),
        "F1-score": np.mean(f1s)
    }

results_df = pd.DataFrame(results).T
print(results_df)

# Plot
results_df.plot(kind="bar", figsize=(10, 6), colormap="viridis", legend=True)
plt.title("Model Comparison (5-Fold CV)")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.show()

# -----------------------------------------------
# STEP 4: Final Model Evaluation on Test Set
# -----------------------------------------------

# Final train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Choose best model (based on F1-score)
best_model_name = results_df["F1-score"].idxmax()
best_model = models[best_model_name]
print(f"\n✅ Best Model: {best_model_name}")

best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

# Save model
joblib.dump(best_model, "personality_predictor.pkl")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
disp.plot(cmap="Blues")
plt.title(f"Confusion Matrix - {best_model_name}")
plt.savefig("confusion_matrix.png")
plt.show()

# Classification Report
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# -----------------------------------------------
# STEP 5: Multiclass ROC-AUC
# -----------------------------------------------

# Binarize target
y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
y_score = best_model.predict_proba(X_test)
n_classes = y_test_bin.shape[1]

fpr, tpr, roc_auc = {}, {}, {}

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves
plt.figure(figsize=(10, 7))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f"{le.inverse_transform([i])[0]} (AUC = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curves - {best_model_name}")
plt.legend(loc="lower right")
plt.grid()
plt.tight_layout()
plt.savefig("roc_curves.png")
plt.show()
