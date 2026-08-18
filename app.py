import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

st.title("Wine Quality Classification")

# Load dataset Red Winequality
url_red = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
wine_red = pd.read_csv(url_red, sep=';')

# Load dataset White Winequality
url_white = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
wine_white = pd.read_csv(url_white, sep=';')

# Combine both datasets
wine_data = pd.concat([wine_red, wine_white], axis=0)

# Step - Preprocessing: create binary labels
wine_data['quality_label'] = (wine_data['quality'] >= 7).astype(int)

# Use binary labels for classification
X = wine_data.drop(['quality', 'quality_label'], axis=1)
y = wine_data['quality_label']

# Train/test split on binary labels
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Evaluation function
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob) if y_prob is not None else None
    }
    return metrics

# Run all models and display results
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "kNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    results[name] = evaluate_model(model, X_test, y_test)

results_df = pd.DataFrame(results).T
st.write("### Model Comparison Results")
st.dataframe(results_df)

# Model selection dropdown
model_choice = st.selectbox(
    "Choose a model",
    ["Logistic Regression", "Decision Tree", "kNN", "Naive Bayes", "Random Forest"]
)

# Select model
if model_choice == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)
elif model_choice == "Decision Tree":
    model = DecisionTreeClassifier()
elif model_choice == "kNN":
    model = KNeighborsClassifier(n_neighbors=5)
elif model_choice == "Naive Bayes":
    model = GaussianNB()
elif model_choice == "Random Forest":
    model = RandomForestClassifier(n_estimators=100)

# Model selection dropdown
model_choice = st.selectbox(
    "Choose a model",
    ["Logistic Regression", "Decision Tree", "kNN", "Naive Bayes", "Random Forest"]
)

# Select model
if model_choice == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)
elif model_choice == "Decision Tree":
    model = DecisionTreeClassifier()
elif model_choice == "kNN":
    model = KNeighborsClassifier(n_neighbors=5)
elif model_choice == "Naive Bayes":
    model = GaussianNB()
elif model_choice == "Random Forest":
    model = RandomForestClassifier(n_estimators=100)

# Train on training data
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None

# Show metrics
st.write("### Selected Model Performance")
st.write("Accuracy:", accuracy_score(y_test, y_pred))
st.write("Precision:", precision_score(y_test, y_pred))
st.write("Recall:", recall_score(y_test, y_pred))
st.write("F1 Score:", f1_score(y_test, y_pred))
st.write("MCC:", matthews_corrcoef(y_test, y_pred))
if y_prob is not None:
    st.write("AUC:", roc_auc_score(y_test, y_prob))
