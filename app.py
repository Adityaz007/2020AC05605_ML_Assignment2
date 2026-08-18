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

from sklearn.model_selection import train_test_split


st.title("Wine Quality Classification")

# Load dataset Red Winequality
url_red = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
wine_red = pd.read_csv(url_red, sep=';')

# Load dataset White Winequality
url_white = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
wine_white = pd.read_csv(url_white, sep=';')

# Step 4: Combine both datasets
wine_data = pd.concat([wine_red, wine_white], axis=0)


X = wine_data.drop('quality', axis=1)
y = wine_data['quality']

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Combine X_test and y_test into one DataFrame
test_data = X_test.copy()
test_data['quality'] = y_test

# Save to CSV
test_data.to_csv("test_data.csv", index=False)

print("Dataset shape:", wine_data.shape)
print(wine_data.head())
print(wine_data.info())

#Step - Preprocessing
wine_data['quality_label'] = (wine_data['quality'] >= 7).astype(int)

X = wine_data.drop(['quality', 'quality_label'], axis=1)
y = wine_data['quality_label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Step Evaluation Metrics

from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

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

#Step - Run all models

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

# Display results
import pandas as pd
results_df = pd.DataFrame(results).T
print(results_df)

if test_data is not None:
    st.write("Test Data Preview:", test_data.head())

    # Separate features and target
    X_test = test_data.drop("quality", axis=1)
    y_test = test_data["quality"]

    # Model selection dropdown
    model_choice = st.selectbox("Choose a model", 
                                ["Logistic Regression", "Decision Tree", "kNN", "Naive Bayes", "Random Forest"])

if model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
        # Normally you’d load a pre-trained model here
        model.fit(X_test, y_test)  # ⚠️ Replace with loading saved model
        y_pred = model.predict(X_test)
elif model_choice == "Decision Tree Classifier":
          model = DecisionTreeClassifier(max_iter=1000)
        # Normally you’d load a pre-trained model here
          model.fit(X_test, y_test)  # ⚠️ Replace with loading saved model
          y_pred = model.predict(X_test)
elif model_choice == "K-Nearest Neighbor Classifier":
          model = KNeighborsClassifier(max_iter=1000)
        # Normally you’d load a pre-trained model here
          model.fit(X_test, y_test)  # ⚠️ Replace with loading saved model
          y_pred = model.predict(X_test)
elif model_choice == "Naive Bayes Classifier - Gaussian or Multinomial":
          model = GaussianNB(max_iter=1000)
        # Normally you’d load a pre-trained model here
          model.fit(X_test, y_test)  # ⚠️ Replace with loading saved model
          y_pred = model.predict(X_test)
elif model_choice == "Ensemble Model - Random Forest":
          model = RandomForestClassifier(max_iter=1000)
        # Normally you’d load a pre-trained model here
          model.fit(X_test, y_test)  # ⚠️ Replace with loading saved model
          y_pred = model.predict(X_test)

# Show metrics
acc = accuracy_score(y_test, y_pred)
st.write("Accuracy:", acc)