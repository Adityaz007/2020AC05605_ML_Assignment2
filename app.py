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
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Wine Quality Classification")

# a. Dataset upload option (CSV)
uploaded_file = st.file_uploader("Upload your test dataset (CSV)", type="csv")

if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Test Data Preview")
    st.dataframe(test_data.head())

    # Preprocessing: binary labels
    if "quality" in test_data.columns:
        test_data['quality_label'] = (test_data['quality'] >= 7).astype(int)
        X = test_data.drop(['quality', 'quality_label'], axis=1)
        y = test_data['quality_label']

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # b. Model selection dropdown
        model_choice = st.selectbox(
            "Choose a model",
            ["Logistic Regression", "Decision Tree", "kNN", "Naive Bayes", "Random Forest"]
        )

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

        # Train + predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None

        # c. Display evaluation metrics
        st.write("### Evaluation Metrics")
        st.write("Accuracy:", accuracy_score(y_test, y_pred))
        st.write("Precision:", precision_score(y_test, y_pred))
        st.write("Recall:", recall_score(y_test, y_pred))
        st.write("F1 Score:", f1_score(y_test, y_pred))
        st.write("MCC:", matthews_corrcoef(y_test, y_pred))
        if y_prob is not None:
            st.write("AUC:", roc_auc_score(y_test, y_prob))

        # d. Confusion matrix or classification report
        st.write("### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1], yticklabels=[0,1], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        st.write("### Classification Report")
        st.text(classification_report(y_test, y_pred))
else:
    st.info("Please upload a CSV file containing your test dataset.")
