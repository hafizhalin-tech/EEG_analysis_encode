# Full Multi-Label EEG Streamlit App (Template)
# NOTE: This is a full working template. Adjust feature extraction + model definitions as needed.

import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt

st.title("🧠 EEG Emotion Classification Dashboard")

uploaded_files = st.file_uploader("Upload EEG Feature Files", type=["csv", "xlsx"], accept_multiple_files=True)
emotion_labels = st.text_input("Enter all emotion labels (comma-separated)", "relax,focus,fear,nervous,surprise")
label_list = [x.strip() for x in emotion_labels.split(',')]

mlb = MultiLabelBinarizer(classes=label_list)

if uploaded_files:
    all_features = []
    all_labels = []
    channel_names = None

    for file in uploaded_files:
        df = pd.read_csv(file) if file.name.endswith("csv") else pd.read_excel(file)

        # Expecting a column named "Labels" containing list-like labels
        labels_raw = df["Labels"].apply(lambda x: x.split('|'))
        Y = mlb.fit_transform(labels_raw)

        X = df.drop(columns=["Labels"])
        if channel_names is None:
            channel_names = [c for c in X.columns if "CH" in c.upper()]

        all_features.append(X)
        all_labels.append(Y)

    X = np.vstack(all_features)
    Y = np.vstack(all_labels)

    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

    # Define models
    models = {
        "Random Forest": MultiOutputClassifier(RandomForestClassifier(n_estimators=150, random_state=42)),
        "KNN": MultiOutputClassifier(KNeighborsClassifier(n_neighbors=5)),
        "SVM (RBF)": MultiOutputClassifier(SVC(kernel='rbf', probability=True))
    }

    results = {}

    st.header("Model Training & Performance")
    for model_name, model in models.items():
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        acc = accuracy_score(Y_test, Y_pred)
        f1 = f1_score(Y_test, Y_pred, average='micro')
        results[model_name] = (model, acc, f1)
        st.write(f"### {model_name}")
        st.write(f"Accuracy: {acc:.4f} | F1-score: {f1:.4f}")

        # Confusion Matrix (per label)
        st.write("Confusion Matrices (per Emotion Label)")
        for idx, label in enumerate(label_list):
            cm = confusion_matrix(Y_test[:, idx], Y_pred[:, idx])
            fig, ax = plt.subplots()
            ax.imshow(cm)
            ax.set_title(f"CM – {label}")
            st.pyplot(fig)

    st.header("Feature Importance Analysis (All Classifiers)")

    # Global feature importance via Mutual Information
    mi_scores = mutual_info_classif(X, Y[:,0])  # MI per-feature vs ONE label (expandable)
    fig, ax = plt.subplots()
    ax.bar(range(len(mi_scores)), mi_scores)
    ax.set_title("Mutual Information Feature Importance")
    st.pyplot(fig)

    st.subheader("Channel‑wise Accuracy")
    channel_acc = {}

    for ch in channel_names:
        idx = X.columns.get_loc(ch)
        X_single = X_scaled[:, idx].reshape(-1, 1)
        X_train_c, X_test_c, Y_train_c, Y_test_c = train_test_split(X_single, Y, test_size=0.2, random_state=42)

        clf = MultiOutputClassifier(RandomForestClassifier())
        clf.fit(X_train_c, Y_train_c)
        pred = clf.predict(X_test_c)

        acc = accuracy_score(Y_test_c, pred)
        channel_acc[ch] = acc

    # Plot bar chart for channel accuracy
    fig, ax = plt.subplots()
    ax.bar(range(len(channel_acc)), list(channel_acc.values()))
    ax.set_xticks(range(len(channel_acc)))
    ax.set_xticklabels(channel_acc.keys(), rotation=45)
    ax.set_title("Single‑Channel Classification Accuracy")
    st.pyplot(fig)

    st.subheader("Comparison: Channel Accuracy vs Feature Importance")

    # Simple channel‑importance = average MI over channel’s features
    channel_importance = {}
    for ch in channel_names:
        idx = X.columns.get_loc(ch)
        channel_importance[ch] = mi_scores[idx]

    # Plot comparison
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(channel_acc.keys(), channel_acc.values(), marker='o', label='Accuracy')
    ax.plot(channel_importance.keys(), channel_importance.values(), marker='s', label='MI Importance')
    ax.legend()
    ax.set_title("Accuracy vs Importance per Channel")
    plt.xticks(rotation=45)
    st.pyplot(fig)
