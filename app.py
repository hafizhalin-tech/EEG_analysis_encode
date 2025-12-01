import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    hamming_loss, f1_score
)
from sklearn.feature_selection import mutual_info_classif

# Classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier


# ==============================
# STREAMLIT PAGE SETUP
# ==============================
st.set_page_config(page_title="EEG Multi-Label Emotion Classifier", layout="wide")

st.title("🧠 EEG Multi-Label Emotion Classification Dashboard")
st.write("Supports multi-hot encoded labels, channel analysis, tuning & feature importance.")


# ==============================
# FILE UPLOAD
# ==============================
uploaded_file = st.file_uploader("📂 Upload EEG Feature File (.csv or .xlsx)", type=["csv", "xlsx"])

if uploaded_file:
    # Load data
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success("📁 File loaded!")
    st.dataframe(df.head())

    # ==============================
    # SELECT MULTI-LABEL COLUMNS
    # ==============================
    st.subheader("🎭 Select Emotion Label Columns (multi-label)")

    # Auto-detect possible label columns (binary columns)
    possible_label_cols = [c for c in df.columns if df[c].nunique() <= 5 and df[c].dtype != "float64"]

    label_cols = st.multiselect(
        "Select Label Columns:",
        options=df.columns,
        default=possible_label_cols
    )

    if len(label_cols) == 0:
        st.warning("⚠ Please select at least one label column.")
        st.stop()

    st.success(f"Selected label columns: {label_cols}")

    # Feature matrix
    feature_cols = [c for c in df.columns if c not in label_cols]

    X = df[feature_cols].values
    y = df[label_cols].values  # Multi-hot encoded labels


    # ==============================
    # TRAIN/TEST SPLIT
    # ==============================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    # ==============================
    # CLASSIFIER SELECTION
    # ==============================
    st.sidebar.header("⚙ Classifier Settings")

    classifier_name = st.sidebar.selectbox(
        "Select Classifier",
        ["KNN", "SVM", "Random Forest", "Neural Network"]
    )

    tuning_mode = st.sidebar.radio(
        "Parameter Tuning Mode",
        ["Automatic (Grid Search)", "Manual"]
    )

    # Build model
    base_model = None
    param_grid = None

    # ------------------- KNN -------------------
    if classifier_name == "KNN":
        if tuning_mode == "Manual":
            k = st.sidebar.slider("n_neighbors", 1, 15, 5)
            base_model = KNeighborsClassifier(n_neighbors=k)
        else:
            base_model = KNeighborsClassifier()
            param_grid = {"n_neighbors": [3, 5, 7, 9, 11]}

    # ------------------- SVM -------------------
    elif classifier_name == "SVM":
        if tuning_mode == "Manual":
            Cv = st.sidebar.selectbox("C", [0.1, 1, 10], 1)
            kernel = st.sidebar.selectbox("Kernel", ["linear", "rbf"], 1)
            base_model = SVC(C=Cv, kernel=kernel, probability=True)
        else:
            base_model = SVC(probability=True)
            param_grid = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}

    # ------------------- Random Forest -------------------
    elif classifier_name == "Random Forest":
        if tuning_mode == "Manual":
            n_est = st.sidebar.selectbox("n_estimators", [50, 100, 200], 1)
            depth = st.sidebar.selectbox("max_depth", [3, 5, 7, None], 3)
            base_model = RandomForestClassifier(
                n_estimators=n_est, max_depth=depth, random_state=42
            )
        else:
            base_model = RandomForestClassifier(random_state=42)
            param_grid = {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 7, None]}

    # ------------------- Neural Network -------------------
    elif classifier_name == "Neural Network":
        if tuning_mode == "Manual":
            hidden = st.sidebar.selectbox("Hidden Layers", [(50,), (100,), (100, 50)], 1)
            activation = st.sidebar.selectbox("Activation", ["relu", "tanh", "logistic"], 0)
            base_model = MLPClassifier(
                hidden_layer_sizes=hidden,
                activation=activation,
                max_iter=1000,
                random_state=42
            )
        else:
            base_model = MLPClassifier(max_iter=1000, random_state=42)
            param_grid = {
                "hidden_layer_sizes": [(50,), (100,), (100, 50)],
                "activation": ["relu", "tanh", "logistic"]
            }

    # One-vs-Rest wrapper for multi-label classification
    model = OneVsRestClassifier(base_model)


    # ==============================
    # RUN BUTTON
    # ==============================
    if st.sidebar.button("🚀 Run Classification"):
        st.subheader("🔍 Classification Results")

        # -------- Grid Search ----------
        if tuning_mode == "Automatic (Grid Search)" and param_grid:
            st.info("⏳ Running Grid Search...")

            grid = GridSearchCV(
                model, param_grid={"estimator__" + k: v for k, v in param_grid.items()},
                cv=3, scoring="accuracy", n_jobs=-1
            )
            grid.fit(X_train, y_train)
            model = grid.best_estimator_

            st.success(f"Best Parameters: {grid.best_params_}")

            # Tuning plot
            plt.figure(figsize=(6, 3))
            plt.plot(grid.cv_results_["mean_test_score"])
            plt.title("Hyperparameter Tuning Accuracy")
            plt.ylabel("Accuracy")
            plt.xlabel("Parameter Index")
            st.pyplot(plt)

        else:
            model.fit(X_train, y_train)

        # -------- Prediction ----------
        y_pred = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, np.round(y_pred))
        f1_micro = f1_score(y_test, y_pred, average="micro")
        ham = hamming_loss(y_test, y_pred)

        st.metric("🎯 Multi-Label Accuracy", f"{acc*100:.2f}%")
        st.metric("🔥 F1-Score (micro)", f"{f1_micro*100:.2f}%")
        st.metric("📉 Hamming Loss", f"{ham:.4f}")

        st.text("Classification Report (per label):")
        st.text(classification_report(y_test, y_pred, target_names=label_cols))


        # ==============================
        # CONFUSION MATRIX PER LABEL
        # ==============================
        st.subheader("🧩 Confusion Matrix (Per Label)")

        for i, label in enumerate(label_cols):
            cm = confusion_matrix(y_test[:, i], y_pred[:, i])

            plt.figure(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title(f"Confusion Matrix - {label}")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            st.pyplot(plt)


        # ==============================
        # CHANNEL-WISE ACCURACY
        # ==============================
        st.subheader("📡 Channel-wise Accuracy (Multi-label)")

        channel_acc = []

        for i, ch in enumerate(feature_cols):
            X_ch = X[:, [i]]
            Xc_train, Xc_test, yc_train, yc_test = train_test_split(
                X_ch, y, test_size=0.2, random_state=42
            )

            # Rebuild model for single channel
            model_ch = OneVsRestClassifier(
                base_model.__class__(**base_model.get_params())
            )
            model_ch.fit(Xc_train, yc_train)
            pred_ch = model_ch.predict(Xc_test)

            # Use micro-F1 as accuracy metric
            acc_ch = f1_score(yc_test, pred_ch, average="micro")
            channel_acc.append(acc_ch * 100)

        ch_df = pd.DataFrame({"Channel": feature_cols, "Accuracy": channel_acc})
        st.dataframe(ch_df)

        plt.figure(figsize=(10, 3))
        sns.barplot(data=ch_df, x="Channel", y="Accuracy", palette="coolwarm")
        plt.xticks(rotation=45)
        plt.title("Single-Channel Multi-Label Accuracy (F1-micro)")
        st.pyplot(plt)


        # ==============================
        # FEATURE IMPORTANCE (multi-label)
        # ==============================
        st.subheader("🔥 Feature Importance (Multi-label)")

        # Random Forest has built-in feature importance
        if classifier_name == "Random Forest":
            importance = model.estimators_[0].feature_importances_
        else:
            # fallback to mutual info
            importance = mutual_info_classif(X_train, y_train[:, 0])

        imp_df = pd.DataFrame({"Feature": feature_cols, "Importance": importance})
        imp_df = imp_df.sort_values("Importance", ascending=False)

        st.dataframe(imp_df)

        plt.figure(figsize=(10, 3))
        sns.barplot(x="Feature", y="Importance", data=imp_df, palette="mako")
        plt.xticks(rotation=45)
        plt.title("Feature Importance")
        st.pyplot(plt)


        # ==============================
        # COMBINED PLOT
        # ==============================
        st.subheader("📊 Channel Accuracy vs Feature Importance")

        merged = ch_df.merge(imp_df, left_on="Channel", right_on="Feature")

        fig, ax1 = plt.subplots(figsize=(10, 3))
        sns.barplot(x="Channel", y="Accuracy", data=merged, ax=ax1, color="skyblue")
        ax1.set_ylabel("Accuracy (F1-micro)")

        ax2 = ax1.twinx()
        sns.lineplot(x="Channel", y="Importance", data=merged, ax=ax2, marker="o", color="red")
        ax2.set_ylabel("Importance")

        plt.xticks(rotation=45)
        plt.title("Channel Accuracy vs Feature Importance")
        st.pyplot(fig)

else:
    st.info("⬆ Upload a dataset to begin.")
