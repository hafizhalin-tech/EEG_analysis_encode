# ===============================================
# EEG Emotion Classification Web App (Streamlit)
# ===============================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, multilabel_confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance

# ----------------------------------------------
# Streamlit Setup
# ----------------------------------------------
st.set_page_config(page_title="EEG Emotion Classifier", layout="wide")
st.title("🧠 EEG Emotion Classification Web App")
st.write("Upload EEG data, select classifier, and explore feature, channel, and parameter analyses interactively.")

# Channel list
EEG_CHANNELS = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2",
                "P8", "T8", "FC6", "F4", "F8", "AF4"]

# ----------------------------------------------
# File Upload
# ----------------------------------------------
uploaded_file = st.file_uploader("📤 Upload EEG CSV or XLSX File", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # === Read file ===
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success(f"✅ File uploaded successfully! Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        st.dataframe(df.head())

        # ------------------------------------------
        # Sidebar - Settings
        # ------------------------------------------
        st.sidebar.header("⚙️ Model Settings")

        label_type = st.sidebar.radio("Label Encoding Type", ["Single Column", "Multi-Column (One-Hot Encoded)"])

        if label_type == "Single Column":
            label_col = st.sidebar.text_input("Label Column Name (e.g., Emotion)", "Emotion")
        else:
            label_cols = st.sidebar.multiselect(
                "Select Multiple Label Columns (One-hot encoded labels)",
                [c for c in df.columns if c not in EEG_CHANNELS],
            )

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_features = st.sidebar.multiselect("Select EEG Features", numeric_cols, default=numeric_cols)

        classifier_name = st.sidebar.selectbox(
            "Select Classifier", ["KNN", "SVM", "Random Forest", "Neural Network"]
        )

        test_size = st.sidebar.slider("Test Size (Ratio for testing)", 0.1, 0.9, 0.2, step=0.05)
        run_button = st.sidebar.button("🚀 Run Classification")

        # =========================================================
        # RUN CLASSIFICATION
        # =========================================================
        if run_button:
            try:
                # ======== Prepare Labels ========
                if label_type == "Single Column":
                    if label_col not in df.columns:
                        st.error(f"Label column '{label_col}' not found.")
                        st.stop()
                    y = df[label_col]
                    le = LabelEncoder()
                    y = le.fit_transform(y)
                    label_classes = le.classes_
                else:
                    if not label_cols:
                        st.error("Please select label columns for one-hot encoding.")
                        st.stop()
                    y = df[label_cols].values
                    label_classes = label_cols

                # ======== Feature Scaling ========
                X = df[selected_features]
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=test_size, random_state=42
                )

                # ======== Model & Parameter Grid ========
                if classifier_name == "KNN":
                    model = KNeighborsClassifier()
                    param_grid = {"n_neighbors": [3, 5, 7, 9]}
                elif classifier_name == "SVM":
                    model = SVC(probability=True)
                    param_grid = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
                elif classifier_name == "Random Forest":
                    model = RandomForestClassifier(random_state=42)
                    param_grid = {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 7, None]}
                else:
                    model = MLPClassifier(max_iter=1000, random_state=42)
                    param_grid = {
                        "hidden_layer_sizes": [(50,), (100,), (100, 50)],
                        "activation": ["relu", "tanh", "logistic"]
                    }

                # ======== Grid Search ========
                st.info("⏳ Performing Grid Search... please wait...")
                grid = GridSearchCV(model, param_grid, cv=3, scoring="accuracy", n_jobs=-1, return_train_score=True)
                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_

                # ======== Predictions ========
                y_pred = best_model.predict(X_test)
                y_pred_train = best_model.predict(X_train)

                # ======== Metrics ========
                if label_type == "Single Column":
                    acc_train = accuracy_score(y_train, y_pred_train)
                    acc_test = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
                else:
                    acc_train = np.mean(np.all(y_train == y_pred_train, axis=1))
                    acc_test = np.mean(np.all(y_test == y_pred, axis=1))
                    prec = precision_score(y_test, y_pred, average="samples", zero_division=0)
                    rec = recall_score(y_test, y_pred, average="samples", zero_division=0)
                    f1 = f1_score(y_test, y_pred, average="samples", zero_division=0)

                # ======== Performance Display ========
                st.subheader("🏆 Model Performance Metrics")
                metrics_df = pd.DataFrame({
                    "Metric": ["Train Accuracy", "Test Accuracy", "Precision", "Recall", "F1-Score"],
                    "Score": [acc_train, acc_test, prec, rec, f1]
                })
                st.table(metrics_df)
                st.success(f"✅ Best Parameters: {grid.best_params_}")

                # ======== Confusion Matrix ========
                st.subheader("🔹 Confusion Matrix")
                if label_type == "Single Column":
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots(figsize=(5, 4))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                                xticklabels=label_classes, yticklabels=label_classes)
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    ax.set_title("Confusion Matrix")
                    st.pyplot(fig)
                else:
                    cms = multilabel_confusion_matrix(y_test, y_pred)
                    fig, axes = plt.subplots(1, len(label_classes), figsize=(3*len(label_classes), 4))
                    for i, (ax, label) in enumerate(zip(axes, label_classes)):
                        sns.heatmap(cms[i], annot=True, fmt="d", cmap="Blues", ax=ax)
                        ax.set_title(label)
                        ax.set_xlabel("Predicted")
                        ax.set_ylabel("Actual")
                    plt.tight_layout()
                    st.pyplot(fig)

                # ======== Hyperparameter Tuning Analysis ========
                st.subheader("🔍 Hyperparameter Tuning Analysis")
                grid_df = pd.DataFrame(grid.cv_results_)

                # Display top 10 results
                top_df = grid_df.sort_values("mean_test_score", ascending=False).head(10)
                st.dataframe(top_df[["params", "mean_train_score", "mean_test_score"]])

                # Plot parameter tuning
                params = list(param_grid.keys())
                if len(params) == 1:
                    param_name = params[0]
                    plt.figure(figsize=(6, 4))
                    plt.plot(grid_df["param_" + param_name], grid_df["mean_test_score"], marker='o')
                    plt.xlabel(param_name)
                    plt.ylabel("Mean CV Accuracy")
                    plt.title(f"{classifier_name} Parameter Tuning")
                    st.pyplot(plt)
                elif len(params) == 2:
                    pivot_table = grid_df.pivot_table(
                        values="mean_test_score",
                        index="param_" + params[0],
                        columns="param_" + params[1]
                    )
                    plt.figure(figsize=(6, 4))
                    sns.heatmap(pivot_table, annot=True, cmap="Blues")
                    plt.title(f"{classifier_name} Grid Search Accuracy")
                    st.pyplot(plt)

                # ======== Feature Importance ========
                st.subheader("🌟 Feature Importance")
                try:
                    if hasattr(best_model, "feature_importances_"):
                        importances = best_model.feature_importances_
                    else:
                        imp_result = permutation_importance(best_model, X_test, y_test, scoring="accuracy", n_repeats=10)
                        importances = imp_result.importances_mean

                    sorted_idx = np.argsort(importances)[::-1]
                    plt.figure(figsize=(8, 4))
                    plt.bar(range(len(importances)), importances[sorted_idx])
                    plt.xticks(range(len(importances)), np.array(selected_features)[sorted_idx], rotation=90)
                    plt.title(f"{classifier_name} Feature Importance")
                    plt.tight_layout()
                    st.pyplot(plt)
                except Exception as e:
                    st.warning(f"⚠️ Feature importance could not be computed: {e}")

                # ======== Channel-wise Accuracy ========
                st.subheader("🧩 Channel-Wise Accuracy Contribution")
                channel_acc = {}
                for ch in EEG_CHANNELS:
                    ch_features = [f for f in selected_features if ch in f]
                    if not ch_features:
                        continue
                    X_ch = df[ch_features]
                    X_train_ch, X_test_ch, y_train_ch, y_test_ch = train_test_split(
                        scaler.fit_transform(X_ch), y, test_size=test_size, random_state=42
                    )
                    best_model.fit(X_train_ch, y_train_ch)
                    y_pred_ch = best_model.predict(X_test_ch)
                    if label_type == "Single Column":
                        acc_ch = accuracy_score(y_test_ch, y_pred_ch)
                    else:
                        acc_ch = np.mean(np.all(y_test_ch == y_pred_ch, axis=1))
                    channel_acc[ch] = acc_ch

                if channel_acc:
                    ch_df = pd.DataFrame(list(channel_acc.items()), columns=["Channel", "Accuracy"])
                    ch_df = ch_df.sort_values("Accuracy", ascending=False)
                    st.dataframe(ch_df)
                    plt.figure(figsize=(8, 4))
                    sns.barplot(x="Channel", y="Accuracy", data=ch_df, palette="coolwarm")
                    plt.ylim(0, 1)
                    plt.title("Channel-wise Classification Accuracy")
                    st.pyplot(plt)

            except Exception as e:
                st.error(f"❌ Error during classification: {e}")

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
else:
    st.info("👆 Please upload a CSV or XLSX EEG dataset to start.")
