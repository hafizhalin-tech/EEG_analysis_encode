# ===============================================
# EEG Emotion Classification Web App (Streamlit)
# ===============================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance

# ---------------- Streamlit Setup ----------------
st.set_page_config(page_title="EEG Emotion Classifier", layout="wide")
st.title("🧠 EEG Emotion Classification")
st.write("Upload EEG data, select features and classifier, then visualize results interactively.")

# ---------------- File Upload ----------------
uploaded_file = st.file_uploader("📤 Upload EEG CSV or XLSX File", type=["csv", "xlsx"])

# EEG channel list
channels = ["AF3", "F7", "F3", "FC5", "T7", "P7",
            "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success(f"✅ File uploaded successfully! Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        st.dataframe(df.head())

        # ---------------- Sidebar Settings ----------------
        st.sidebar.header("⚙️ Model Settings")
        label_col = st.sidebar.text_input("Label Column Name (e.g., Emotion)", "Emotion")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_features = st.sidebar.multiselect("Select EEG Features", numeric_cols, default=numeric_cols)
        classifier_name = st.sidebar.selectbox("Select Classifier", ["KNN", "SVM", "Random Forest", "Neural Network"])
        test_size = st.sidebar.slider("Test Size (Ratio of data for testing)", 0.1, 0.95, 0.2, step=0.05)
        run_button = st.sidebar.button("🚀 Run Classification")

        if run_button:
            if label_col not in df.columns:
                st.error(f"Label column '{label_col}' not found.")
            else:
                try:
                    # ---------------- Label Encoding (Single or Multi) ----------------
                    y_raw = df[label_col]
                    if y_raw.dtype == "object":
                        # Handle comma-separated multi-label (e.g., "relax,focus")
                        y_processed = y_raw.astype(str).str.split(",")
                        mlb = MultiLabelBinarizer()
                        y = mlb.fit_transform(y_processed)
                        label_mode = "multi"
                    else:
                        # Single-label encoding
                        le = LabelEncoder()
                        y = le.fit_transform(y_raw)
                        label_mode = "single"

                    X = df[selected_features]

                    # ---------------- Train-Test Split ----------------
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42
                    )

                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)

                    # ---------------- Classifier & Parameter Grid ----------------
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
                        param_grid = {"hidden_layer_sizes": [(50,), (100,), (100, 50)],
                                      "activation": ["relu", "tanh", "logistic"]}

                    # ---------------- Grid Search ----------------
                    st.info("⏳ Running Grid Search... please wait...")
                    grid = GridSearchCV(model, param_grid, cv=3, scoring="accuracy", n_jobs=-1, return_train_score=True)
                    grid.fit(X_train, y_train)
                    best_model = grid.best_estimator_

                    # ---------------- Performance Metrics ----------------
                    y_pred = best_model.predict(X_test)
                    y_pred_train = best_model.predict(X_train)

                    if label_mode == "multi":
                        acc_test = np.mean(np.all(y_test == y_pred, axis=1))
                        acc_train = np.mean(np.all(y_train == y_pred_train, axis=1))
                        prec = recall = f1 = np.nan
                    else:
                        acc_test = accuracy_score(y_test, y_pred)
                        acc_train = accuracy_score(y_train, y_pred_train)
                        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

                    # ---------------- Display Metrics ----------------
                    st.subheader("🏆 Model Performance Metrics")
                    metrics_df = pd.DataFrame({
                        "Metric": ["Train Accuracy", "Test Accuracy", "Precision", "Recall", "F1-Score"],
                        "Score": [acc_train, acc_test, prec, recall, f1]
                    })
                    st.table(metrics_df)
                    st.write(f"**Best Parameters:** {grid.best_params_}")

                    # ---------------- Accuracy Bar Chart ----------------
                    st.subheader("📊 Training vs Testing Accuracy")
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.barplot(x=["Training", "Testing"], y=[acc_train, acc_test], palette="Blues", ax=ax)
                    ax.set_ylim(0, 1)
                    for i, v in enumerate([acc_train, acc_test]):
                        ax.text(i, v + 0.01, f"{v:.2f}", ha='center')
                    st.pyplot(fig)

                    # ---------------- Hyperparameter Tuning Visualization ----------------
                    st.subheader("🔍 Parameter Tuning Results (Mean CV Accuracy)")
                    grid_df = pd.DataFrame(grid.cv_results_)
                    params = list(param_grid.keys())
                    try:
                        pivot_table = grid_df.pivot_table(
                            values="mean_test_score",
                            index="param_" + params[0],
                            columns=("param_" + params[1] if len(params) > 1 else None)
                        )
                        plt.figure(figsize=(6, 4))
                        sns.heatmap(pivot_table, annot=True, cmap="Blues")
                        plt.title(f"{classifier_name} Grid Search Accuracy")
                        st.pyplot(plt)
                    except Exception:
                        plt.figure(figsize=(6, 4))
                        plt.plot(grid_df["mean_test_score"], marker='o')
                        plt.title("Mean CV Accuracy per Combination")
                        st.pyplot(plt)

                    # ---------------- Confusion Matrix ----------------
                    if label_mode == "single":
                        st.subheader("🔹 Confusion Matrix")
                        cm = confusion_matrix(y_test, y_pred)
                        plt.figure(figsize=(5, 4))
                        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                                    xticklabels=le.classes_, yticklabels=le.classes_)
                        plt.xlabel("Predicted")
                        plt.ylabel("Actual")
                        st.pyplot(plt)

                    # ---------------- Channel-wise Accuracy ----------------
                    st.subheader("🎧 Channel-wise Classification Accuracy")
                    channel_acc = []
                    X_scaled = scaler.fit_transform(X)
                    for ch in channels:
                        ch_features = [c for c in X.columns if c.startswith(ch)]
                        if len(ch_features) == 0:
                            channel_acc.append(np.nan)
                            continue
                        X_ch = X[ch_features]
                        X_train_ch, X_test_ch, y_train_ch, y_test_ch = train_test_split(X_ch, y, test_size=test_size, random_state=42)
                        model_ch = RandomForestClassifier(random_state=42)
                        model_ch.fit(X_train_ch, y_train_ch)
                        y_pred_ch = model_ch.predict(X_test_ch)
                        if label_mode == "multi":
                            acc_ch = np.mean(np.all(y_test_ch == y_pred_ch, axis=1))
                        else:
                            acc_ch = accuracy_score(y_test_ch, y_pred_ch)
                        channel_acc.append(acc_ch)

                    # ---------------- Feature Importance per Channel ----------------
                    if hasattr(best_model, "feature_importances_"):
                        importances = pd.Series(best_model.feature_importances_, index=X.columns)
                        channel_importance = []
                        for ch in channels:
                            ch_imp = importances[[c for c in X.columns if c.startswith(ch)]].sum() if any(X.columns.str.startswith(ch)) else 0
                            channel_importance.append(ch_imp)
                    else:
                        # Use permutation importance as fallback
                        imp_result = permutation_importance(best_model, X_test, y_test, scoring="accuracy", n_repeats=5)
                        importances = pd.Series(imp_result.importances_mean, index=X.columns)
                        channel_importance = []
                        for ch in channels:
                            ch_imp = importances[[c for c in X.columns if c.startswith(ch)]].sum() if any(X.columns.str.startswith(ch)) else 0
                            channel_importance.append(ch_imp)

                    # ---------------- Combined Visualization ----------------
                    st.subheader("📈 Channel Accuracy vs Feature Importance")
                    fig, ax1 = plt.subplots(figsize=(10, 6))
                    sns.barplot(x=channels, y=channel_acc, color="skyblue", label="Channel Accuracy", ax=ax1)
                    ax1.set_ylabel("Channel Accuracy", color="blue")
                    ax1.set_ylim(0, 1)

                    ax2 = ax1.twinx()
                    sns.lineplot(x=channels, y=channel_importance, color="red", marker="o", label="Feature Importance", ax=ax2)
                    ax2.set_ylabel("Feature Importance", color="red")

                    plt.title("Channel-wise Accuracy vs Overall Model Importance")
                    plt.xticks(rotation=45)
                    ax1.legend(loc="upper left")
                    ax2.legend(loc="upper right")
                    plt.tight_layout()
                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"❌ Error during classification: {e}")

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
else:
    st.info("👆 Please upload a CSV or XLSX EEG dataset to start.")
