import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import mutual_info_classif

st.set_page_config(page_title="EEG Emotion Classification", layout="wide")

# ==================== APP HEADER ====================
st.title("🧠 EEG Emotion Classification Dashboard")
st.write("Compare classifier performance, channel importance, and tuning analysis.")

# ==================== FILE UPLOAD ====================
uploaded_file = st.file_uploader("📂 Upload EEG Feature File (.csv or .xlsx)", type=["csv", "xlsx"])
if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    st.success("✅ File loaded successfully!")

    st.write("### Preview of Data")
    st.dataframe(df.head())

    # ==================== PREPROCESSING ====================
    st.subheader("⚙️ Data Preprocessing")
    label_col = st.selectbox("Select Label Column", df.columns)
    feature_cols = [c for c in df.columns if c != label_col]
    X = df[feature_cols].values
    y = df[label_col].values

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    labels = le.classes_

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ==================== CLASSIFIER SELECTION ====================
    st.sidebar.header("🧩 Classifier & Parameters")

    if "run_triggered" not in st.session_state:
        st.session_state.run_triggered = False

    classifier_name = st.sidebar.selectbox(
        "Select Classifier", ["KNN", "SVM", "Random Forest", "Neural Network"]
    )

    tuning_mode = st.sidebar.radio("Parameter Tuning Mode", ["Automatic (Grid Search)", "Manual"])

    # --- Manual Parameter Inputs with session_state memory ---
    model, param_grid = None, None

    if classifier_name == "KNN":
        if tuning_mode == "Manual":
            n_neighbors = st.sidebar.selectbox(
                "n_neighbors", [1, 3, 5, 7, 9],
                index=st.session_state.get("knn_n", 2)
            )
            st.session_state.knn_n = [1, 3, 5, 7, 9].index(n_neighbors)
            model = KNeighborsClassifier(n_neighbors=n_neighbors)
        else:
            model = KNeighborsClassifier()
            param_grid = {"n_neighbors": [3, 5, 7, 9]}

    elif classifier_name == "SVM":
        if tuning_mode == "Manual":
            C_val = st.sidebar.selectbox(
                "C", [0.1, 1, 10],
                index=st.session_state.get("svm_C", 1)
            )
            kernel_val = st.sidebar.selectbox(
                "Kernel", ["linear", "rbf"],
                index=st.session_state.get("svm_kernel", 1)
            )
            st.session_state.svm_C = [0.1, 1, 10].index(C_val)
            st.session_state.svm_kernel = ["linear", "rbf"].index(kernel_val)
            model = SVC(C=C_val, kernel=kernel_val, probability=True)
        else:
            model = SVC(probability=True)
            param_grid = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}

    elif classifier_name == "Random Forest":
        if tuning_mode == "Manual":
            n_estimators = st.sidebar.selectbox(
                "n_estimators", [50, 100, 200],
                index=st.session_state.get("rf_n", 1)
            )
            max_depth = st.sidebar.selectbox(
                "max_depth", [3, 5, 7, None],
                index=st.session_state.get("rf_d", 3)
            )
            st.session_state.rf_n = [50, 100, 200].index(n_estimators)
            st.session_state.rf_d = [3, 5, 7, None].index(max_depth)
            model = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth, random_state=42
            )
        else:
            model = RandomForestClassifier(random_state=42)
            param_grid = {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 7, None]}

    elif classifier_name == "Neural Network":
        if tuning_mode == "Manual":
            hidden = st.sidebar.selectbox(
                "Hidden Layers", [(50,), (100,), (100, 50)],
                index=st.session_state.get("mlp_h", 1)
            )
            activation = st.sidebar.selectbox(
                "Activation", ["relu", "tanh", "logistic"],
                index=st.session_state.get("mlp_a", 0)
            )
            st.session_state.mlp_h = [(50,), (100,), (100, 50)].index(hidden)
            st.session_state.mlp_a = ["relu", "tanh", "logistic"].index(activation)
            model = MLPClassifier(hidden_layer_sizes=hidden, activation=activation,
                                  max_iter=1000, random_state=42)
        else:
            model = MLPClassifier(max_iter=1000, random_state=42)
            param_grid = {
                "hidden_layer_sizes": [(50,), (100,), (100, 50)],
                "activation": ["relu", "tanh", "logistic"]
            }

    # ==================== RUN CLASSIFICATION BUTTON ====================
    run_button = st.sidebar.button("🚀 Run Classification")

    if run_button:
        st.session_state.run_triggered = True

    if st.session_state.run_triggered:
        st.subheader("🔍 Classification Results")

        # === Model training ===
        if tuning_mode == "Automatic (Grid Search)" and param_grid:
            st.info("⏳ Performing Automatic Grid Search...")
            grid = GridSearchCV(model, param_grid, cv=3, scoring="accuracy", n_jobs=-1, return_train_score=True)
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            st.success(f"✅ Best Parameters: {grid.best_params_}")

            # Plot tuning results
            plt.figure(figsize=(6, 3))
            plt.plot(grid.cv_results_['mean_test_score'], 'o-', label='Validation Accuracy')
            plt.title("Hyperparameter Tuning Performance")
            plt.xlabel("Parameter Set Index")
            plt.ylabel("Accuracy")
            plt.legend()
            st.pyplot(plt)

        else:
            st.info("⚙️ Using Manual Hyperparameter Settings")
            model.fit(X_train, y_train)
            best_model = model

        # === Prediction and evaluation ===
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.metric("🎯 Overall Test Accuracy", f"{acc*100:.2f}%")

        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred, target_names=labels))

        # === Confusion Matrix ===
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.title("Confusion Matrix")
        st.pyplot(plt)

        # ==================== CHANNEL-WISE ACCURACY ====================
        st.subheader("📡 Channel-wise Classification Accuracy")
        channel_acc = []
        for i, ch in enumerate(feature_cols):
            X_ch = X[:, [i]]
            X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_ch, y_encoded, test_size=0.2, random_state=42)
            model_c = best_model.__class__(**best_model.get_params())
            model_c.fit(X_train_c, y_train_c)
            y_pred_c = model_c.predict(X_test_c)
            acc_c = accuracy_score(y_test_c, y_pred_c)
            channel_acc.append(acc_c)

        ch_df = pd.DataFrame({"Channel": feature_cols, "Accuracy": np.array(channel_acc)*100})
        st.dataframe(ch_df)

        plt.figure(figsize=(10, 4))
        sns.barplot(data=ch_df, x="Channel", y="Accuracy", palette="coolwarm")
        plt.xticks(rotation=45)
        plt.title("Channel-wise Classification Accuracy")
        st.pyplot(plt)

        # ==================== FEATURE IMPORTANCE ====================
        st.subheader("🔥 Feature Importance Analysis")

        if hasattr(best_model, "feature_importances_"):
            importance = best_model.feature_importances_
        elif classifier_name == "SVM":
            importance = np.mean(np.abs(best_model.coef_), axis=0)
        elif classifier_name == "KNN":
            importance = mutual_info_classif(X_train, y_train)
        elif classifier_name == "Neural Network":
            importance = mutual_info_classif(X_train, y_train)
        else:
            importance = np.zeros(X.shape[1])

        imp_df = pd.DataFrame({"Feature": feature_cols, "Importance": importance})
        imp_df = imp_df.sort_values(by="Importance", ascending=False)
        st.dataframe(imp_df)

        plt.figure(figsize=(10, 4))
        sns.barplot(data=imp_df, x="Feature", y="Importance", palette="mako")
        plt.xticks(rotation=45)
        plt.title(f"Feature Importance - {classifier_name}")
        st.pyplot(plt)

        # ==================== COMBINED VISUALIZATION ====================
        st.subheader("📊 Channel Accuracy vs Feature Importance")

        merged_df = ch_df.merge(imp_df, left_on="Channel", right_on="Feature", how="inner")
        fig, ax1 = plt.subplots(figsize=(10, 4))
        sns.barplot(x="Channel", y="Accuracy", data=merged_df, color="skyblue", label="Accuracy", ax=ax1)
        ax2 = ax1.twinx()
        sns.lineplot(x="Channel", y="Importance", data=merged_df, color="red", marker="o", label="Importance", ax=ax2)
        ax1.set_ylabel("Accuracy (%)")
        ax2.set_ylabel("Importance")
        ax1.set_xticklabels(merged_df["Channel"], rotation=45)
        plt.title("Channel Accuracy vs Actual Contribution")
        st.pyplot(fig)

else:
    st.info("⬆️ Please upload your EEG feature dataset to begin.")
