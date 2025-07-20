import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve
)
import numpy as np
import pickle
import io

st.set_page_config(page_title="ML App", layout="wide")

# Caching data loading
@st.cache_data
def load_example_data(name):
    return sns.load_dataset(name)

@st.cache_data
def load_uploaded_data(uploaded_file):
    if uploaded_file.type == "text/csv":
        return pd.read_csv(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        return pd.read_excel(uploaded_file)
    elif uploaded_file.type == "text/tab-separated-values":
        return pd.read_csv(uploaded_file, sep='\t')
    else:
        return pd.read_csv(uploaded_file)

# Caching model training
@st.cache_resource
def train_model(_model, X_train, y_train):
    _model.fit(X_train, y_train)
    return _model

def preprocess_data(X, y):
    # Encode categorical features before imputation
    X_encoded = X.copy()
    encoders = {}
    for col in X_encoded.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        encoders[col] = le

    # Impute missing values
    imp = IterativeImputer()
    X_imputed = pd.DataFrame(imp.fit_transform(X_encoded), columns=X_encoded.columns)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    return X_scaled, y, encoders, scaler, imp

def get_problem_type(y, user_choice):
    if user_choice == "Regression":
        return "Regression"
    elif user_choice == "Classification":
        return "Classification"
    # Fallback: auto-detect
    if pd.api.types.is_numeric_dtype(y):
        if y.nunique() > 10:
            return "Regression"
        else:
            return "Classification"
    else:
        return "Classification"

def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}

def classification_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}

def main():
    st.title("🔬 Machine Learning Playground")
    st.write("Welcome! This app lets you train and evaluate ML models on your data or example datasets. Upload your data or use a sample, select features and target, and compare models easily.")

    # Data source selection
    data_source = st.sidebar.radio("Data Source", ["Upload", "Example"])
    data = None

    if data_source == "Upload":
        uploaded_file = st.sidebar.file_uploader("Upload your dataset", type=['csv', 'xlsx', 'tsv'])
        if uploaded_file:
            data = load_uploaded_data(uploaded_file)
    else:
        dataset_name = st.sidebar.selectbox("Select example dataset", ["titanic", "tips", "iris"])
        data = load_example_data(dataset_name)

    if data is not None and not data.empty:
        st.subheader("Data Overview")
        st.write("**Head:**")
        st.write(data.head())
        st.write(f"**Shape:** {data.shape}")
        st.write("**Description:**")
        st.write(data.describe(include='all'))
        st.write("**Info:**")
        buffer = io.StringIO()
        data.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
        st.write("**Columns:**", data.columns.tolist())

        # Feature and target selection
        st.subheader("Select Features and Target")
        features = st.multiselect("Select feature columns", data.columns.tolist())
        target = st.selectbox("Select target column", data.columns.tolist())

        # Ask user for problem type
        user_problem_type = st.radio("What type of problem is this?", ["Auto-detect", "Regression", "Classification"])
        if features and target:
            X = data[features]
            y = data[target]
            problem_type = get_problem_type(y, user_problem_type if user_problem_type != "Auto-detect" else None)
            st.info(f"Detected problem type: **{problem_type}**")

            # Train-test split
            test_size = st.slider("Test split size", 0.1, 0.5, 0.2, step=0.05)

            # Model selection
            if problem_type == "Regression":
                model_options = {
                    "Linear Regression": LinearRegression(),
                    "Decision Tree": DecisionTreeRegressor(),
                    "Random Forest": RandomForestRegressor(),
                    "SVM": SVR()
                }
            else:
                model_options = {
                    "Logistic Regression": LogisticRegression(max_iter=1000),
                    "Decision Tree": DecisionTreeClassifier(),
                    "Random Forest": RandomForestClassifier(),
                    "SVM": SVC(probability=True)
                }
            selected_models = st.sidebar.multiselect("Select models to train", list(model_options.keys()), default=list(model_options.keys()))

            # Run analysis button
            if st.button("Run Analysis"):
                # Preprocess
                X_processed, y_processed, encoders, scaler, imputer = preprocess_data(X, y)
                X_train, X_test, y_train, y_test = train_test_split(X_processed, y_processed, test_size=test_size, random_state=42)

                results = {}
                best_score = None
                best_model_name = None
                best_model = None

                st.subheader("Model Evaluation")
                for name in selected_models:
                    model = model_options[name]
                    trained_model = train_model(model, X_train, y_train)
                    y_pred = trained_model.predict(X_test)
                    if problem_type == "Regression":
                        metrics = regression_metrics(y_test, y_pred)
                        score = metrics["R2"]
                    else:
                        metrics = classification_metrics(y_test, y_pred)
                        score = metrics["F1"]
                    results[name] = metrics

                    # Highlight best
                    if best_score is None or score > best_score:
                        best_score = score
                        best_model_name = name
                        best_model = trained_model

                # Show results
                st.write(pd.DataFrame(results).T.style.highlight_max(axis=0, color='lightgreen'))
                st.success(f"Best model: **{best_model_name}**")

                # Extra plots
                if problem_type == "Regression":
                    st.write("**Scatter plot of predictions vs actual:**")
                    st.scatter_chart(pd.DataFrame({"Actual": y_test, "Predicted": y_pred}))
                else:
                    st.write("**Confusion Matrix:**")
                    cm = confusion_matrix(y_test, best_model.predict(X_test))
                    st.write(pd.DataFrame(cm))

                    # ROC Curve
                    if hasattr(best_model, "predict_proba"):
                        y_score = best_model.predict_proba(X_test)[:, 1] if len(np.unique(y_test)) == 2 else None
                        if y_score is not None:
                            fpr, tpr, _ = roc_curve(y_test, y_score)
                            st.line_chart(pd.DataFrame({"FPR": fpr, "TPR": tpr}))
                            st.write(f"AUROC: {roc_auc_score(y_test, y_score):.3f}")

                # Download model
                if st.checkbox("Download best model as pickle"):
                    model_bytes = pickle.dumps(best_model)
                    st.download_button("Download Model", model_bytes, file_name="best_model.pkl")

                # Prediction
                if st.checkbox("Make prediction with best model"):
                    st.write("Provide input for prediction:")
                    input_data = {}
                    for i, col in enumerate(features):
                        dtype = X[col].dtype
                        if dtype == 'object' or dtype.name == 'category':
                            options = list(data[col].astype(str).unique())
                            input_data[col] = st.selectbox(f"{col}", options)
                        else:
                            min_val = float(X[col].min())
                            max_val = float(X[col].max())
                            mean_val = float(X[col].mean())
                            input_data[col] = st.slider(f"{col}", min_val, max_val, mean_val)
                    # Prepare input
                    input_df = pd.DataFrame([input_data])
                    # Encode
                    for col, le in encoders.items():
                        input_df[col] = le.transform(input_df[col].astype(str))
                    # Impute (not needed for single row, but for consistency)
                    input_df = pd.DataFrame(imputer.transform(input_df), columns=input_df.columns)
                    # Scale
                    input_scaled = scaler.transform(input_df)
                    pred = best_model.predict(input_scaled)
                    st.write(f"Prediction: **{pred[0]}**")

if __name__ == "__main__":
    main()