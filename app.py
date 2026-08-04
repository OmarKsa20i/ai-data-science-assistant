import os

import pandas as pd
import streamlit as st

import utils.charts as charts
import utils.data_utils as du
import utils.model_utils as mu
import utils.preprocessing as prep


st.set_page_config(
    page_title="AI Data Science Assistant", page_icon="🤖", layout="wide"
)
st.title("🤖 AI Data Science Assistant")
st.write(
    "Upload your dataset, understand it, prepare it, train a model, evaluate it, "
    "and make predictions."
)

# 1. Data Understanding
st.header("1. Data Understanding")
uploaded_file = st.file_uploader("📂 Upload your dataset", type=["csv", "xlsx"])
if uploaded_file is None:
    st.info("Upload a dataset to start.")
    st.stop()

if (
    "df" not in st.session_state
    or st.session_state.get("uploaded_file_name") != uploaded_file.name
):
    st.session_state.df = du.load_dataset(uploaded_file)
    st.session_state.uploaded_file_name = uploaded_file.name

df = st.session_state.df
st.success("✅ File uploaded successfully!")
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

st.subheader("Dataset Dashboard")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📄 Rows", df.shape[0])
with col2:
    st.metric("📋 Columns", df.shape[1])
with col3:
    st.metric("❌ Missing Values", df.isnull().sum().sum())
with col4:
    st.metric("🔁 Duplicates", df.duplicated().sum())

st.subheader("💡 Dataset Insights")
missing_values = df.isnull().sum().sum()
duplicate_rows = df.duplicated().sum()
if missing_values == 0:
    st.success("✅ No missing values found.")
else:
    st.warning(f"⚠️ Dataset contains {missing_values} missing values.")
if duplicate_rows == 0:
    st.success("✅ No duplicate rows found.")
else:
    st.warning(f"⚠️ Dataset contains {duplicate_rows} duplicate rows.")

numeric_columns = df.select_dtypes(include="number").columns
st.info(f"📊 Numeric Columns: {len(numeric_columns)}")

if len(numeric_columns) > 0:
    st.subheader("📈 Statistics")
    stats_column = st.selectbox("📊 Select a column for statistics", numeric_columns)
    values = df[stats_column]
    mode = values.mode()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Mean", round(values.mean(), 2))
        st.metric("📉 Median", round(values.median(), 2))
        st.metric("🔢 Mode", round(mode.iloc[0], 2) if not mode.empty else "N/A")
    with col2:
        st.metric("⬇️ Minimum", round(values.min(), 2))
        st.metric("⬆️ Maximum", round(values.max(), 2))
    with col3:
        st.metric("📍 Standard Deviation", round(values.std(), 2))
        st.metric("📌 Variance", round(values.var(), 2))

    st.subheader("📊 Data Visualization")
    selected_column = st.selectbox("📊 Select a numeric column", numeric_columns)
    chart_type = st.selectbox("📈 Select Chart Type", list(charts.CHART_RECOMMENDATIONS))
    st.subheader("💡 Smart Chart Recommendation")
    st.info(
        f"💡 Recommendation: {chart_type}\n\n"
        f"Reason: {charts.CHART_RECOMMENDATIONS[chart_type]}"
    )
    st.plotly_chart(
        charts.create_chart(df, selected_column, chart_type), use_container_width=True
    )

    st.subheader("📊 Correlation Analysis")
    st.plotly_chart(
        charts.create_correlation_heatmap(df, numeric_columns),
        use_container_width=True,
    )

# 2. Data Preparation
st.header("2. Data Preparation")
st.subheader("🧹 Data Cleaning")

if st.button("🗑️ Remove Duplicates"):
    rows_before = st.session_state.df.shape[0]
    st.session_state.df = du.remove_duplicates(st.session_state.df)
    df = st.session_state.df
    st.success(f"Removed {rows_before - df.shape[0]} duplicate rows.")
    st.rerun()

if st.button("❌ Remove Missing Values"):
    rows_before = st.session_state.df.shape[0]
    st.session_state.df = du.remove_missing_values(st.session_state.df)
    df = st.session_state.df
    st.success(f"Removed {rows_before - df.shape[0]} rows with missing values.")
    st.rerun()

st.download_button(
    label="💾 Download Cleaned Dataset",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="cleaned_dataset.csv",
    mime="text/csv",
)

# 3. Modeling
st.header("3. Modeling")
st.subheader("🤖 Machine Learning Task Detection")
target_column = st.selectbox("🎯 Select Target Column", df.columns)
is_classification = df[target_column].dtype == "object"
if is_classification:
    st.success("🤖 Classification Problem Detected")
    st.info(
        "💡 Recommended Model: Random Forest Classifier\n\n"
        "Reason: This model performs well on most classification problems."
    )
else:
    st.success("🤖 Regression Problem Detected")
    st.info(
        "💡 Recommended Model: Linear Regression\n\n"
        "Reason: This model is simple, fast, and suitable for predicting continuous numeric values."
    )

st.subheader("📦 Data Preparation for Modeling")
try:
    X_train, X_test, y_train, y_test = prep.prepare_train_test_data(df, target_column)
except ValueError as error:
    st.error(str(error))
    st.stop()

X = pd.concat([X_train, X_test])
st.success("✅ Dataset split successfully!")
st.write(f"Training Samples: {len(X_train)}")
st.write(f"Testing Samples: {len(X_test)}")

if is_classification:
    model = mu.train_random_forest(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = mu.get_accuracy(y_test, predictions)
    st.success("✅ Random Forest model trained successfully!")
else:
    model = mu.train_linear_regression(X_train, y_train)
    predictions = model.predict(X_test)
    tree_model = mu.train_decision_tree(X_train, y_train)
    tree_predictions = tree_model.predict(X_test)
    r2, mae = mu.get_regression_metrics(y_test, predictions)
    tree_r2, _ = mu.get_regression_metrics(y_test, tree_predictions)
    st.success("✅ Linear Regression model trained successfully!")
# ============================
# 4. Evaluation
# ============================

st.header("4. Evaluation")

if is_classification:

    # Display classification accuracy
    st.metric("Accuracy", f"{accuracy:.2%}")

else:

    # Display regression predictions
    st.subheader("Model Prediction")

    results = pd.DataFrame({
        "Actual Value": y_test,
        "Predicted Value": predictions
    })

    st.dataframe(results.head())

    # Display evaluation metrics
    col1, col2 = st.columns(2)

    with col1:
        st.metric("R² Score", round(r2, 3))

    with col2:
        st.metric("MAE", round(mae, 2))

    # Compare models
    st.subheader("🏆 Model Comparison")

    comparison_df = pd.DataFrame({
        "Model": [
            "Linear Regression",
            "Decision Tree"
        ],
        "R² Score": [
            round(r2, 3),
            round(tree_r2, 3)
        ]
    })

    st.dataframe(comparison_df)

# 5. Deployment
st.header("5. Deployment")
st.subheader("💾 Save Model")
os.makedirs("models", exist_ok=True)
if st.button("💾 Save Model"):
    mu.save_model(model, "models/model.joblib")
    st.success("✅ Model saved successfully!")

st.subheader("📂 Load Model")
if st.button("📂 Load Model"):
    if os.path.exists("models/model.joblib"):
        mu.load_model("models/model.joblib")
        st.success("✅ Model loaded successfully!")
    else:
        st.error("❌ No saved model found.")

st.subheader("🔮 Predict New Data")
input_data = {
    column: st.number_input(f"Enter {column}", value=0.0) for column in X.columns
}
if st.button("🔮 Predict"):
    if os.path.exists("models/model.joblib"):
        loaded_model = mu.load_model("models/model.joblib")
        prediction = loaded_model.predict(pd.DataFrame([input_data]))
        st.success(f"Prediction: {prediction[0]}")
    else:
        st.error("❌ No saved model found.")
