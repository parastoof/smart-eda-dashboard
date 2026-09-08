from utils.profiling import get_data_quality, detect_outliers
import plotly.express as px
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Smart EDA Dashboard", page_icon="📊", layout="wide")
st.title("Smart EDA Dashboard")
st.write("An interactive dashboard for exploring and understanding datasets.")

# Upload Dataset
uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Load dataset
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    st.success("Dataset loaded successfully!")

    # Dataset Overview
    st.header("Dataset Overview")

    # Calculate metrics
    rows = df.shape[0]
    columns = df.shape[1]
    missing_values = df.isna().sum().sum()
    duplicates = df.duplicated().sum()

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Rows", f"{rows:,}")
    col2.metric("Columns", columns)
    col3.metric("Missing Values", f"{missing_values:,}")
    col4.metric("Duplicates", f"{duplicates:,}")

    # Data Preview
    st.subheader("Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Data Quality
    st.header("Data Quality")
    quality_df = get_data_quality(df)
    st.dataframe(quality_df, use_container_width=True, hide_index=True)

    # Numerical Analysis
    st.header("Numerical Analysis")
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    if numeric_columns:
        selected_feature = st.selectbox("Select a numerical feature", numeric_columns)
        series = df[selected_feature].dropna()

        # Statistics
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Mean", f"{series.mean():.2f}")
        col2.metric("Median", f"{series.median():.2f}")
        col3.metric("Std", f"{series.std():.2f}")
        col4.metric("Min", f"{series.min():.2f}")
        col5.metric("Max", f"{series.max():.2f}")

        # Histogram
        fig_hist = px.histogram(df, x=selected_feature, 
                                title=f"Distribution of {selected_feature}", 
                                marginal="box")
        st.plotly_chart(fig_hist, use_container_width=True)

    else:
        st.info("No numerical features found.")


    # --------------------------------------------------
    # Categorical Analysis
    # --------------------------------------------------

    st.header("Categorical Analysis")

    categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if categorical_columns:
        selected_feature = st.selectbox("Select a categorical feature", categorical_columns, 
                                        key="categorical_feature")
        series = df[selected_feature]

        # Statistics
        unique_values = series.nunique()
        missing_values = series.isna().sum()
        if not series.dropna().empty:
            most_frequent = series.mode().iloc[0]
        else:
            most_frequent = "N/A"

        col1, col2, col3 = st.columns(3)
        col1.metric("Unique Categories", f"{unique_values:,}")
        col2.metric("Missing Values", f"{missing_values:,}")
        col3.metric("Most Frequent", str(most_frequent))

        # Category distribution
        value_counts = (series.value_counts(dropna=False).reset_index())
        value_counts.columns = ["Category", "Count"]

        # Bar chart
        fig = px.bar(value_counts.head(20), x="Category", y="Count", 
                     title=f"Distribution of {selected_feature}")
        st.plotly_chart(fig, use_container_width=True)

        # Table
        st.subheader("Category Distribution")
        st.dataframe(value_counts, use_container_width=True, hide_index=True)

    else:
        st.info("No categorical features found.")

    # Outlier Analysis
    st.header("Outlier Analysis")
    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    if numeric_columns:
        selected_feature = st.selectbox("Select a numerical feature", numeric_columns, 
                                        key="outlier_feature")
        result = detect_outliers(df[selected_feature])

        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Q1", f"{result['q1']:.2f}")
        col2.metric("Q3", f"{result['q3']:.2f}")
        col3.metric("Outliers", f"{result['outlier_count']:,}")
        col4.metric("Outlier %", f"{result['outlier_percentage']:.2f}%")

        # Bounds
        st.subheader("IQR Boundaries")
        col1, col2 = st.columns(2)
        col1.metric("Lower Bound", f"{result['lower_bound']:.2f}")
        col2.metric("Upper Bound", f"{result['upper_bound']:.2f}")

        # Box plot
        fig = px.box(df, y=selected_feature, points="outliers", 
                     title=f"Outlier Analysis — {selected_feature}")
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("No numerical features found.")