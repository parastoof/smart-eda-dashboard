from utils.profiling import get_data_quality, detect_outliers, detect_target_type
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

    # Display metrics in a table
    metrics_df = pd.DataFrame(
        {
            "Metric": ["Value"],
            "Rows": f"{rows:,}",
            "Columns": columns,
            "Missing Values": f"{missing_values:,}",
            "Duplicates": f"{duplicates:,}",
        }
    )
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

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

        # Statistics in table format
        stats_df = pd.DataFrame(
            {
                "Statistic": ["Value" ],
                "Mean": f"{series.mean():.2f}",
                "Median": f"{series.median():.2f}",
                "Std": f"{series.std():.2f}",
                "Min": f"{series.min():.2f}",
                "Max": f"{series.max():.2f}"
            }
        )
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

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

        summary_df = pd.DataFrame(
            [
                {
                    "Metric": "Category Summary",
                    "Unique Categories": f"{unique_values:,}",
                    "Missing Values": f"{missing_values:,}",
                    "Most Frequent": str(most_frequent),
                }
            ]
        )
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

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
        outlier_stats_df = pd.DataFrame(
            {
                "Metric": ["Value"],
                "Q1": f"{result['q1']:.2f}",
                "Q3": f"{result['q3']:.2f}",
                "Lower Bound": f"{result['lower_bound']:.2f}",
                "Upper Bound": f"{result['upper_bound']:.2f}",
                "Outliers": f"{result['outlier_count']:,}",
                "Outlier %": f"{result['outlier_percentage']:.2f}%"
            }
        )
        st.dataframe(outlier_stats_df, use_container_width=True, hide_index=True)

        # Box plot
        fig = px.box(df, x=selected_feature, points="outliers", 
                     title=f"Outlier Analysis — {selected_feature}")
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("No numerical features found.")

    # Correlation Analysis
    st.header("Correlation Analysis")
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] > 1:
        correlation_matrix = numeric_df.corr()
        fig = px.imshow(correlation_matrix, text_auto=".2f",
                        aspect="auto", 
                        color_continuous_scale="RdBu_r", 
                        title="Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough numerical features for correlation analysis.")
    
    # Target Analysis
    st.header("Target Analysis")
    target_column = st.selectbox("Select the target variable", df.columns, 
                                 key="target_variable")
    target = df[target_column]
    target_type = detect_target_type(target)
    st.write(f"Detected problem type: **{target_type}**")
    
    if target_type == "Classification":
        value_counts = (target.value_counts(dropna=False).reset_index())
        value_counts.columns = ["Class", "Count"]

        # Metrics
        col1, col2 = st.columns(2)
        col1.metric("Number of Classes", target.nunique())
        col2.metric("Missing Values", target.isna().sum())
    
        # Distribution
        fig = px.bar(value_counts, x="Class", y="Count", title=f"Target Distribution — {target_column}")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(value_counts, use_container_width=True, hide_index=True)
    
    else:
        target_clean = target.dropna()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mean", f"{target_clean.mean():.2f}")
        col2.metric("Median", f"{target_clean.median():.2f}")
        col3.metric("Std", f"{target_clean.std():.2f}")
        col4.metric("Missing Values", target.isna().sum())

        fig = px.histogram(target_clean, 
                           x=target_column, 
                           marginal="box", 
                           title=f"Target Distribution — {target_column}")
    
        st.plotly_chart(fig, use_container_width=True)