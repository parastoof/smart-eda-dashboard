from utils.profiling import get_data_quality, detect_outliers, detect_target_type
from utils.visualization import create_histogram, create_bar_chart, create_boxplot, create_correlation_heatmap, create_target_classification_chart, create_target_regression_chart
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
    with st.expander("Dataset Overview", expanded=True):
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
    with st.expander("Data Quality"):
        quality_df = get_data_quality(df)
        st.dataframe(quality_df, use_container_width=True, hide_index=True)

    # Numerical Analysis
    st.header("Numerical Analysis")
    with st.expander("Numerical Analysis"):
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
            fig_hist = create_histogram(df, selected_feature)
            st.plotly_chart(fig_hist, use_container_width=True)

        else:
            st.info("No numerical features found.")


    # Categorical Analysis

    st.header("Categorical Analysis")
    with st.expander("Categorical Analysis"):
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
            fig = create_bar_chart(value_counts, selected_feature)
            st.plotly_chart(fig, use_container_width=True)

            # Table
            st.subheader("Category Distribution")
            st.dataframe(value_counts, use_container_width=True, hide_index=True)

        else:
            st.info("No categorical features found.")

    # Outlier Analysis
    st.header("Outlier Analysis")
    with st.expander("Outlier Analysis"):
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
            fig = create_boxplot(df, selected_feature)
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("No numerical features found.")

    # Correlation Analysis
    st.header("Correlation Analysis")
    with st.expander("Correlation Analysis"):
        numeric_df = df.select_dtypes(include="number")
        if numeric_df.shape[1] > 1:
            correlation_matrix = numeric_df.corr()
            fig = create_correlation_heatmap(correlation_matrix)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough numerical features for correlation analysis.")
        
    # Target Analysis
    st.header("Target Analysis")
    with st.expander("Target Analysis"):
        target_column = st.selectbox("Select the target variable", df.columns, 
                                    key="target_variable")
        target = df[target_column]
        target_type = detect_target_type(target)
        st.write(f"Detected problem type: **{target_type}**")
        
        if target_type == "Classification":
            value_counts = (target.value_counts(dropna=False).reset_index())
            value_counts.columns = ["Class", "Count"]

            # Metrics (horizontal table)
            metrics_df = pd.DataFrame(
                [
                    {
                        "Number of Classes": f"{target.nunique():,}",
                        "Missing Values": f"{target.isna().sum():,}",
                    }
                ]
            )
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
            # Distribution
            fig = create_target_classification_chart(value_counts, target_column)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(value_counts, use_container_width=True, hide_index=True)
        
        else:
            target_clean = target.dropna()
            summary_df = pd.DataFrame(
                [{
                    "Mean": f"{target_clean.mean():.2f}",
                    "Median": f"{target_clean.median():.2f}",
                    "Std": f"{target_clean.std():.2f}",
                    "Missing Values": f"{target.isna().sum():,}",
                }]
            )
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            fig = create_target_regression_chart(target_clean, target_column)
            st.plotly_chart(fig, use_container_width=True)