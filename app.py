import streamlit as st
import pandas as pd


st.set_page_config(
    page_title="Smart EDA Dashboard",
    page_icon="📊",
    layout="wide"
)


st.title("Smart EDA Dashboard")

st.write(
    "An interactive dashboard for exploring and understanding datasets."
)


# --------------------------------------------------
# Upload Dataset
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx"]
)


if uploaded_file is not None:

    # Load dataset
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)

    else:
        df = pd.read_excel(uploaded_file)

    st.success("Dataset loaded successfully!")


    # --------------------------------------------------
    # Dataset Overview
    # --------------------------------------------------

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


    # --------------------------------------------------
    # Data Preview
    # --------------------------------------------------

    st.subheader("Data Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )