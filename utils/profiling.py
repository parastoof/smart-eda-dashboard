import pandas as pd


def get_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    quality = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing Values": df.isna().sum().values,
        "Missing %": (
            df.isna().mean().values * 100
        ).round(2),
        "Unique Values": df.nunique().values,
    })

    return quality