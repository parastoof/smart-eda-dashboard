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


def detect_outliers(series: pd.Series) -> dict:
    series = series.dropna()

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = series[
        (series < lower_bound) |
        (series > upper_bound)
    ]

    return {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outlier_count": len(outliers),
        "outlier_percentage": (
            len(outliers) / len(series) * 100
        )
    }