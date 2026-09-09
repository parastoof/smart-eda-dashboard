import plotly.express as px

EDA_COLORS = ["#4C78A8", "#F58518",
              "#54A24B", "#E45756",
              "#72B7B2", "#B279A2",
              "#FF9DA6", "#A0D12D"]

def create_histogram(df, feature):
    fig = px.histogram(df, x=feature, 
                       title=f"Distribution of {feature}",
                       marginal="box",
                       color_discrete_sequence=EDA_COLORS)
    return fig

def create_bar_chart(value_counts, feature):
    fig = px.bar(value_counts.head(20), x="Category", y="Count", 
                title=f"Distribution of {feature}",
                color_discrete_sequence=EDA_COLORS)
    return fig

def create_boxplot(df, feature):
    fig = px.box(df, x=feature, points="outliers", 
                 title=f"Outlier Analysis — {feature}",
                 color_discrete_sequence=EDA_COLORS)
    return fig

def create_correlation_heatmap(correlation_matrix):
    fig = px.imshow(correlation_matrix, text_auto=".2f",
                    aspect="auto", 
                    color_continuous_scale="RdBu_r", 
                    title="Correlation Matrix",
                    zmin=-1, zmax=1,)
    return fig

def create_target_classification_chart(value_counts, target_column):
    fig = px.bar(value_counts,
        x="Class",
        y="Count",
        title=f"Target Distribution — {target_column}",
        color="Class",
        color_discrete_sequence=EDA_COLORS,
    )

    return fig


def create_target_regression_chart(target, target_column):
    fig = px.histogram(target,
        x=target_column,
        marginal="box",
        title=f"Target Distribution — {target_column}",
        color_discrete_sequence=EDA_COLORS,
    )

    return fig

            