"""Chart creation helpers used by the Streamlit interface."""

import plotly.express as px


CHART_RECOMMENDATIONS = {
    "Histogram": "Histogram is the best choice for visualizing the distribution of continuous numeric data.",
    "Bar Chart": "Bar Chart is useful for comparing values across different categories.",
    "Line Chart": "Line Chart is ideal for showing trends over time.",
    "Box Plot": "Box Plot helps detect outliers and understand data distribution.",
    "Scatter Plot": "Scatter Plot is useful for exploring relationships between two numeric variables.",
}


def create_chart(df, column, chart_type):
    """Create the Plotly figure for the selected chart type."""
    if chart_type == "Histogram":
        return px.histogram(df, x=column, title=f"Distribution of {column}")
    if chart_type == "Bar Chart":
        return px.bar(df, x=column, title=f"Bar Chart of {column}")
    if chart_type == "Line Chart":
        return px.line(df, y=column, title=f"Line Chart of {column}")
    if chart_type == "Box Plot":
        return px.box(df, y=column, title=f"Box Plot of {column}")
    return px.scatter(df, x=df.index, y=column, title=f"Scatter Plot of {column}")


def create_correlation_heatmap(df, numeric_columns):
    """Create the correlation heatmap for numeric columns."""
    return px.imshow(
        df[numeric_columns].corr(),
        text_auto=True,
        color_continuous_scale="RdBu_r",
        title="Correlation Heatmap",
    )
