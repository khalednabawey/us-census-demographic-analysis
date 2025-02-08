import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go


# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])


@st.cache_data
def load_data(uploaded_file=None):
    try:
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_csv('./data/acs2017_census_tract_data.csv')
        return data
    except Exception as e:
        raise ValueError(f"Error loading data: {e}")


def clean_data(df):
    # Drop rows with missing values
    df.dropna(inplace=True)

    # Initialize the Isolation Forest model
    iso_forest = IsolationForest(
        n_estimators=100, contamination=0.1, random_state=42)

    # Fit the model
    iso_forest.fit(df.select_dtypes("number"))

    outliers = iso_forest.predict(df.select_dtypes("number"))
    df.loc[:, 'outlier'] = outliers
    df = df[df['outlier'] == 1]
    df.drop(columns="outlier", inplace=True)

    return df


# Load and clean data
data = clean_data(load_data(uploaded_file))

# Title of the dashboard
st.title('US Census Demographic Analysis Dashboard')

# Display the dataset
st.subheader('US Census Dataset')
st.write(data.head())

# Column for plots
st.subheader('Select Column for Analysis')
column = st.selectbox('Choose a column:',
                      data.select_dtypes("number").columns[1:])

# Distribution Plot
st.subheader(f'{column} Distribution')
hist_fig = px.histogram(data, x=column, title=f'{column} Distribution')
st.plotly_chart(hist_fig)

# Box Plot
st.subheader(f'{column} Box Plot')
box_fig = px.box(data, y=column, title=f'{column} Box Plot')
st.plotly_chart(box_fig)

# Correlation Heatmap
st.subheader('Correlation Heatmap')
correlation_matrix = data.select_dtypes("number").corr()
heatmap_fig = go.Figure(
    data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale="Blues"
    )
)
heatmap_fig.update_layout(title="Correlation Heatmap")
st.plotly_chart(heatmap_fig)

# Scatter Plot for Feature Relationships
st.subheader('Feature Relationships')
feature_x = st.selectbox(
    "Select Feature X:", data.select_dtypes("number").columns[1:])
feature_y = st.selectbox(
    "Select Feature Y:", data.select_dtypes("number").columns[1:])

scatter_fig = px.scatter(
    data, x=feature_x, y=feature_y,
    title=f'Relationship between {feature_x} and {feature_y}',
    trendline="ols"
)
st.plotly_chart(scatter_fig)

# Trend Analysis
st.subheader('Trend Analysis')
trend_col = st.selectbox("Select Column for Trend Analysis:",
                         data.select_dtypes("number").columns[1:])
trend_fig = px.line(data, x=data.index, y=trend_col,
                    title=f'{trend_col} Trend Over Index')
st.plotly_chart(trend_fig)

# Pairwise Comparisons using a Dropdown
st.subheader('Pairwise Comparisons')
pair_x = st.selectbox('Select X-axis Column:',
                      data.select_dtypes("number").columns[1:], key='pair_x')
pair_y = st.selectbox('Select Y-axis Column:',
                      data.select_dtypes("number").columns[1:], key='pair_y')
pair_fig = px.scatter(
    data, x=pair_x, y=pair_y,
    # Add grouping based on the first column (e.g., State)
    color=data[data.columns[0]],
    title=f'{pair_x} vs {pair_y}',
    marginal_x="box", marginal_y="violin"
)
st.plotly_chart(pair_fig)
