import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import io
import subprocess
import time
import sys
import os

# Set pandas display options
pd.set_option("styler.render.max_elements", 1000000)  # Increase max elements
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# API endpoint configuration
API_URL = "http://127.0.0.1:8000/predict/"


def start_backend_server():
    """Start the FastAPI backend server in a subprocess."""
    try:
        # Ensure the backend directory exists
        backend_path = os.path.join(os.path.dirname(__file__), "backend")
        if not os.path.exists(backend_path):
            st.error("Backend directory not found!")
            return None

        # Start the backend server
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "backend.main:app",
                "--host", "127.0.0.1", "--port", "8000"],
            cwd=backend_path,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        # Allow the server some time to start
        time.sleep(3)

        return process
    except Exception as e:
        st.error(f"Failed to start the backend server: {str(e)}")
        return None


def validate_csv(file):
    """Validate the uploaded CSV file"""
    try:
        df = pd.read_csv(file)
        if df.empty:
            return False, "The uploaded file is empty"
        return True, df
    except Exception as e:
        return False, f"Error reading CSV file: {str(e)}"


def show_analysis(df):
    """Show analysis of the uploaded data"""
    # Column for plots
    st.subheader('Select Column for Analysis')
    column = st.selectbox('Choose a column:',
                          df.select_dtypes(include=['int64', 'float64']).columns)

    # Distribution Plot
    st.subheader(f'{column} Distribution')
    hist_fig = px.histogram(df, x=column, title=f'{column} Distribution')
    st.plotly_chart(hist_fig, use_container_width=True)

    # Box Plot
    st.subheader(f'{column} Box Plot')
    box_fig = px.box(df, y=column, title=f'{column} Box Plot')
    st.plotly_chart(box_fig, use_container_width=True)

    # Correlation Heatmap
    st.subheader('Correlation Heatmap')
    correlation_matrix = df.select_dtypes(include=['int64', 'float64']).corr()
    heatmap_fig = go.Figure(
        data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale="Blues"
        )
    )
    heatmap_fig.update_layout(title="Correlation Heatmap")
    st.plotly_chart(heatmap_fig, use_container_width=True)

    # Scatter Plot for Feature Relationships
    st.subheader('Feature Relationships')
    col1, col2 = st.columns(2)
    with col1:
        feature_x = st.selectbox(
            "Select Feature X:",
            df.select_dtypes(include=['int64', 'float64']).columns,
            key='feature_x'
        )
    with col2:
        feature_y = st.selectbox(
            "Select Feature Y:",
            df.select_dtypes(include=['int64', 'float64']).columns,
            key='feature_y'
        )

    scatter_fig = px.scatter(
        df,
        x=feature_x,
        y=feature_y,
        title=f'Relationship between {feature_x} and {feature_y}',
        trendline="ols"
    )
    st.plotly_chart(scatter_fig, use_container_width=True)


def show_predictions(df, uploaded_file):
    """Show predictions interface"""
    if st.button("Get Predictions", type="primary"):
        with st.spinner("Making predictions..."):
            try:
                uploaded_file.seek(0)
                files = {"file": uploaded_file}
                response = requests.post(API_URL, files=files)

                if response.status_code == 200:
                    results = response.json()
                    predictions = [pred["prediction"]
                                   for pred in results["predictions"]]

                    df["Predicted_Income"] = predictions

                    st.subheader("📈 Prediction Results")

                    # Statistics cards
                    stats = results["statistics"]
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Mean Predicted Income",
                                  f"${stats['mean']:,.2f}")
                    with col2:
                        st.metric("Minimum", f"${stats['min']:,.2f}")
                    with col3:
                        st.metric("Maximum", f"${stats['max']:,.2f}")
                    with col4:
                        st.metric("Standard Deviation",
                                  f"${stats['std']:,.2f}")

                    # Results tabs
                    tab1, tab2 = st.tabs(
                        ["📊 Distribution", "📋 Detailed Results"])

                    with tab1:
                        fig = px.histogram(
                            df,
                            x="Predicted_Income",
                            title="Distribution of Predicted Income",
                            labels={
                                "Predicted_Income": "Predicted Income ($)"},
                            nbins=30
                        )
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)

                    with tab2:
                        # Create a container with fixed height for scrolling
                        st.write("Scroll through all results:")
                        container = st.container()
                        with container:
                            # Format the Predicted_Income column
                            df_display = df.copy()
                            df_display['Predicted_Income'] = df_display['Predicted_Income'].apply(
                                lambda x: f"${x:,.2f}")

                            # Display full dataframe with scrolling
                            st.dataframe(
                                df_display,
                                use_container_width=True,
                                height=500  # Set fixed height for scrolling
                            )

                        # Download full results
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download Complete Results as CSV",
                            data=csv,
                            file_name="predictions_results.csv",
                            mime="text/csv"
                        )

                else:
                    st.error(f"Error from API: {response.status_code}")
                    st.error(response.text)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")


def main():

    try:
        # Page config
        st.set_page_config(
            page_title="Census Demographics Analysis",
            page_icon="📊",
            layout="wide"
        )

        # Title and description
        st.title("📊 Census Demographics Analysis")
        st.markdown("""
        This application analyzes and predicts income levels based on census demographic data.
        Upload your CSV file containing census features to get started.
        """)

        # File uploader
        uploaded_file = st.file_uploader(
            "Upload your census data (CSV file)", type=["csv"])

        if uploaded_file:
            is_valid, result = validate_csv(uploaded_file)

            if is_valid:
                df = result

                # Display data overview
                st.subheader("📋 Data Overview")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Number of Records", len(df))
                with col2:
                    st.metric("Number of Features", len(df.columns))
                with col3:
                    st.metric("Missing Values", df.isna().sum().sum())

                # Show sample of uploaded data
                st.subheader("🔍 Sample of Uploaded Data")
                st.dataframe(df.head(), use_container_width=True)

                # Summary Statistics
                st.subheader('Summary Statistics')
                st.dataframe(df.describe(), use_container_width=True)

                # Navigation tabs
                tab1, tab2 = st.tabs(["🔮 Predictions", "📊 Analysis"])

                with tab1:
                    show_predictions(df, uploaded_file)

                with tab2:
                    show_analysis(df)

            else:
                st.error(result)

        else:
            st.info("""
            1. Prepare your CSV file with census demographic features
            2. Upload the file using the button above
            3. Use the tabs to:
               - Get income predictions
               - Analyze your data with interactive visualizations
            
            Make sure your CSV file contains all required features in the correct format.
            """)

        # Footer
        st.markdown("---")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    finally:
        # Cleanup: terminate the backend server when the app is closed
        # if backend_process is not None:
        #     backend_process.terminate()
        pass


if __name__ == "__main__":
    main()
