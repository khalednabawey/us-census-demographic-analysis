import pandas as pd
from sklearn.ensemble import IsolationForest
import numpy as np
import pickle as pkl
from sklearn.decomposition import PCA


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the dataset by handling missing values, duplicates, and outliers."""
    # Drop duplicates
    df = df.drop_duplicates()

    # Fill missing values
    df.dropna(inplace=True)

    # Get only numeric columns for outlier detection
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    # Initialize the Isolation Forest model
    iso_forest = IsolationForest(
        n_estimators=100, contamination=0.1, random_state=42)

    # Fit the model on numeric columns only
    iso_forest.fit(df[numeric_cols])

    outliers = iso_forest.predict(df[numeric_cols])
    df.loc[:, 'outlier'] = outliers
    df = df[df['outlier'] == 1]
    df.drop(columns="outlier", inplace=True)

    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the dataset by encoding categorical variables, scaling features, and applying PCA."""
    # Drop columns that weren't in the training data
    columns_to_drop = ['TractId', 'State', 'County', 'Income']
    df = df.drop(columns=columns_to_drop, errors='ignore')

    # Get numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    # Apply log transformation only to numeric columns
    df[numeric_cols] = df[numeric_cols].apply(np.log1p)

    # Load and apply the feature scaler
    with open("./backend/models/feature_scaler.pkl", 'rb') as f:
        scaler = pkl.load(f)

    # Ensure columns match the scaler's expected features
    expected_features = scaler.feature_names_in_
    df = df[expected_features]

    scaled_data = pd.DataFrame(
        scaler.transform(df),
        columns=df.columns,
        index=df.index
    )

    # Load and apply PCA transformation
    with open("./backend/models/pca.pkl", 'rb') as f:
        pca = pkl.load(f)

    pca_data = pd.DataFrame(
        pca.transform(scaled_data),
        columns=[f'PC{i+1}' for i in range(pca.n_components_)],
        index=df.index
    )

    return pca_data


def prepare_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for prediction by applying necessary transformations."""
    try:
        # Make a copy to avoid modifying the original dataframe
        df = df.copy()

        # Drop any rows with missing values
        df = df.dropna()

        # Process the data
        processed_data = preprocess_data(df)

        if processed_data.empty:
            raise ValueError("No valid data after preprocessing")

        print(
            f"Input shape: {df.shape}, Processed shape: {processed_data.shape}")

        return processed_data

    except Exception as e:
        raise ValueError(f"Error in data preparation: {str(e)}")
