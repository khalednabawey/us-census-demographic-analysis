# US Census Demographics Income Prediction

## Overview

This project is a machine learning application that predicts income levels based on US Census demographic data. It features a Neural Network model trained on census data, with a FastAPI backend and Streamlit frontend for real-time predictions and data analysis.

## Features

- 🔮 Income prediction based on demographic features
- 📊 Interactive data visualization and analysis
- ⚡ Real-time prediction capabilities
- 💾 Downloadable prediction results
- 🔄 Automated data preprocessing pipeline
- 📈 Model performance metrics visualization

## Tech Stack

- **Frontend**: Streamlit
- **Backend**: FastAPI
- **Machine Learning**: TensorFlow, scikit-learn
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly
- **Development**: Python 3.10+

## Project Structure

us-census-demographic-analysis/
├── app.py # Streamlit frontend application
├── run.py # Application orchestration script
├── backend/
│ ├── init.py
│ ├── main.py # FastAPI backend server
│ ├── data_processor.py # Data preprocessing pipeline
│ └── models/
│ ├── NN-model.h5 # Trained neural network model
│ ├── pca.pkl # PCA transformation model
│ ├── feature_scaler.pkl # Feature scaling model
│ └── target_scaler.pkl # Target scaling model
└── requirements.txt

## Data Processing Pipeline

The application implements a robust data processing pipeline:

### 1. Data Cleaning

- Removes duplicate entries
- Handles missing values
- Detects and removes outliers using Isolation Forest
- Ensures data quality and consistency

### 2. Preprocessing

- Applies log transformation to numeric features
- Performs feature scaling using pre-trained scaler
- Reduces dimensionality using PCA
- Ensures consistent feature engineering

### 3. Prediction Preparation

- Validates input data
- Applies preprocessing transformations
- Prepares data for model inference

### 4. Data Analysis (Tableau Dashboard)
You can view the US Census Demographics data interactive dashboard [here](https://public.tableau.com/views/USCensusDemographicAnalysis_17339499055870/AnalysisDashboard?:language=en-GB&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link){:target="_blank"}.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/khalednabawey/us-census-demographic-analysis.git
cd us-census-demographic-analysis
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:

```bash
python run.py
```

2. Access the application:

- Frontend: http://localhost:8501
- Backend API: http://localhost:8000

3. Using the Application:
   - Upload your CSV file containing census demographic data
   - View data analysis and visualizations
   - Get income predictions
   - Download results as CSV

## API Endpoints

### Health Check

```http
GET /health-check/
```

Returns the API and model status

### Predictions

```http
POST /predict/
```

Accepts CSV file and returns predictions

### Root

```http
GET /
```

Returns API documentation and information

## Input Data Format

The model expects a CSV file with the following features:

- Demographic information (age, education, occupation)
- Economic indicators
- Geographic data
- Social statistics

Example format:

```csv
Age,Education,Occupation,WorkHours,Industry,...
45,Bachelors,Professional,40,Technology,...
```

## Development

### Running Tests

```bash
pytest tests/
```

### Local Development

1. Start backend server:

```bash
uvicorn backend.main:app --reload
```

2. Start frontend:

```bash
streamlit run app.py
```

## Model Architecture

### Neural Network Details

- Input Layer: Matches PCA components
- Hidden Layers: Dense layers with ReLU activation
- Output Layer: Single node for income prediction
- Training: Adam optimizer with MSE loss

### Performance Metrics

- Training Accuracy
- Validation R² Score
- Mean Squared Error

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- US Census Bureau for the dataset
- Kaggle for hosting the competition
- All contributors and maintainers

## Contact

Khaled Nabawi - khalednabawi10@gmail.com
Project Link: https://github.com/khalednabawey/us-census-demographic-analysis
