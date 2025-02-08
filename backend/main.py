from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import tensorflow as tf
import uvicorn
import os
from .import data_processor
import numpy as np
import io
import pickle as pkl

app = FastAPI()

# Load the trained model
try:
    model = tf.keras.models.load_model("./backend/models/NN-model.h5")
except Exception as e:
    model = None
    print(f"Warning: Could not load model: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint providing API information."""
    return {
        "message": "Welcome to the Census Demographics Prediction API",
        "endpoints": {
            "predict": "/predict/ [POST] - Make predictions from CSV file",
            "health": "/health-check/ [GET] - Check API health status"
        }
    }


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Endpoint to receive data and make predictions."""
    try:
        # Validate file extension
        if not file.filename.endswith('.csv'):
            raise HTTPException(
                status_code=400,
                detail="Only CSV files are supported"
            )

        # Read the file content
        content = await file.read()
        if not content:
            raise HTTPException(
                status_code=400,
                detail="Empty file uploaded"
            )

        # Try to read the CSV file with different encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                input_data = pd.read_csv(io.StringIO(content.decode(encoding)))
                break  # If successful, break the loop
            except UnicodeDecodeError:
                continue  # Try next encoding
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail="Failed to parse CSV file: Invalid format"
                )
        else:  # If no encoding worked
            raise HTTPException(
                status_code=400,
                detail="Could not read the CSV file with any supported encoding"
            )

        # Validate that the DataFrame is not empty
        if input_data.empty:
            raise HTTPException(
                status_code=400,
                detail="The uploaded CSV file is empty"
            )

        # Ensure the model is loaded
        if model is None:
            raise HTTPException(
                status_code=500,
                detail="Model is not loaded"
            )

        # Process the data using our data processing pipeline
        try:
            processed_data = data_processor.prepare_for_prediction(input_data)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail="Error processing data: Invalid format or missing columns"
            )

        # Make predictions
        try:
            predictions = model.predict(processed_data)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail="Error making predictions"
            )

        try:
            with open('./backend/models/target_scaler.pkl', 'rb') as f:
                target_scaler = pkl.load(f)

            original_scale_predictions = target_scaler.inverse_transform(
                predictions)

            # Ensure predictions match the input data length
            if len(original_scale_predictions) != len(input_data):
                raise HTTPException(
                    status_code=400,
                    detail=f"Prediction length ({len(original_scale_predictions)}) does not match input data length ({len(input_data)})"
                )

            # Format predictions
            formatted_predictions = [
                {
                    "index": i,
                    "prediction": float(pred[0])
                }
                for i, pred in enumerate(original_scale_predictions)
            ]

            # Calculate statistics
            stats = {
                "mean": float(np.mean(original_scale_predictions)),
                "min": float(np.min(original_scale_predictions)),
                "max": float(np.max(original_scale_predictions)),
                "std": float(np.std(original_scale_predictions))
            }

            # Return predictions with metadata
            return {
                "status": "success",
                "num_predictions": len(predictions),
                "statistics": stats,
                "predictions": formatted_predictions
            }

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail="Error processing predictions"
            )

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail="An unexpected error occurred"
        )


@app.get("/health-check/")
async def health_check():
    """Endpoint to check API health and model status."""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
