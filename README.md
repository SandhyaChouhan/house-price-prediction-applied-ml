# House Price Prediction – Applied Machine Learning

## Overview
This project presents an end-to-end applied machine learning solution to predict house prices using real-world real estate data. The focus is on building a scalable, production-ready ML system rather than only training a model.

## Problem Statement
Predict the sale price of residential properties based on structural, categorical, and temporal features.

## Dataset
- ~247,000 real estate property records
- Features include:
  - Size, Bedrooms, Bathrooms
  - Location, Property Type, Condition
  - Year Built, Date Sold
- Target variable: **Price**

## Methodology
1. Data cleaning and preprocessing
2. Feature engineering (Property Age, Sold Year, Sold Month)
3. Model training using LightGBM
4. Evaluation using MAE and RMSE
5. Deployment as a FastAPI service

## Model
- **LightGBM Regressor**
- Optimized for tabular data
- Multicore training enabled (`n_jobs = -1`)

## Results
- Mean Absolute Error (MAE): ~19,231  
- Root Mean Squared Error (RMSE): ~29,228  

These results indicate strong predictive performance for real estate pricing.

## Deployment
- Model deployed using **FastAPI**
- Served via **Uvicorn**
- JSON-based real-time prediction API
- Swagger UI available for testing

## Why This Is Applied ML
- Handles real-world noisy data
- Uses domain-driven feature engineering
- Selects appropriate ML model (not overfitting with deep learning)
- End-to-end pipeline from data to deployment

## Future Improvements
- Geospatial features for location modeling
- SHAP-based model explainability
- Cloud deployment (AWS/GCP)
