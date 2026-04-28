Agri Price Prediction System
Project Overview

Agri Price Prediction System is a machine learning-based web application developed using Python and Streamlit to predict agricultural commodity prices based on historical data.

The system analyzes factors such as past prices, trends, and seasonal patterns to predict future prices using a trained Random Forest model. It also provides interactive visualizations for better understanding of market behavior.

Objectives
Predict agricultural commodity prices accurately
Analyze trends and seasonal patterns
Detect price volatility
Provide data-driven insights for decision making
Features
Crop price prediction
Future forecasting (next 7 days)
Trend and volatility analysis
Feature importance visualization
Scenario simulation
Interactive Streamlit interface
Machine learning-based prediction
Technologies Used
Python
Streamlit
Scikit-learn
Pandas
NumPy
Project Structure
agri-price-predictor/
│
├── app/
│   └── app.py
│
├── dataset/
│   └── agri_price_dataset.csv
└── README.md
Installation

Clone repository:

git clone https://github.com/yourusername/agri-price-predictor.git
cd agri-price-predictor

Create virtual environment:

python -m venv venv

Activate environment:

Install dependencies:

pip install -r requirements.txt
Run Project
streamlit run app/app.py
Output

Open browser:
http://localhost:8501

Input Parameters Used
Date
Commodity (optional)
Lag features (previous prices)
Rolling mean (trend)
Rolling standard deviation (volatility)
Month
Day of week
Prediction Graphs
Actual vs Predicted Prices
Feature Importance Chart
Trend Analysis
Volatility Analysis
Machine Learning Model

Random Forest Regressor is used for prediction.

AI/ML Concepts Used
Supervised Learning
Regression
Ensemble Learning (Random Forest)
Feature Engineering
Time Series Forecasting
Model Evaluation (MAE, RMSE)
Explainable AI (Feature Importance)
Future Enhancements
Integration with weather data 
Deep learning models (LSTM)
Real-time market data integration
Cloud deployment
Improved prediction accuracy
Team Members
Saktheeshwaran K
Saravana S
Saravanakumar C
