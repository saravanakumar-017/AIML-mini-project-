# AI-Based Agri Price Prediction System

## Overview
This project is a Streamlit-based web application that predicts agricultural commodity prices using Machine Learning techniques.

The system analyzes historical data such as date, crop details, and previous prices to forecast future market trends and help farmers and traders make better decisions.

---

## Features
- Upload CSV dataset
- Automatic data preprocessing and encoding
- Feature engineering (lag, rolling stats, time features)
- Random Forest model training
- Actual vs Predicted visualization
- Feature importance analysis
- Future price forecasting (next 7 days)
- Market insights (trend and volatility)
- Custom user input prediction
- Scenario simulation (multiple predictions)

---

## Technologies Used
- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn

---

## Project Structure
mini-pro/
├── app.py
├── src/
├── data/
├── models/
├── requirements.txt
└── README.md

---

## How to Run the Project

### 1. Install dependencies
pip install -r requirement.txt

### 2. Run the application
streamlit run app/app.py

---

## Input Dataset Requirements
Your CSV file must contain:
- Date column
- Price column
- Other categorical or numerical features

---

## Model Details
- Algorithm: Random Forest Regressor
- Train/Test Split: 80/20
- Evaluation Metrics:
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)

---

## Forecasting
The system predicts future prices using:
- Lag features (previous 7 days)
- Rolling mean and standard deviation
- Time-based features

---

## Market Insights
- Volatility detection
- Trend analysis (Uptrend / Downtrend)

---

## Team Members
Saravanakumar C
Saktheeshwaran K
Saravana S


---

## Future Improvements
- Add LSTM deep learning model integration
- Real-time API data integration
- Advanced visualization dashboard

---

## Conclusion
This project demonstrates how AI can be used in agriculture to predict prices and support decision-making using data-driven insights.
