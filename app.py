 import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Agri AI Predictor", layout="wide")

st.title("🌾 AI-Based Agri Price Prediction System")
st.markdown("### Advanced Dashboard with Forecasting & Market Intelligence")

# ===============================
# FILE UPLOAD
# ===============================
file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if file:

    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # ===============================
    # ENCODING (SAFE)
    # ===============================
    encoders = {}

    def encode(df):
        df = df.copy()
        for col in df.select_dtypes(include='object').columns:
            if col != 'Date':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                encoders[col] = le
        return df

    df = encode(df)

    # ===============================
    # FEATURE ENGINEERING
    # ===============================
    def create_features(df):
        df = df.copy()

        for lag in range(1, 8):
            df[f'lag_{lag}'] = df['Price'].shift(lag)

        df['rolling_mean'] = df['Price'].rolling(7).mean()
        df['rolling_std'] = df['Price'].rolling(7).std()

        df['month'] = df['Date'].dt.month
        df['dayofweek'] = df['Date'].dt.dayofweek

        return df.dropna()

    df_feat = create_features(df)

    # ===============================
    # TRAIN MODEL
    # ===============================
    X = df_feat.drop(['Price', 'Date'], axis=1)
    y = df_feat['Price']

    split = int(len(df_feat)*0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = RandomForestRegressor(n_estimators=300, max_depth=12)
    model.fit(X_train, y_train)

    # ===============================
    # PERFORMANCE
    # ===============================
    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))

    col1, col2 = st.columns(2)
    col1.metric("MAE", round(mae,2))
    col2.metric("RMSE", round(rmse,2))

    # ===============================
    # GRAPH
    # ===============================
    st.subheader("📉 Actual vs Predicted")
    st.line_chart(pd.DataFrame({"Actual": y_test, "Predicted": pred}))

    # ===============================
    # FEATURE IMPORTANCE
    # ===============================
    st.subheader("🔍 Feature Importance")
    imp = pd.Series(model.feature_importances_, index=X.columns)
    st.bar_chart(imp.sort_values(ascending=False))

    # ===============================
    # FUTURE FORECAST
    # ===============================
    def forecast(model, df, steps=7):
        df = df.copy()
        future = []

        for _ in range(steps):
            last = df.iloc[-1:].copy()
            X_last = last.drop(['Price', 'Date'], axis=1)

            p = model.predict(X_last)[0]
            future.append(p)

            new = last.copy()
            new['Price'] = p
            new['Date'] += pd.Timedelta(days=1)

            df = pd.concat([df, new])
            df = create_features(df)

        return future

    if st.button("🔮 Predict Next 7 Days"):
        future = forecast(model, df_feat)
        st.write("Future Prices:", future)

    # ===============================
    # MARKET ANALYSIS
    # ===============================
    st.subheader("📊 Market Insights")

    df_feat['volatility'] = df_feat['Price'].rolling(7).std()
    df_feat['trend'] = df_feat['Price'].rolling(7).mean()

    vol = "High ⚠️" if df_feat['volatility'].iloc[-1] > df_feat['volatility'].mean() else "Stable ✅"
    trend = "Up 📈" if df_feat['Price'].iloc[-1] > df_feat['trend'].iloc[-1] else "Down 📉"

    st.write("Volatility:", vol)
    st.write("Trend:", trend)

    # ===============================
    # USER INPUT PREDICTION
    # ===============================
    st.subheader("🎯 Custom Prediction")

    user_data = {}
    for col in X.columns:
        val = st.number_input(f"{col}", value=float(X[col].mean()))
        user_data[col] = val

    if st.button("Predict Price"):
        user_df = pd.DataFrame([user_data])
        pred_price = model.predict(user_df)[0]
        st.success(f"Predicted Price: {round(pred_price,2)}")

    # ===============================
    # ALL POSSIBLE CONDITIONS (AUTO)
    # ===============================
    st.subheader("🧠 Scenario Simulation (Auto Predictions)")

    if st.button("Generate Scenarios"):
        sample = X.sample(50)  # simulate conditions
        preds = model.predict(sample)

        result = sample.copy()
        result["Predicted Price"] = preds


        st.dataframe(result)

else:
    st.info("Upload dataset to start 🚀")
