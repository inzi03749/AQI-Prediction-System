import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="AQI Prediction System", layout="wide")

# -------------------------------------------------
# DARK PROFESSIONAL STYLING
# -------------------------------------------------
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
h1, h2, h3 {
    color: #00E5FF;
}
.stMetric {
    background-color: #1C1F26;
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("🌍 AQI Prediction System - Delhi")

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
df = pd.read_csv("final_dataset.csv")
df = df.fillna(df.mean())

# Keep a copy for plotting
df_plot = df.copy()
df_plot["Date"] = pd.to_datetime(df_plot["Date"])
df_plot = df_plot.sort_values("Date")

model = joblib.load("models/xgb_model.pkl")

target = "AQI"

# Use original numeric dataframe for model
X = df.drop(columns=[target])
y = df[target]

# -------------------------------------------------
# PROPER TEST ACCURACY
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

y_pred_test = model.predict(X_test)
r2 = r2_score(y_test, y_pred_test)

# -------------------------------------------------
# KPI CARDS
# -------------------------------------------------
latest_aqi = df["AQI"].iloc[-1]
avg_aqi = df["AQI"].mean()
max_aqi = df["AQI"].max()
min_aqi = df["AQI"].min()

col1, col2, col3, col4 = st.columns(4)

col1.metric("📊 Model Accuracy", f"{round(r2*100,2)}%")
col2.metric("📍 Latest AQI", round(latest_aqi,2))
col3.metric("📈 Avg AQI", round(avg_aqi,2))
col4.metric("⚠️ Max AQI Recorded", round(max_aqi,2))

st.markdown("---")

# -------------------------------------------------
# AQI GAUGE METER
# -------------------------------------------------
st.subheader("Current AQI Level")

gauge_fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=latest_aqi,
    title={'text': "Current AQI"},
    gauge={
        'axis': {'range': [0, 500]},
        'bar': {'color': "white"},
        'steps': [
            {'range': [0, 50], 'color': "green"},
            {'range': [50, 100], 'color': "lightgreen"},
            {'range': [100, 200], 'color': "yellow"},
            {'range': [200, 300], 'color': "orange"},
            {'range': [300, 500], 'color': "red"},
        ],
    }
))

st.plotly_chart(gauge_fig, use_container_width=True)

st.markdown("---")

# -------------------------------------------------
# ACTUAL VS PREDICTED (INTERACTIVE)
# -------------------------------------------------
st.subheader("Actual vs Predicted AQI")

plot_df = pd.DataFrame({
    "Actual AQI": y_test,
    "Predicted AQI": y_pred_test
})

fig1 = px.scatter(
    plot_df,
    x="Actual AQI",
    y="Predicted AQI",
    trendline="ols"
)

fig1.add_shape(
    type="line",
    x0=plot_df["Actual AQI"].min(),
    y0=plot_df["Actual AQI"].min(),
    x1=plot_df["Actual AQI"].max(),
    y1=plot_df["Actual AQI"].max(),
    line=dict(dash="dash")
)

st.plotly_chart(fig1, use_container_width=True)

# -------------------------------------------------
# SIDEBAR INPUT SECTION
# -------------------------------------------------
st.sidebar.header("🔮 Predict Future AQI")

selected_date = st.sidebar.date_input("Select Date")

day = selected_date.day
month = selected_date.month
year = selected_date.year

input_values = {
    "Date": day,
    "Month": month,
    "Year": year
}

remaining_cols = [col for col in X.columns if col not in ["Date", "Month", "Year"]]

for col_name in remaining_cols:
    input_values[col_name] = st.sidebar.number_input(
        col_name,
        value=float(X[col_name].mean())
    )

input_array = np.array([input_values[col] for col in X.columns]).reshape(1, -1)

if st.sidebar.button("Predict AQI"):
    prediction = model.predict(input_array)[0]

    st.sidebar.success(f"Predicted AQI: {round(prediction,2)}")

    if prediction <= 50:
        st.sidebar.info("Air Quality: Good 😊")
    elif prediction <= 100:
        st.sidebar.info("Air Quality: Satisfactory")
    elif prediction <= 200:
        st.sidebar.warning("Air Quality: Moderate")
    elif prediction <= 300:
        st.sidebar.warning("Air Quality: Poor")
    else:
        st.sidebar.error("Air Quality: Very Poor / Severe 🚨")

# -------------------------------------------------
# 7-DAY FORECAST
# -------------------------------------------------
st.subheader("Next 7 Days AQI Forecast")

last_date = df_plot["Date"].max()
last_row = X.iloc[-1].copy()

future_predictions = []
future_dates = []

for i in range(1, 8):
    future_date = last_date + pd.Timedelta(days=i)

    last_row["Date"] = future_date.day
    last_row["Month"] = future_date.month
    last_row["Year"] = future_date.year

    pred = model.predict(np.array(last_row).reshape(1, -1))[0]

    future_predictions.append(pred)
    future_dates.append(future_date)

future_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted AQI": future_predictions
})

fig3 = px.line(future_df, x="Date", y="Predicted AQI", markers=True)

st.plotly_chart(fig3, use_container_width=True)