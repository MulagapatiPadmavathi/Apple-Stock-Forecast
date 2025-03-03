import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import GRU, Dense
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta

# Set Streamlit page title
st.title("Apple Stock Price Prediction using GRU")

# Load the GRU model (Train and Save it separately before deployment)
def create_gru_model():
    model = Sequential([
        GRU(50, activation='relu', return_sequences=True, input_shape=(1,1)),
        GRU(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Load pre-trained model (Ensure this file is saved after training)
gru_model = create_gru_model()
gru_model.load_weights('gru_model.h5')  # Load saved model weights

# Upload dataset
uploaded_file = st.file_uploader("Upload Apple Stock CSV File", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    st.write("### Historical Data")
    st.write(df.tail())  # Display last few rows

    # Preprocessing
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[['Close']])
    
    # Predict last few days
    X_test = df_scaled[-30:-1].reshape(-1, 1, 1)
    y_pred = gru_model.predict(X_test)
    y_pred_actual = scaler.inverse_transform(y_pred)

    # Display predictions
    st.write("### Last 30 Days Predictions")
    pred_df = pd.DataFrame({'Date': df.index[-29:], 'Predicted_Close': y_pred_actual.flatten()})
    st.write(pred_df)

    # Forecast next 30 days
    st.write("### Forecast for Next 30 Days")

    future_days = 30
    last_value = df_scaled[-1].reshape(1, 1, 1)
    predictions = []
    dates = [df.index[-1] + timedelta(days=i) for i in range(1, future_days + 1)]

    for _ in range(future_days):
        predicted_value = gru_model.predict(last_value)
        predictions.append(predicted_value[0][0])
        last_value = np.array(predicted_value).reshape(1, 1, 1)

    future_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    forecast_df = pd.DataFrame({'Date': dates, 'Forecasted_Close': future_prices.flatten()})
    st.write(forecast_df)

    # Plot results
    st.write("### Stock Price Forecast")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df['Close'], label='Historical Prices', color='blue')
    ax.plot(dates, future_prices, label='GRU Forecast', color='red', linestyle='dashed')
    ax.set_xlabel("Date")
    ax.set_ylabel("Stock Price")
    ax.set_title("Apple Stock Price Forecast for Next 30 Days")
    ax.legend()
    st.pyplot(fig)
