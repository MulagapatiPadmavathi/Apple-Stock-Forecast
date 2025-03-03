# Apple Stock Forecast

## Overview  
This project predicts Apple’s stock prices for the next 30 days using historical data from 2012 to 2019. We performed exploratory data analysis (EDA), trend analysis, and experimented with multiple forecasting models, ultimately selecting the GRU model for its superior performance.  

## Objectives  
- Predict Apple stock prices for the next 30 days.  
- Perform EDA and identify short-term and long-term trends.  
- Compare different forecasting models (ARIMA, LSTM, GRU, etc.).  
- Evaluate external factors influencing stock prices.  
- Deploy the best-performing model for real-time forecasting.  

## Dataset  
- **Data Source**: Yahoo Finance (2012-2019)  
- **Features**: Open, High, Low, Close prices  

## Models Evaluated  
- **ARIMA**  
- **LSTM**  
- **GRU** (Selected as the best model)  
- **Other time series models**  

## Model Performance (GRU)  
- **Mean Squared Error (MSE):** *3.7548*  
- **Root Mean Squared Error (RMSE):** *5.2294*  

## Tools & Technologies  
- Python  
- TensorFlow / Keras  
- Scikit-learn  
- Pandas, NumPy, Matplotlib, Seaborn  
- Streamlit (for deployment)  

## Results  
- GRU outperformed other models in forecasting accuracy.  
- Short-term and long-term trends were successfully analyzed.  
- The model is deployed for real-time predictions.  

## Deployment  
The trained GRU model is deployed using **Streamlit**, enabling users to input dates and receive predicted stock prices.  
