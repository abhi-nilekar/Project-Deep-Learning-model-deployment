import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import streamlit as st

# Function to fetch data from Yahoo Finance
@st.cache(allow_output_mutation=True)
def fetch_data_from_yahoo(tickers):
    data = pd.DataFrame()
    for ticker in tickers:
        df = yf.download(ticker, start="2010-01-01", end="2024-01-01")
        df['Ticker'] = ticker
        data = pd.concat([data, df], ignore_index=True)
    return data

# Function to fetch data for a selected ticker
def fetch_data(data, selected_ticker):
    return data[data['Ticker'] == selected_ticker]

# Function to build and train the LSTM model
def train_model(X_train, y_train):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))  # Output layer

    model.compile(optimizer='adam', loss='mean_squared_error')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=0)

    return model

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ["Home", "Fetch Data", "Forecast", "Download", "About Us"])

if page == "Home":
    st.title("Stock Price Forecasting App")
    st.write("Welcome to the Stock Price Forecasting App! Use the navigation bar to explore.")

elif page == "Fetch Data":
    st.title("Fetch Data")
    nifty50_tickers = ['RELIANCE.NS', 'INFY.NS', 'TCS.NS', 'HDFC.NS', 'ICICI.NS', 'HDFCBANK.NS', 'SBIN.NS', 'HINDUNILVR.NS', 'KOTAKBANK.NS', 'BAJFINANCE.NS', 'BHARTIARTL.NS', 'AXISBANK.NS', 'LT.NS', 'BAJAJFINSV.NS', 'WIPRO.NS', 'JSWSTEEL.NS', 'SUNPHARMA.NS', 'TITAN.NS', 'NESTLEIND.NS', 'ASIANPAINT.NS', 'HCLTECH.NS', 'ULTRACEMCO.NS', 'POWERGRID.NS', 'MARUTI.NS', 'DIVISLAB.NS', 'BPCL.NS', 'NTPC.NS', 'BRITANNIA.NS', 'SHREECEM.NS', 'CIPLA.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'ADANIPORTS.NS', 'TECHM.NS', 'GRASIM.NS', 'BAJAJ-AUTO.NS', 'COALINDIA.NS', 'HINDALCO.NS', 'INDUSINDBK.NS', 'M&M.NS', 'ONGC.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS', 'IOC.NS', 'HEROMOTOCO.NS', 'DMART.NS', 'UPL.NS', 'GAIL.NS', 'ADANIGREEN.NS', 'ADANITRANS.NS']
    
    if st.button("Fetch Data for Nifty 50"):
        data = fetch_data_from_yahoo(nifty50_tickers)
        st.success("Data for all Nifty 50 stocks has been fetched.")

elif page == "Forecast":
    st.title("Forecast")
    tickers = data['Ticker'].unique()
    selected_ticker = st.selectbox("Select a stock:", tickers)

    if st.button("Forecast"):
        # Fade out other elements
        st.markdown("<style>.fade { opacity: 0.3; }</style>", unsafe_allow_html=True)

        # Show the spinner while the model is running
        with st.spinner('Building model and forecasting the prices, please wait...'):
            stock_data = fetch_data(data, selected_ticker)
            stock_data = stock_data.reset_index()
            stock_data['Date'] = stock_data['Date'].astype(str)
            
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))

            # Create sequences for LSTM
            def create_dataset(data, time_step=1):
                X, y = [], []
                for i in range(len(data) - time_step - 1):
                    X.append(data[i:(i + time_step), 0])
                    y.append(data[i + time_step, 0])
                return np.array(X), np.array(y)

            time_step = 60  # 60 days of historical data
            X, y = create_dataset(scaled_data, time_step)
            X = X.reshape(X.shape[0], X.shape[1], 1)

            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            model = train_model(X_train, y_train)

            # Forecast the next 30 days
            last_data = scaled_data[-time_step:]
            forecast = []
            for _ in range(30):
                next_day = model.predict(last_data[np.newaxis, :, :])
                forecast.append(next_day[0, 0])
                last_data = np.concatenate([last_data[1:], [[next_day[0, 0]]]], axis=0)

            forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

            # Create a DataFrame with forecasted prices and ticker name
            forecast_df = pd.DataFrame({
                'Date': pd.date_range(start=stock_data['Date'].iloc[-1], periods=30, freq='D'),
                'Forecasted_Price': forecast.flatten(),
                'Ticker': selected_ticker
            })

        # Clear the fade effect
        st.markdown("<style>.fade { opacity: 1; }</style>", unsafe_allow_html=True)

        # Display the forecasted DataFrame
        st.write("### Forecasted Prices for the Next 30 Days")
        st.dataframe(forecast_df)

        # Plotting the line chart
        st.line_chart(forecast_df.set_index('Date')['Forecasted_Price'], use_container_width=True)
        st.write("**Hover over the chart to see the forecasted prices.**")

        # Risk Analysis using Standard Deviation
        st.write("## Risk Analysis")
        mean_price = forecast_df['Forecasted_Price'].mean()
        std_dev = forecast_df['Forecasted_Price'].std()
        last_price = forecast_df['Forecasted_Price'].iloc[-1]

        if last_price > mean_price:
            risk_level = "Profit Making Stock"
            color = "green"
        elif last_price < mean_price - 2 * std_dev:
            risk_level = "High Risk Stock"
            color = "red"
        elif last_price < mean_price - std_dev:
            risk_level = "Moderate Risk Stock"
            color = "orange"
        elif last_price < mean_price:
            risk_level = "Low Risk Stock"
            color = "yellow"

        # Display risk level with color
        st.markdown(f"<h3 style='color:{color};'>{risk_level}</h3>", unsafe_allow_html=True)
        st.image('risk matrix 101.png',width=500)

        # Download the forecast DataFrame as CSV
        csv = forecast_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, f"{selected_ticker}_forecast.csv", "text/csv")

elif page == "About Us":
    st.title("About Us")
    st.write("Meet our team:")

    # Team member information
    team_members = [
        {
            'image': 'https://via.placeholder.com/150',
            'name': 'John Doe'
        },
        {
            'image': 'https://via.placeholder.com/150',
            'name': 'Jane Smith'
        },
        {
            'image': 'https://via.placeholder.com/150',
            'name': 'Michael Johnson'
        },
        {
            'image': 'https://via.placeholder.com/150',
            'name': 'Emily Davis'
        },
        {
            'image': 'https://via.placeholder.com/150',
            'name': 'David Wilson'
        }
    ]

    # Display team members
    for member in team_members:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(member['image'], width=150)
        with col2:
            st.write(f"# {member['name']}")