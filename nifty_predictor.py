# Nifty 50 Predictor using Kotak Neo API and LightGBM (v4 - Simplified Polling Architecture)
#
# --- How to Run in Google Colab ---
# 1. Upload this script (`nifty_predictor.py`) and `requirements.txt` to your Colab environment.
# 2. In a new Colab notebook, run the following commands in a cell to install TA-Lib and other dependencies:
#    !wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
#    !tar -xzvf ta-lib-0.4.0-src.tar.gz
#    %cd ta-lib/
#    !./configure --prefix=/usr
#    !make
#    !make install
#    !pip install --upgrade pip
#    !pip install -r ../requirements.txt
# 3. Fill in your Kotak Neo API credentials and WhatsApp webhook URL in the configuration section below.
# 4. Run the script in a new cell:
#    !python ../nifty_predictor.py
#
# --- IMPORTANT: OPERATIONAL DESIGN & LIMITATIONS ---
#
# 1.  **Architecture:** This script uses a simple and robust **polling-based architecture**. It does NOT use
#     websockets. Every 5 minutes, it fetches the latest data, makes a prediction, and waits.
#
# 2.  **Manual Start:** Due to the Kotak Neo API's security design, you must **manually enter an OTP**
#     when the script starts.
#
# 3.  **Session Expiry:** The API session will expire after a few hours. When this happens, the script
#     will stop. You will need to **manually restart the script** and enter a new OTP to continue.
#     This script is designed for attended operation during market hours, not for 24/7 unattended use.
#
# 4.  **No PCR:** This version focuses on a purely technical model based on price action (RSI, MAs, etc.).
#     The complex and inefficient live Put-Call Ratio (PCR) calculation has been removed to improve
#     reliability and focus on a functional core.
#
# Disclaimer: This script is for educational purposes only. Trading in the stock market involves risk.
# The developer is not responsible for any financial losses.

import time
import pandas as pd
import numpy as np
import lightgbm as lgb
import requests
from neo_api_client import NeoAPI
import yfinance as yf
import talib
import os
from datetime import datetime

# --- Configuration ---
KOTAK_CONSUMER_KEY = "YOUR_CONSUMER_KEY"
KOTAK_CONSUMER_SECRET = "YOUR_CONSUMER_SECRET"
KOTAK_MOBILE_NUMBER = "YOUR_MOBILE_NUMBER"
KOTAK_PASSWORD = "YOUR_PASSWORD"
WHATSAPP_WEBHOOK_URL = "YOUR_WHATSAPP_WEBHOOK_URL"
NIFTY_SYMBOL = "NIFTY 50"
NIFTY_TICKER = "^NSEI"

# --- Kotak Neo API Functions ---
def authenticate_kotak_neo():
    """Authenticates with Kotak Neo API."""
    client = NeoAPI(consumer_key=KOTAK_CONSUMER_KEY, consumer_secret=KOTAK_CONSUMER_SECRET, environment='prod')
    client.login(mobilenumber=KOTAK_MOBILE_NUMBER, password=KOTAK_PASSWORD)
    otp = input("Enter OTP received on your mobile: ")
    client.session_2fa(OTP=otp)
    return client

def get_nifty_instrument_token(client):
    """Gets the instrument token for Nifty 50 index."""
    try:
        scrips_filepath = client.scrip_master("ind_nifty")
        df = pd.read_csv(scrips_filepath)
        token = df[df['dSymbol'] == NIFTY_SYMBOL]['instrumentToken'].iloc[0]
        return token
    except Exception as e:
        print(f"Error getting Nifty 50 instrument token: {e}")
        return None

# --- Data & Model Functions ---
def get_historical_data():
    """Fetches historical Nifty 50 data using yfinance for model training."""
    print("Fetching historical data for training...")
    nifty = yf.Ticker(NIFTY_TICKER)
    data = nifty.history(period="60d", interval="5m")
    data.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
    data.dropna(inplace=True)
    print(f"Fetched {len(data)} rows of historical data.")
    return data

def create_features(data):
    """Creates technical analysis features from historical data."""
    data['rsi'] = talib.RSI(data['close'])
    data['sma_fast'] = talib.SMA(data['close'], timeperiod=10)
    data['sma_slow'] = talib.SMA(data['close'], timeperiod=30)
    return data

def train_model(data):
    """Trains a LightGBM model on historical data."""
    print("Training model...")
    # 1. Create features from the price data
    data_with_features = create_features(data.copy())

    # 2. Define the target variable (predicting the next candle's direction)
    data_with_features['target'] = (data_with_features['close'].shift(-1) > data_with_features['close']).astype(int)

    # 3. Create the feature set (X) and target set (y)
    features = ['rsi', 'sma_fast', 'sma_slow', 'open', 'high', 'low', 'close', 'volume']

    # 4. CRITICAL: Shift features to prevent data leakage.
    # We use data from candle 't-1' to predict the outcome of candle 't'.
    X = data_with_features[features].shift(1)
    y = data_with_features['target']

    # 5. Drop rows with NaN values that were introduced by feature creation and shifting
    X.dropna(inplace=True)
    y = y[X.index] # Align target with features

    # 6. Train the model
    model = lgb.LGBMClassifier(objective='binary', metric='binary_logloss')
    model.fit(X, y)
    print("Model training complete.")
    return model, features

def predict_signal(model, latest_features):
    """Predicts the next price movement and returns 'BUY' or 'SELL'."""
    prediction = model.predict(latest_features)
    return 'BUY' if prediction[0] == 1 else 'SELL'

# --- Notification ---
def send_whatsapp_notification(message):
    """Sends a notification to the configured WhatsApp webhook URL."""
    if WHATSAPP_WEBHOOK_URL == "YOUR_WHATSAPP_WEBHOOK_URL" or not WHATSAPP_WEBHOOK_URL:
        print(f"INFO: WhatsApp Webhook URL not configured. Skipping notification.")
        return
    try:
        requests.post(WHATSAPP_WEBHOOK_URL, json={'text': message}, timeout=10)
        print(f"Notification sent: {message}")
    except requests.exceptions.RequestException as e:
        print(f"Error sending WhatsApp notification: {e}")

# --- Main Script ---
if __name__ == "__main__":
    print("Starting Nifty 50 prediction bot (v4 - Simplified Polling)...")

    # 1. Authenticate and get client and token
    try:
        client = authenticate_kotak_neo()
        nifty_token = get_nifty_instrument_token(client)
        if not nifty_token:
            raise Exception("Could not retrieve Nifty 50 instrument token.")
        print("Authentication successful.")
    except Exception as e:
        print(f"Authentication or setup failed: {e}")
        exit()

    # 2. Train initial model
    try:
        historical_data = get_historical_data()
        live_data_df = historical_data.copy()
        model, features = train_model(live_data_df.copy())
    except Exception as e:
        print(f"Failed to train initial model: {e}")
        exit()

    # 3. Start the polling loop
    print("Starting prediction loop... Press Ctrl+C to stop.")
    while True:
        try:
            # We sleep at the start of the loop to wait for the next 5-minute interval.
            print("\n" + "="*50)
            print(f"Cycle start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. Waiting for 5 minutes...")
            time.sleep(300)

            # Fetch the latest completed 5-minute candle from yfinance
            # This is necessary for feature calculation as the Kotak API does not provide historical intraday candles.
            latest_bar = yf.Ticker(NIFTY_TICKER).history(period="1d", interval="5m").iloc[-1:]
            latest_bar.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)

            if not latest_bar.empty and (live_data_df.empty or latest_bar.index[-1] > live_data_df.index[-1]):
                print(f"New 5m candle received for {latest_bar.index[-1].strftime('%Y-%m-%d %H:%M:%S')}")
                live_data_df = pd.concat([live_data_df, latest_bar])
                live_data_df = live_data_df[~live_data_df.index.duplicated(keep='last')]

                # Create features for the latest data
                data_with_features = create_features(live_data_df.copy())

                # Get the latest features for prediction (use the second to last row, as the last is for the current, incomplete candle)
                latest_features_for_prediction = data_with_features[features].iloc[-1:]

                # Generate signal
                signal = predict_signal(model, latest_features_for_prediction)
                print(f"Model generated signal: {signal}")

                # Fetch live price from Kotak API for notification
                quote_resp = client.quotes(instrument_tokens=[{'instrument_token': str(nifty_token), 'exchange_segment': 'ind_nifty'}], quote_type='ltp')
                live_ltp = float(quote_resp['message'][0]['last_traded_price'])

                message = f"Nifty 50 Signal: {signal} (Live Price: {live_ltp:.2f}, Candle Close: {latest_bar['close'].iloc[0]:.2f})"
                send_whatsapp_notification(message)

            else:
                print("No new candle data from yfinance yet. Waiting for next cycle.")

        except KeyboardInterrupt:
            print("\nBot stopped by user.")
            break
        except Exception as e:
            print(f"An error occurred in the prediction loop: {e}")
            print("Waiting for 60 seconds before retrying...")
            time.sleep(60)