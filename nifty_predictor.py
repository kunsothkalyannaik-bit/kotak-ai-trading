# Nifty 50 Futures Predictor using Kotak Neo API and LightGBM (v6 - Futures-Based Bot)
#
# --- How to Run in Google Colab ---
# 1. Upload this script (`nifty_predictor.py`) and `requirements.txt` to your Colab environment.
# 2. In a new Colab notebook, run the installation commands (this may take a minute):
#    !wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && tar -xzvf ta-lib-0.4.0-src.tar.gz
#    %cd ta-lib/
#    !./configure --prefix=/usr && make && make install
#    !pip install --upgrade pip -q
#    !pip install -r ../requirements.txt -q
#    %cd ..
#
# --- IMPORTANT: TWO-MODE OPERATION FOR NIFTY 50 FUTURES ---
#
# This script tracks and predicts the near-month NIFTY 50 FUTURES contract.
# It has two modes, set by the `--mode` flag.
#
# 1. Data Collection Mode (`--mode collect`):
#    - Use this mode first. The script will connect to the Kotak Neo API and start recording
#      live 5-minute Nifty 50 Futures candles into a file named `nifty_futures_data.csv`.
#    - It will NOT make any predictions in this mode.
#    - Let this run for at least a few hours/days during market hours to build a dataset.
#    - Example command: !python nifty_predictor.py --mode collect
#
# 2. Prediction Mode (`--mode predict`):
#    - Use this mode only AFTER you have collected enough data (e.g., a few hundred candles).
#    - The script will load all data from `nifty_futures_data.csv`, train a model on it, and then
#      start making live 'BUY'/'SELL' predictions on each new 5-minute Futures candle.
#    - Example command: !python nifty_predictor.py --mode predict
#
# --- Session Expiry ---
# Due to the API's OTP requirement, you must manually start the script. The session will
# expire after a few hours, and you will need to restart the script to continue.

import time
import pandas as pd
import numpy as np
import lightgbm as lgb
import requests
from neo_api_client import NeoAPI
import talib
import os
import threading
import argparse
from datetime import datetime, timedelta

# --- Configuration ---
KOTAK_CONSUMER_KEY = "YOUR_CONSUMER_KEY"
KOTAK_CONSUMER_SECRET = "YOUR_CONSUMER_SECRET"
KOTAK_MOBILE_NUMBER = "YOUR_MOBILE_NUMBER"
KOTAK_PASSWORD = "YOUR_PASSWORD"
NIFTY_FUTURES_SYMBOL = "NIFTY" # Symbol for futures in the NFO scrip master
DATA_FILE = "nifty_futures_data.csv"

# --- Candle Aggregator Class ---
class CandleAggregator:
    def __init__(self, on_candle_callback, interval_minutes=5):
        self.interval = timedelta(minutes=interval_minutes)
        self.on_candle_callback = on_candle_callback
        self.current_candle = None

    def add_tick(self, price):
        now = datetime.now()
        if self.current_candle is None:
            self._start_new_candle(now, price)
            return

        if now >= self.current_candle['timestamp'] + self.interval:
            self.current_candle['close'] = price
            if self.on_candle_callback:
                self.on_candle_callback(self.current_candle.copy())
            self._start_new_candle(now, price)
        else:
            self.current_candle['high'] = max(self.current_candle['high'], price)
            self.current_candle['low'] = min(self.current_candle['low'], price)
            self.current_candle['close'] = price

    def _start_new_candle(self, timestamp, price):
        minute_offset = timestamp.minute % self.interval.total_seconds() / 60
        candle_start_time = timestamp.replace(second=0, microsecond=0) - timedelta(minutes=minute_offset)
        self.current_candle = {'timestamp': candle_start_time, 'open': price, 'high': price, 'low': price, 'close': price}
        print(f"Starting new 5m candle at {candle_start_time.strftime('%Y-%m-%d %H:%M:%S')}")

# --- Data Handling ---
def save_candle_to_csv(candle):
    df = pd.DataFrame([candle])
    file_exists = os.path.isfile(DATA_FILE)
    df.to_csv(DATA_FILE, mode='a', header=not file_exists, index=False)
    print(f"Saved candle for {candle['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")

def load_candles_from_csv():
    if not os.path.isfile(DATA_FILE):
        return pd.DataFrame()
    df = pd.read_csv(DATA_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    return df

# --- Model & Prediction ---
def create_features(data):
    data['rsi'] = talib.RSI(data['close'])
    data['sma_fast'] = talib.SMA(data['close'], timeperiod=10)
    data['sma_slow'] = talib.SMA(data['close'], timeperiod=30)
    return data

def train_model(data):
    print("Training model...")
    data_with_features = create_features(data.copy())
    data_with_features['target'] = (data_with_features['close'].shift(-1) > data_with_features['close']).astype(int)
    features = ['rsi', 'sma_fast', 'sma_slow', 'open', 'high', 'low', 'close']
    X = data_with_features[features].shift(1)
    y = data_with_features['target']
    X.dropna(inplace=True)
    y = y[X.index]
    model = lgb.LGBMClassifier(objective='binary')
    model.fit(X, y)
    print("Model training complete.")
    return model, features

def display_signal_alert(signal, live_price, candle_close):
    border = "**************************************************"
    title = f"--- TRADING SIGNAL: {signal.upper()} ---"
    price_line = f"Live Price: {live_price:.2f}"
    candle_line = f"Trigger Candle Close: {candle_close:.2f}"
    print(f"\n{border}\n{title.center(len(border))}\n{price_line.center(len(border))}\n{candle_line.center(len(border))}\n{border}\n")

# --- API & Websocket ---
def authenticate_kotak_neo():
    client = NeoAPI(consumer_key=KOTAK_CONSUMER_KEY, consumer_secret=KOTAK_CONSUMER_SECRET, environment='prod')
    client.login(mobilenumber=KOTAK_MOBILE_NUMBER, password=KOTAK_PASSWORD)
    otp = input("Enter OTP received on your mobile: ")
    client.session_2fa(OTP=otp)
    return client

def get_nifty_futures_token(client):
    """
    Gets the instrument token for the near-month Nifty 50 Futures contract.
    This is a tradable instrument and will provide a live tick feed.
    """
    try:
        scrips_filepath = client.scrip_master("nfo")
        df = pd.read_csv(scrips_filepath)

        # Filter for Nifty futures
        nifty_futures = df[(df['pSymbol'] == NIFTY_FUTURES_SYMBOL) & (df['pInstrumentType'] == 'FUT')]

        # Find the nearest expiry date
        nifty_futures['pExpiryDate'] = pd.to_datetime(nifty_futures['pExpiryDate'])
        near_month_expiry = nifty_futures['pExpiryDate'].min()

        # Get the token for the near-month contract
        token = nifty_futures[nifty_futures['pExpiryDate'] == near_month_expiry]['instrumentToken'].iloc[0]
        print(f"Found near-month Nifty Futures token: {token} (Expiry: {near_month_expiry.strftime('%Y-%m-%d')})")
        return token
    except Exception as e:
        print(f"Error getting Nifty 50 Futures instrument token: {e}")
        return None

def setup_websocket(client, token, on_tick_callback):
    """Sets up the websocket for the Nifty Futures contract."""
    client.on_message = lambda msg: on_tick_callback(float(msg[0]['ltp'])) if isinstance(msg, list) and 'ltp' in msg[0] else None
    client.on_error = lambda err: print(f"Websocket Error: {err}")
    # Use the correct exchange segment 'nfo' for futures
    client.subscribe(instrument_tokens=[{"instrument_token": str(token), "exchange_segment": "nfo"}], isIndex=False)
    ws_thread = threading.Thread(target=client.connect, daemon=True)
    ws_thread.start()
    print("Websocket connected and listening for live Nifty Futures price ticks...")

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Nifty 50 Futures Prediction Bot")
    parser.add_argument('--mode', type=str, required=True, choices=['collect', 'predict'], help='Operation mode: "collect" or "predict"')
    args = parser.parse_args()

    print(f"Starting bot in '{args.mode}' mode.")

    try:
        client = authenticate_kotak_neo()
        nifty_futures_token = get_nifty_futures_token(client)
        if not nifty_futures_token: raise Exception("Could not get Nifty Futures token.")
    except Exception as e:
        print(f"Authentication failed: {e}")
        return

    if args.mode == 'collect':
        print("--- DATA COLLECTION MODE ---")
        print(f"Appending new 5-minute candles to '{DATA_FILE}'. Press Ctrl+C to stop.")
        aggregator = CandleAggregator(on_candle_callback=save_candle_to_csv)
        setup_websocket(client, nifty_futures_token, on_tick_callback=aggregator.add_tick)

    elif args.mode == 'predict':
        print("--- PREDICTION MODE ---")
        live_data_df = load_candles_from_csv()
        if len(live_data_df) < 50: # Arbitrary number, need enough data for feature calculation and training
            print(f"Not enough data to train model. Found {len(live_data_df)} candles. Please run in 'collect' mode first.")
            return

        model, features = train_model(live_data_df)

        def prediction_callback(candle):
            nonlocal live_data_df, model, features
            save_candle_to_csv(candle) # Save the new candle first

            new_row = pd.DataFrame([candle])
            new_row['timestamp'] = pd.to_datetime(new_row['timestamp'])
            new_row.set_index('timestamp', inplace=True)

            live_data_df = pd.concat([live_data_df, new_row])
            live_data_df = live_data_df[~live_data_df.index.duplicated(keep='last')]

            data_with_features = create_features(live_data_df.copy())
            latest_features = data_with_features[features].iloc[-1:]

            signal = "HOLD" # Default to HOLD
            if not latest_features.isnull().values.any():
                signal = "BUY" if model.predict(latest_features)[0] == 1 else "SELL"

            display_signal_alert(signal, candle['close'], candle['close'])

        aggregator = CandleAggregator(on_candle_callback=prediction_callback)
        setup_websocket(client, nifty_futures_token, on_tick_callback=aggregator.add_tick)

    # Keep the main script running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nBot stopped by user.")

if __name__ == "__main__":
    main()