
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
import time
import schedule
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import re

# Configuration
LOTTERY_CONFIG = {
    'Lotto Max': {
        'url': 'https://www.olg.ca/en/lottery/play-lotto-max-encore/past-results.html',
        'csv': 'lottery_results.csv',
        'model': 'modelmax.h5',
        'num_numbers': 8,
        'max_num': 50
    },
    'Lotto 649': {
        'url': 'https://www.olg.ca/en/lottery/play-lotto-649-encore/past-results.html',
        'csv': 'lotto649.csv',
        'model': 'model649.h5',
        'num_numbers': 7,
        'max_num': 49
    }
}

# Custom CSS for number display
st.markdown(""" 
<style> 
.prediction-card { 
    padding: 20px; 
    background: #1a1a2e; 
    border-radius: 10px; 
    box-shadow: 0 0 15px rgba(30, 136, 229, 0.7); 
} 
.number-badge { 
    display: inline-block; 
    width: 50px; 
    height: 50px; 
    border-radius: 50%; 
    background: linear-gradient(45deg, #0d47a1, #42a5f5); 
    color: white; 
    text-shadow: 0 0 10px rgba(255,255,255,0.8); 
    text-align: center; 
    line-height: 50px; 
    font-weight: bold; 
} 
.prediction-label { 
    color: #42a5f5; 
    font-size: 1.3em; 
    font-weight: bold; 
} 
</style> 


""", unsafe_allow_html=True)

def generate_predictions(base_prediction, config):
    """Generate multiple predictions with perturbations"""
    predictions = []
    
    # Best prediction (original)
    predictions.append(np.clip(base_prediction, 1, config['max_num']))
    
    # Better prediction (small perturbation)
    perturbation = np.random.randint(-2, 3, size=config['num_numbers'])
    better_pred = base_prediction + perturbation
    predictions.append(np.clip(better_pred, 1, config['max_num']))
    
    # Good prediction (larger perturbation)
    perturbation = np.random.randint(-3, 4, size=config['num_numbers'])
    good_pred = base_prediction + perturbation
    predictions.append(np.clip(good_pred, 1, config['max_num']))
    
    return predictions

def display_prediction(numbers, label):
    """Display a prediction with label"""
    st.markdown(f"""
    <div class='prediction-card'>
        <div class='prediction-label'>{label}</div>
    """, unsafe_allow_html=True)
    
    cols = st.columns(len(numbers))
    for col, num in zip(cols, numbers):
        with col:
            st.markdown(f"<div class='number-badge'>{num}</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

def train_and_save_model(df, config):
    """Function to train and save model with MAE loss"""
    # Preprocess the data
    X = df.iloc[:, 1:].values / config['max_num']  # Scale the data
    y = df.iloc[:, 1:].values / config['max_num']
    
    # Build the model (simple model for this example)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(config['num_numbers'], activation='sigmoid')
    ])
    
    # Compile the model with MAE loss
    model.compile(optimizer='adam', loss='mae', metrics=['mae'])
    
    # Train the model
    model.fit(X, y, epochs=50, batch_size=32)
    
    # Save the model
    model.save(config['model'])
    st.success(f"Model trained and saved as {config['model']}")
    return model

def update_and_predict(lottery_name):
    """Update data and make predictions"""
    config = LOTTERY_CONFIG[lottery_name]
    
    with st.spinner(f"Updating {lottery_name} data..."):
        # ... [keep existing scraping code] ...

     with st.spinner("Retraining model..."):
        try:
            model = tf.keras.models.load_model(config['model'], compile=False)
        except:
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', 
                                    input_shape=(config['num_numbers'],)),
                tf.keras.layers.Dense(config['num_numbers'])
            ])
        
        # Explicitly use MAE class for compilation
        model.compile(optimizer='adam', 
                    loss=tf.keras.losses.MeanAbsoluteError(),
                    metrics=[tf.keras.metrics.MeanAbsoluteError()])
        
        # ... [keep existing training code] ...
        model.save(config['model'])
    
    return True

# Update the prediction loading in main()
    
    # Repeat similar changes for Lotto 649 section


def main():
    st.title("Ontario Lottery Predictor")
    st.markdown("---")
    
    # Create tabs for different lotteries
    tab1, tab2 = st.tabs(["Lotto Max", "Lotto 649"])
    
    with tab1:
        st.header("Lotto Max Predictions")
        if st.button("Update Lotto Max"):
            if update_and_predict('Lotto Max'):
                st.success("Lotto Max updated successfully!")
        
        col1, col2 = st.columns(2)
        with col1:
            try:
                df = pd.read_csv(LOTTERY_CONFIG['Lotto Max']['csv'])
                last_numbers = df.iloc[-1, 1:].tolist()
                display_prediction(last_numbers, "Last Draw Numbers")
            except:
                st.warning("No Lotto Max data available")
        
        with col2:
            try:
                df = pd.read_csv(LOTTERY_CONFIG['Lotto Max']['csv'])
                model = tf.keras.models.load_model(
                LOTTERY_CONFIG['Lotto Max']['model'],
                compile=False
            )
                model.compile(optimizer='adam',
                        loss=tf.keras.losses.MeanAbsoluteError(),
                        metrics=[tf.keras.metrics.MeanAbsoluteError()])
            
                X = df.iloc[-10:, 1:].values / LOTTERY_CONFIG['Lotto Max']['max_num']
                pred = model.predict(X[np.newaxis, ...])[0]
                pred_numbers = np.clip(np.round(pred * LOTTERY_CONFIG['Lotto Max']['max_num']).astype(int), 1, 50)
            
                predictions = generate_predictions(pred_numbers, LOTTERY_CONFIG['Lotto Max'])
                labels = ["Best Prediction", "Better Prediction", "Good Prediction"]
            
                for label, pred in zip(labels, predictions):
                    display_prediction(pred, label)
                
            except Exception as e:
                st.warning(f"Prediction not available: {str(e)}")
    
    with tab2:
        st.header("Lotto 649 Predictions")
        if st.button("Update Lotto 649"):
            if update_and_predict('Lotto 649'):
                st.success("Lotto 649 updated successfully!")
        
        col1, col2 = st.columns(2)
        with col1:
            try:
                df = pd.read_csv(LOTTERY_CONFIG['Lotto 649']['csv'])
                last_numbers = df.iloc[-1, 1:].tolist()
                display_prediction(last_numbers, "Last Draw Numbers")
            except:
                st.warning("No Lotto 649 data available")
        
        with col2:
            try:
                df = pd.read_csv(LOTTERY_CONFIG['Lotto 649']['csv'])
                model = tf.keras.models.load_model(LOTTERY_CONFIG['Lotto 649']['model'])
                X = df.iloc[-1, 1:].values.astype(np.float32).reshape(1, -1) / LOTTERY_CONFIG['Lotto 649']['max_num']
                base_pred = model.predict(X)[0]
                base_numbers = np.round(base_pred * LOTTERY_CONFIG['Lotto 649']['max_num']).astype(int)
                
                predictions = generate_predictions(base_numbers, LOTTERY_CONFIG['Lotto 649'])
                labels = ["Best Prediction", "Better Prediction", "Good Prediction"]
                
                for label, pred in zip(labels, predictions):
                    display_prediction(pred, label)
                    
            except Exception as e:
                st.warning(f"Prediction not available: {str(e)}")
    
    st.markdown("---")
    st.markdown("**Note:** Predictions are based on historical patterns and should not be considered financial advice")

if __name__ == "__main__":
    main()



# Update the model compilation section in both lottery configurations
