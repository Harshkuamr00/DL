import streamlit as st
import torch
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from model import LightningLSTM
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler as SKMinMaxScaler

# Set page config
st.set_page_config(page_title="IBM Stock Analysis & Prediction", layout="wide")

# Load model and scalers
@st.cache_resource
def load_model():
    model = LightningLSTM()
    model.load_state_dict(torch.load("lstm_stock_model.pth"))
    model.eval()
    return model

@st.cache_resource
def load_scalers():
    # Create scalers for features and target
    df = pd.read_csv("daily_ibm_stock_data.csv")
    features = df[['open', 'high', 'low']].values
    target = df['close'].values
    
    feature_scaler = SKMinMaxScaler(feature_range=(0, 1))
    target_scaler = SKMinMaxScaler(feature_range=(0, 1))
    
    feature_scaler.fit(features)
    target_scaler.fit(target.reshape(-1, 1))
    
    return feature_scaler, target_scaler

@st.cache_data
def load_data():
    return pd.read_csv("daily_ibm_stock_data.csv")

model = load_model()
feature_scaler, target_scaler = load_scalers()
df = load_data()

# Title and intro
st.title("IBM Stock Price Analysis & Prediction")
st.markdown("---")

# Tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs(["Data Analysis", "Custom Predictions", "Performance Metrics", "Summary"])

with tab1:
    st.header("Stock Data Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Price", f"${df['close'].iloc[-1]:.2f}")
    with col2:
        st.metric("Min Price", f"${df['close'].min():.2f}")
    with col3:
        st.metric("Max Price", f"${df['close'].max():.2f}")
    with col4:
        st.metric("Avg Price", f"${df['close'].mean():.2f}")
    
    st.divider()
    
    # Price Trend Chart
    st.subheader("Price Trend Over Time")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['close'],
        mode='lines',
        name='Close Price',
        line=dict(color='#1f77b4', width=2)
    ))
    fig.update_layout(
        xaxis_title="Days",
        yaxis_title="Price ($)",
        hovermode='x unified',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # OHLC Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("OHLC Comparison")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df.index, df['open'], label='Open', alpha=0.7)
        ax.plot(df.index, df['high'], label='High', alpha=0.7)
        ax.plot(df.index, df['low'], label='Low', alpha=0.7)
        ax.plot(df.index, df['close'], label='Close', linewidth=2)
        ax.set_xlabel('Days')
        ax.set_ylabel('Price ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Price Distribution")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(df['close'], bins=50, color='#2ca02c', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Price ($)')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    # Volume Analysis
    st.subheader("Trading Volume")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['volume'],
        name='Volume',
        marker=dict(color='#ff7f0e')
    ))
    fig.update_layout(
        xaxis_title="Days",
        yaxis_title="Volume",
        height=300
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Daily Returns
    df['daily_return'] = df['close'].pct_change() * 100
    st.subheader("Daily Returns Analysis")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg Daily Return", f"{df['daily_return'].mean():.4f}%")
    with col2:
        st.metric("Max Daily Return", f"{df['daily_return'].max():.4f}%")
    with col3:
        st.metric("Volatility (Std Dev)", f"{df['daily_return'].std():.4f}%")
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.hist(df['daily_return'].dropna(), bins=50, color='#d62728', alpha=0.7, edgecolor='black')
    ax.axvline(df['daily_return'].mean(), color='black', linestyle='--', linewidth=2)
    ax.set_xlabel('Daily Return (%)')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

with tab2:
    st.header("Custom Stock Price Predictions")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Configuration")
        
        # Prediction days
        num_days = st.slider("Days to Predict", min_value=1, max_value=30, value=10, key="pred_days")
        
        # Sequence length
        sequence_length = st.slider("Sequence Length (days)", min_value=5, max_value=30, value=10, key="seq_len")
        
        # Custom sequence input method
        input_method = st.radio("Input Method", ["Use Recent Data", "Custom Manual Input"])
        
        custom_sequence = None
        if input_method == "Custom Manual Input":
            st.markdown("**Enter daily OHLC data** (one day per line)")
            st.caption("Format: open,high,low")
            sequence_text = st.text_area(
                "Custom Sequence",
                value="100,105,98\n101,106,99\n102,107,100",
                height=150,
                key="custom_seq"
            )
            try:
                lines = sequence_text.strip().split('\n')
                custom_sequence = np.array([list(map(float, line.split(','))) for line in lines if line.strip()])
                st.success(f"✓ Loaded {len(custom_sequence)} days of data")
            except:
                st.error("Invalid format. Use: open,high,low (comma-separated)")
                custom_sequence = None
        
        predict_btn = st.button("Generate Predictions", type="primary", use_container_width=True)
    
    with col2:
        if predict_btn:
            try:
                # Prepare sequence
                if input_method == "Use Recent Data":
                    if sequence_length > len(df):
                        st.error(f"Sequence length ({sequence_length}) exceeds available data ({len(df)} days)")
                    else:
                        features = df[['open', 'high', 'low']].values[-sequence_length:]
                else:
                    if custom_sequence is None or len(custom_sequence) < 5:
                        st.error("Please provide at least 5 days of custom data")
                        st.stop()
                    features = custom_sequence[-sequence_length:] if len(custom_sequence) >= sequence_length else custom_sequence
                
                # Scale features using feature_scaler
                scaled_features = feature_scaler.transform(features)
                
                # Make predictions
                predictions = []
                current_sequence = np.array(scaled_features)
                
                for _ in range(num_days):
                    input_tensor = torch.tensor(current_sequence[-sequence_length:], dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        pred = model(input_tensor).item()
                    predictions.append(pred)
                    new_row = np.append(current_sequence[-1][1:], pred)
                    current_sequence = np.vstack([current_sequence, new_row])
                
                # Inverse transform using target_scaler
                predictions_array = np.array(predictions).reshape(-1, 1)
                predictions_unscaled = target_scaler.inverse_transform(predictions_array).flatten()
                
                st.success(" Predictions generated successfully!")
                
                st.markdown("---")
                st.subheader(" Prediction Results")
                
                # Metrics
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                with col_m1:
                    st.metric("Min Predicted", f"${min(predictions_unscaled):.2f}")
                with col_m2:
                    st.metric("Max Predicted", f"${max(predictions_unscaled):.2f}")
                with col_m3:
                    st.metric("Avg Predicted", f"${np.mean(predictions_unscaled):.2f}")
                with col_m4:
                    st.metric("Latest Actual", f"${df['close'].iloc[-1]:.2f}")
                
                st.markdown("---")
                
                # Chart
                days = list(range(1, num_days + 1))
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(-20, 0)),
                    y=df['close'].tail(20).values,
                    mode='lines',
                    name='Historical',
                    line=dict(color='#1f77b4', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=days,
                    y=predictions_unscaled,
                    mode='lines+markers',
                    name='Predicted',
                    line=dict(color='#ff7f0e', width=2, dash='dash'),
                    marker=dict(size=8)
                ))
                fig.update_layout(
                    xaxis_title="Days",
                    yaxis_title="Price ($)",
                    hovermode='x unified',
                    height=450,
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Predictions Table
                st.subheader(" Detailed Predictions")
                pred_df = pd.DataFrame({
                    "Day": days,
                    "Predicted Price ($)": [f"${p:.2f}" for p in predictions_unscaled],
                    "Change from Prev (%)": [f"{((predictions_unscaled[i] - predictions_unscaled[i-1]) / predictions_unscaled[i-1] * 100):.2f}%" if i > 0 else "N/A" for i in range(len(predictions_unscaled))]
                })
                st.dataframe(pred_df, use_container_width=True, hide_index=True)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

with tab3:
    st.header("Model Performance Metrics")
    
    # Evaluate model on validation set
    features = df[['open', 'high', 'low']].values
    target = df['close'].values
    
    scaled_features = feature_scaler.transform(features)
    scaled_target = target_scaler.transform(target.reshape(-1, 1)).flatten()
    
    # Create sequences
    sequence_length = 10
    X, y = [], []
    for i in range(len(scaled_features) - sequence_length):
        X.append(scaled_features[i:i+sequence_length])
        y.append(scaled_target[i+sequence_length])
    
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)
    
    # Split data
    train_size = len(X) - 10
    X_val = X[train_size:]
    y_val = y[train_size:]
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        y_val_pred = model(X_val).squeeze().numpy()
    
    y_val_pred_actual = target_scaler.inverse_transform(y_val_pred.reshape(-1, 1)).flatten()
    y_val_actual = target_scaler.inverse_transform(y_val.numpy().reshape(-1, 1)).flatten()
    
    # Calculate metrics
    mae = mean_absolute_error(y_val_actual, y_val_pred_actual)
    mse = mean_squared_error(y_val_actual, y_val_pred_actual)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val_actual, y_val_pred_actual)
    mape = np.mean(np.abs((y_val_actual - y_val_pred_actual) / y_val_actual)) * 100
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("MAE", f"${mae:.4f}")
    with col2:
        st.metric("MSE", f"{mse:.4f}")
    with col3:
        st.metric("RMSE", f"${rmse:.4f}")
    with col4:
        st.metric("R² Score", f"{r2:.4f}")
    with col5:
        st.metric("MAPE", f"{mape:.2f}%")
    
    st.divider()
    
    # Prediction vs Actual Chart
    st.subheader("Validation: Predicted vs Observed (Last 10 Days)")
    fig = go.Figure()
    days = list(range(1, len(y_val_actual) + 1))
    fig.add_trace(go.Scatter(
        x=days, y=y_val_actual,
        mode='lines+markers',
        name='Observed',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=8)
    ))
    fig.add_trace(go.Scatter(
        x=days, y=y_val_pred_actual,
        mode='lines+markers',
        name='Predicted',
        line=dict(color='#ff7f0e', width=2),
        marker=dict(size=8)
    ))
    fig.update_layout(
        xaxis_title="Days",
        yaxis_title="Close Price ($)",
        hovermode='x unified',
        height=400,
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Residuals
    st.subheader("Prediction Errors (Residuals)")
    residuals = y_val_actual - y_val_pred_actual
    
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=days,
            y=residuals,
            mode='lines+markers',
            name='Residuals',
            line=dict(color='#d62728', width=2),
            marker=dict(size=8)
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        fig.update_layout(
            xaxis_title="Prediction Index",
            yaxis_title="Error ($)",
            height=350,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(residuals, bins=30, color='#2ca02c', alpha=0.7, edgecolor='black')
        ax.axvline(residuals.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {residuals.mean():.3f}')
        ax.set_xlabel('Error ($)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

with tab4:
    st.header("📋 Complete Analysis Summary")
    
    summary_data = {
        'Metric': [
            'Total Trading Days',
            'Min Close Price',
            'Max Close Price',
            'Average Price',
            'Price Volatility (Std Dev)',
            'Current Close Price',
            'Avg Daily Return',
            '',
            'Model MAE',
            'Model RMSE',
            'R² Score',
            'MAPE'
        ],
        'Value': [
            f"{len(df)}",
            f"${df['close'].min():.2f}",
            f"${df['close'].max():.2f}",
            f"${df['close'].mean():.2f}",
            f"${df['close'].std():.2f}",
            f"${df['close'].iloc[-1]:.2f}",
            f"{df['daily_return'].mean():.4f}%",
            '',
            f"${mae:.4f}",
            f"${rmse:.4f}",
            f"{r2:.4f}",
            f"{mape:.2f}%"
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    st.markdown("""
    ### Key Points 
    - **Historical Data**: Shows the pattern of IBM stock over time
    - **Volatility**: Indicates price stability - higher values mean more fluctuation
    - **Daily Returns**: Average percentage change helping identify trends
    - **Predictions**: Based on LSTM neural network analyzing historical patterns
    - **Model Performance**: MAE, RMSE, R², and MAPE indicate prediction accuracy
    """
                )