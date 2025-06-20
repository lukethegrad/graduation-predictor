import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load models (must be saved under /models directory)
model_q10 = load_model("models/model_q10.keras", compile=False)
model_q50 = load_model("models/model_q50.keras", compile=False)
model_q90 = load_model("models/model_q90.keras", compile=False)

# Re-create scaler used during training (fit on input sequence itself for now)
def scale_sequences(sequences):
    flat = sequences.reshape(-1, sequences.shape[-1])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(flat).reshape(sequences.shape)
    return scaled

def inverse_scale_targets(log_y):
    return np.expm1(log_y)

# Prepares latest 14-day sequence from uploaded DataFrame
def prepare_sequence_for_prediction(df, sequence_length=14):
    df = df.sort_values("date").reset_index(drop=True)

    # Feature engineering
    epsilon = 1e-6
    df["streams_7_days_ago"] = df["daily_streams"].shift(7).fillna(0)
    df["week_over_week_growth"] = (df["daily_streams"] - df["streams_7_days_ago"]) / (df["streams_7_days_ago"] + epsilon)

    df["streams_3d_sum"] = df["daily_streams"].rolling(window=3, min_periods=1).sum()
    df["streams_3d_sum_3d_ago"] = df["streams_3d_sum"].shift(3).fillna(0)
    df["growth_3d_over_3d"] = (df["streams_3d_sum"] - df["streams_3d_sum_3d_ago"]) / (df["streams_3d_sum_3d_ago"] + epsilon)

    df["streams_7d_sum"] = df["daily_streams"].rolling(window=7, min_periods=1).sum()
    df["streams_7d_sum_7d_ago"] = df["streams_7d_sum"].shift(7).fillna(0)
    df["growth_7d_over_7d"] = (df["streams_7d_sum"] - df["streams_7d_sum_7d_ago"]) / (df["streams_7d_sum_7d_ago"] + epsilon)

    df["cumulative_streams"] = df["daily_streams"].cumsum()
    df["mean_3d_streams"] = df["daily_streams"].rolling(window=3, min_periods=1).mean()
    df["mean_7d_streams"] = df["daily_streams"].rolling(window=7, min_periods=1).mean()
    df["daily_change"] = df["daily_streams"].diff().fillna(0)
    df["daily_acceleration"] = df["daily_change"].diff().fillna(0)

    df = df.fillna(0)

    feature_cols = [
        "daily_streams", "week_over_week_growth", "growth_3d_over_3d", "growth_7d_over_7d",
        "cumulative_streams", "mean_3d_streams", "mean_7d_streams", "daily_change", "daily_acceleration"
    ]

    if len(df) < sequence_length:
        return None, None

    seq = df[feature_cols].values[-sequence_length:]
    cumulative_today = df["cumulative_streams"].iloc[-1]

    return seq, cumulative_today

# Predict all quantiles and add real cumulative
def predict_all_quantiles(sequence, cumulative_today):
    if sequence is None:
        return None

    sequence = np.expand_dims(sequence, axis=0)  # shape (1, 14, features)
    sequence = scale_sequences(sequence)

    q10 = inverse_scale_targets(model_q10.predict(sequence)[0])
    q50 = inverse_scale_targets(model_q50.predict(sequence)[0])
    q90 = inverse_scale_targets(model_q90.predict(sequence)[0])

    # Add cumulative streams so far
    q10 += cumulative_today
    q50 += cumulative_today
    q90 += cumulative_today

    return {
        "P10": q10,
        "P50": q50,
        "P90": q90
    }
