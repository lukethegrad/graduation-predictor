import streamlit as st
import pandas as pd

st.title("🎧 GRADUATION* Streaming Predictor")

# ---- Step 1: File Upload ----
uploaded_file = st.file_uploader("📂 Upload your streaming CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📋 Raw Uploaded Data")
    st.dataframe(df.head())

    # ---- Step 2: Standardize and map column names ----
    original_cols = df.columns.tolist()
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    # Define mapping for known alternate names
    column_mapping = {
        "trackid": "track_id",
        "track_id": "track_id",
        "trackid_": "track_id",
        "song_id": "track_id",
        "song": "track_id",
        "track": "track_id",
        "streams": "daily_streams",
        "stream_count": "daily_streams",
        "daily_streams": "daily_streams",
        "date_uploaded": "date",
        "stream_date": "date",
        "timestamp": "date",
        "date": "date"
    }

    # Apply column name mapping
    df.rename(columns={col: column_mapping.get(col, col) for col in df.columns}, inplace=True)

    # Show mappings applied
    st.caption(f"🛠️ Standardized columns from: {original_cols} → {df.columns.tolist()}")

    # ---- Step 3: Validate Required Columns ----
    required_cols = {"track_id", "date", "daily_streams"}
    if not required_cols.issubset(df.columns):
        st.error(f"❌ Your file must include columns: {required_cols}")
    else:
        # ---- Step 4: Clean data ----
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df =
