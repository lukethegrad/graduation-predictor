import streamlit as st
import pandas as pd

st.title("🎧 GRADUATION* Streaming Predictor")

# ---- Step 1: Upload file ----
uploaded_file = st.file_uploader("📂 Upload your streaming CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📋 Raw Uploaded Data")
    st.dataframe(df.head())

    # ---- Step 2: Normalize column names ----
    original_cols = df.columns.tolist()
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    st.caption(f"🔍 Standardized columns: {original_cols} → {df.columns.tolist()}")

    # ---- Step 3: Detect raw distributor format ----
    if {"artist", "title", "streams", "date"}.issubset(df.columns):
        st.caption("📦 Detected raw distributor format — cleaning now...")
        df["track_id"] = df["artist"].str.strip() + " - " + df["title"].str.strip()
        df.rename(columns={"streams": "daily_streams"}, inplace=True)
        df = df[["track_id", "date", "daily_streams"]]

    # ---- Step 4: Alias mapping for other cases ----
    column_mapping = {
        "trackid": "track_id",
        "track": "track_id",
        "song": "track_id",
        "trackname": "track_id",
        "plays": "daily_streams",
        "stream_count": "daily_streams",
        "streams": "daily_streams",
        "daily": "daily_streams",
        "date_uploaded": "date",
        "stream_date": "date",
        "timestamp": "date"
    }
    df.rename(columns={col: column_mapping.get(col, col) for col in df.columns}, inplace=True)

    # ---- Step 5: Handle single-track file format ----
    if {"date", "daily_streams"}.issubset(df.columns) and "track_id" not in df.columns:
        st.caption("🧩 Detected simplified single-track format — adding placeholder track_id.")
        df["track_id"] = "uploaded_track_1"

    # ---- Step 6: Validate required columns ----
    required_cols = {"track_id", "date", "daily_streams"}
    if not required_cols.issubset(df.columns):
        st.error(f"❌ Could not find required columns after cleaning. Found: {df.columns.tolist()}")
    else:
        # ---- Step 7: Clean and normalize data ----
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "daily_streams", "track_id"])

        cleaned_list = []
        for track_id, track_df in df.groupby("track_id"):
            track_df = track_df.sort_values("date").copy()

            # Drop all rows before the first known streaming day
            track_df = track_df.dropna(subset=["daily_streams"])

            # Full daily date range from first real streaming day to last
            full_range = pd.date_range(start=track_df["date"].min(), end=track_df["date"].max())

            # Reindex to fill missing days between first and last streaming dates
            track_df = track_df.set_index("date").reindex(full_range)
            track_df["track_id"] = track_id
            track_df["daily_streams"] = track_df["daily_streams"].interpolate(method="linear")

            # Reset index and assign day number
            track_df = track_df.reset_index().rename(columns={"index": "date"})
            track_df["day"] = range(1, len(track_df) + 1)

            cleaned_list.append(track_df)

        df_cleaned = pd.concat(cleaned_list, ignore_index=True)

        st.subheader("🧼 Cleaned & Normalized Data")
        st.dataframe(df_cleaned.head(20))

        csv_download = df_cleaned.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Cleaned CSV", csv_download, file_name="cleaned_streaming_data.csv")
        st.session_state["cleaned_data"] = df_cleaned
