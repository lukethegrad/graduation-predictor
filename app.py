import streamlit as st
import pandas as pd

st.title("🎧 GRADUATION* Streaming Predictor")

# ---- Step 1: File Upload ----
uploaded_file = st.file_uploader("📂 Upload your streaming CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📋 Raw Uploaded Data")
    st.dataframe(df.head())

    # ---- Step 2: Auto-detect and clean known stream report formats ----
    # If file has Artist, Title, Streams, Date — assume raw format
    raw_format_cols = {"artist", "title", "streams", "date"}
    normalized_cols = [col.strip().lower().replace(" ", "_") for col in df.columns]
    df.columns = normalized_cols

    if raw_format_cols.issubset(set(normalized_cols)):
        st.caption("📦 Detected raw distributor format. Auto-cleaning...")

        # Combine artist and title into a track_id
        df["track_id"] = df["artist"].str.strip() + " - " + df["title"].str.strip()

        # Rename streams and date
        df.rename(columns={"streams": "daily_streams"}, inplace=True)

        # Drop other irrelevant columns
        keep_cols = ["track_id", "date", "daily_streams"]
        df = df[keep_cols]

    # ---- Step 3: Normalize columns ----
    column_mapping = {
        "trackid": "track_id",
        "track": "track_id",
        "song": "track_id",
        "trackname": "track_id",
        "plays": "daily_streams",
        "stream_count": "daily_streams",
        "streams": "daily_streams",
        "date_uploaded": "date",
        "stream_date": "date",
        "timestamp": "date"
    }
    df.rename(columns={col: column_mapping.get(col, col) for col in df.columns}, inplace=True)

    # Validate required columns
    required_cols = {"track_id", "date", "daily_streams"}
    if not required_cols.issubset(df.columns):
        st.error(f"❌ Your file must include columns: {required_cols}")
    else:
        # ---- Step 4: Clean types ----
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "daily_streams", "track_id"])

        # ---- Step 5: Normalize + fill gaps ----
        cleaned_list = []
        for track_id, track_df in df.groupby("track_id"):
            track_df = track_df.sort_values("date").copy()
            full_range = pd.date_range(start=track_df["date"].min(), end=track_df["date"].max())
            track_df = track_df.set_index("date").reindex(full_range)
            track_df["track_id"] = track_id
            track_df["daily_streams"] = track_df["daily_streams"].interpolate(method="linear")
            track_df = track_df.reset_index().rename(columns={"index": "date"})
            track_df["day"] = range(1, len(track_df) + 1)
            cleaned_list.append(track_df)

        df_cleaned = pd.concat(cleaned_list, ignore_index=True)

        st.subheader("🧼 Cleaned & Normalized Data")
        st.dataframe(df_cleaned.head(20))

        csv_download = df_cleaned.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Cleaned CSV", csv_download, file_name="cleaned_streaming_data.csv")

        st.session_state["cleaned_data"] = df_cleaned
