import streamlit as st
import pandas as pd

st.title("🎧 GRADUATION* Streaming Predictor")

# ---- Step 1: File Upload ----
uploaded_file = st.file_uploader("📂 Upload your streaming CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📋 Raw Uploaded Data")
    st.dataframe(df.head())

    # ---- Step 2: Standardize Column Names ----
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    # Required columns
    required_cols = {"track_id", "date", "daily_streams"}
    if not required_cols.issubset(df.columns):
        st.error(f"CSV must contain the following columns: {required_cols}")
    else:
        # ---- Step 3: Parse and clean date column ----
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "daily_streams", "track_id"])

        # ---- Step 4: Normalize timeline per track and interpolate missing days ----
        cleaned_list = []

        for track_id, track_df in df.groupby("track_id"):
            track_df = track_df.sort_values("date").copy()

            # Create full daily date range from first to last stream date
            full_range = pd.date_range(start=track_df["date"].min(), end=track_df["date"].max())

            # Set date as index and reindex to full daily range
            track_df = track_df.set_index("date").reindex(full_range)

            # Restore necessary columns
            track_df["track_id"] = track_id

            # Interpolate missing daily stream values
            track_df["daily_streams"] = track_df["daily_streams"].interpolate(method="linear")

            # Reset index back to date
            track_df = track_df.reset_index().rename(columns={"index": "date"})

            # Add day index (Day 1 = first streaming day)
            track_df["day"] = range(1, len(track_df) + 1)

            cleaned_list.append(track_df)

        # Combine all cleaned tracks into one DataFrame
        df_cleaned = pd.concat(cleaned_list, ignore_index=True)

        st.subheader("🧼 Cleaned & Normalized Data")
        st.dataframe(df_cleaned.head(20))

        # Optional: Download cleaned CSV
        csv_download = df_cleaned.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Cleaned CSV", csv_download, file_name="cleaned_streaming_data.csv")

        # Store for later use
        st.session_state["cleaned_data"] = df_cleaned

