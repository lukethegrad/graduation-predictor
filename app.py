from predict import prepare_sequence_for_prediction, predict_all_quantiles

import streamlit as st
import pandas as pd

st.title("🎧 GRADUATION* Streaming Predictor")

uploaded_file = st.file_uploader("📂 Upload your streaming CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📋 Raw Uploaded Data")
    st.dataframe(df.head())

    original_cols = df.columns.tolist()
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    st.caption(f"🔍 Standardized columns: {original_cols} → {df.columns.tolist()}")

    if {"artist", "title", "streams", "date"}.issubset(df.columns):
        st.caption("📦 Detected raw distributor format — cleaning now...")
        df["track_id"] = df["artist"].str.strip() + " - " + df["title"].str.strip()
        df.rename(columns={"streams": "daily_streams"}, inplace=True)
        df = df[["track_id", "date", "daily_streams"]]

    column_mapping = {
        "trackid": "track_id", "track": "track_id", "song": "track_id", "trackname": "track_id",
        "plays": "daily_streams", "stream_count": "daily_streams", "streams": "daily_streams", "daily": "daily_streams",
        "date_uploaded": "date", "stream_date": "date", "timestamp": "date"
    }
    df.rename(columns={col: column_mapping.get(col, col) for col in df.columns}, inplace=True)

    if {"date", "daily_streams"}.issubset(df.columns) and "track_id" not in df.columns:
        st.caption("🧩 Detected simplified single-track format — adding placeholder track_id.")
        df["track_id"] = "uploaded_track_1"

    required_cols = {"track_id", "date", "daily_streams"}
    if not required_cols.issubset(df.columns):
        st.error(f"❌ Could not find required columns after cleaning. Found: {df.columns.tolist()}")
    else:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "daily_streams", "track_id"])

        cleaned_list = []
        for track_id, track_df in df.groupby("track_id"):
            track_df = track_df.sort_values("date").copy()
            track_df = track_df[["date", "daily_streams"]]

            valid_rows = track_df[track_df["daily_streams"].notna() & (track_df["daily_streams"] > 0)]
            if valid_rows.empty:
                continue

            first_row = valid_rows.iloc[0]
            first_date = first_row["date"]
            first_value = first_row["daily_streams"]
            last_date = track_df["date"].max()

            full_dates = pd.DataFrame({"date": pd.date_range(start=first_date, end=last_date)})
            merged = pd.merge(full_dates, track_df, on="date", how="left")

            merged.loc[merged["date"] == first_date, "daily_streams"] = first_value
            merged["daily_streams"] = merged["daily_streams"].interpolate(method="linear")
            merged["track_id"] = track_id
            merged["day"] = range(1, len(merged) + 1)

            cleaned_list.append(merged)

        df_cleaned = pd.concat(cleaned_list, ignore_index=True)

        # 👇 PREDICTION SECTION STARTS HERE
        st.subheader("📈 Predicting Total Streams (P10 / P50 / P90)")
        sequence, current_total = prepare_sequence_for_prediction(df_cleaned)

        if sequence is None:
            st.warning("⚠️ Not enough data to make a prediction. Please upload at least 14 days of data.")
        else:
            prediction = predict_all_quantiles(sequence, current_total)
            if prediction:
                horizons = [14, 30, 90, 180, 365]
                result_df = pd.DataFrame({
                    "Horizon (days)": horizons,
                    "P10 Prediction": prediction["P10"].astype(int),
                    "P50 Prediction": prediction["P50"].astype(int),
                    "P90 Prediction": prediction["P90"].astype(int),
                })

                st.dataframe(result_df)
                st.download_button("📥 Download Predictions", result_df.to_csv(index=False), file_name="streaming_predictions.csv")

        # ✅ Show cleaned data below
        st.subheader("🧼 Cleaned & Normalized Data")
        st.dataframe(df_cleaned.head(20))

        csv_download = df_cleaned.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Cleaned CSV", csv_download, file_name="cleaned_streaming_data.csv")
        st.session_state["cleaned_data"] = df_cleaned
