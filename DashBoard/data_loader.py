from __future__ import annotations

import pandas as pd
import streamlit as st

from config import RAW_DATA_PATH
from model_registry import MODEL_REGISTRY, ModelInfo

DATE_COLUMN_CANDIDATES = ("ForecastDate", "Date", "forecast_date", "date")


@st.cache_data(show_spinner=False)
def load_test_data() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_PATH, parse_dates=["Date"])
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values(["Date", "Store"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_model_predictions(model_key: str) -> pd.DataFrame:
    if model_key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model key: {model_key}")

    model_info: ModelInfo = MODEL_REGISTRY[model_key]
    df = pd.read_csv(model_info.result_path)

    date_column = next((col for col in DATE_COLUMN_CANDIDATES if col in df.columns), None)
    if date_column is not None:
        df[date_column] = pd.to_datetime(df[date_column])
        if date_column != "ForecastDate":
            df = df.rename(columns={date_column: "ForecastDate"})
    else:
        df["ForecastDate"] = pd.NaT

    return df.sort_values(["ForecastDate", "Store"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_all_predictions() -> pd.DataFrame:
    frames = []
    for key, info in MODEL_REGISTRY.items():
        frame = load_model_predictions(key).copy()
        frame["model_key"] = key
        frame["model_name"] = info.display_name
        frame["model_color"] = info.color
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def merge_predictions_with_test(model_key: str) -> pd.DataFrame:
    test_df = load_test_data()
    prediction_df = load_model_predictions(model_key)
    merged = test_df.merge(
        prediction_df,
        how="left",
        left_on=["Store", "Date"],
        right_on=["Store", "ForecastDate"],
        suffixes=("", "_pred"),
    )
    return merged


def get_store_snapshot(store_id: int) -> pd.Series | None:
    test_df = load_test_data()
    store_rows = test_df[test_df["Store"] == store_id].sort_values("Date", ascending=False)
    if store_rows.empty:
        return None
    return store_rows.iloc[0]


def get_store_predictions(store_id: int) -> pd.DataFrame:
    predictions = load_all_predictions()
    store_predictions = predictions[predictions["Store"] == store_id].copy()
    return store_predictions.sort_values("model_key")
