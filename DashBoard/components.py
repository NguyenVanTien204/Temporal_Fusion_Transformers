from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, Sequence

import altair as alt
import pandas as pd
import streamlit as st

from model_registry import ModelInfo


def render_summary_cards(metrics: OrderedDict[str, str | int | float]) -> None:
    columns = st.columns(len(metrics))
    for column, (label, value) in zip(columns, metrics.items()):
        column.metric(label=label, value=value)


def render_model_overview(primary: ModelInfo, others: Iterable[ModelInfo]) -> None:
    st.subheader("Model overview")
    st.markdown(
        f"**Primary model:** {primary.display_name} — {primary.description}."
        " Inference is executed on CPU when CUDA is unavailable, even though the weights were trained on GPU."
    )
    additional = list(others)
    if additional:
        st.caption("Also plotting: " + ", ".join(model.display_name for model in additional))


def render_predictions_chart(chart_df: pd.DataFrame, store_order: Sequence[int]) -> None:
    st.subheader("Predicted sales by store")
    if chart_df.empty:
        st.info("No predictions to plot for the current selection.")
        return

    tidy = chart_df.copy()
    tidy["Store"] = tidy["Store"].astype(str)
    tidy["PredictedSales"] = tidy["PredictedSales"].astype(float)
    store_order_labels = [str(store_id) for store_id in store_order]

    chart = (
        alt.Chart(tidy)
        .mark_bar()
        .encode(
            x=alt.X("Store:N", sort=store_order_labels, title="Store"),
            y=alt.Y("PredictedSales:Q", title="Predicted sales"),
            color=alt.Color("model_name:N", title="Model"),
            tooltip=["model_name", "Store", alt.Tooltip("PredictedSales:Q", format=",.0f"), "ForecastDate"],
        )
        .properties(height=420)
    )
    st.altair_chart(chart, use_container_width=True)


def render_store_details(store_row: pd.Series | None, prediction_rows: pd.DataFrame) -> None:
    st.subheader("Store spotlight")
    if store_row is None:
        st.warning("The selected store does not exist in the test file.")
        return

    left, right = st.columns(2)

    feature_frame = (
        pd.DataFrame(store_row).reset_index().rename(columns={"index": "Feature", 0: "Value"})
    )
    left.markdown(f"**Store {int(store_row['Store'])} context**")
    left.dataframe(feature_frame, hide_index=True, use_container_width=True)

    if prediction_rows.empty:
        right.info("No predictions available for this store.")
    else:
        prediction_view = (
            prediction_rows[["model_name", "PredictedSales"]]
            .rename(columns={"model_name": "Model", "PredictedSales": "Predicted sales"})
        )
        right.markdown("**Model comparison**")
        right.dataframe(
            prediction_view.style.format({"Predicted sales": "{:.0f}"}),
            hide_index=True,
            use_container_width=True,
        )


def render_data_sample(df: pd.DataFrame, rows: int = 10) -> None:
    st.subheader("Raw data sample")
    st.dataframe(df.head(rows), use_container_width=True)
