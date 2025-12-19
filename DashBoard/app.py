from __future__ import annotations

from collections import OrderedDict
from typing import List

import pandas as pd
import streamlit as st

import components
from config import DEFAULT_TOP_N, MAX_TOP_N, PAGE_ICON, PAGE_TITLE
from data_loader import (
    get_store_predictions,
    get_store_snapshot,
    load_all_predictions,
    load_test_data,
    merge_predictions_with_test,
)
from model_registry import MODEL_KEYS, MODEL_REGISTRY, get_model, iter_models


def _summary_metrics(test_df: pd.DataFrame) -> OrderedDict[str, str]:
    summary = OrderedDict()
    summary["Stores"] = f"{test_df['Store'].nunique():,}"
    summary["Avg customers"] = f"{int(test_df['Customers'].mean()):,}"
    summary["Promo share"] = f"{test_df['Promo'].mean() * 100:.1f}%"
    summary["Open ratio"] = f"{test_df['Open'].mean() * 100:.1f}%"
    summary["School holiday"] = f"{test_df['SchoolHoliday'].mean() * 100:.1f}%"
    return summary


def _select_store_options(test_df: pd.DataFrame) -> List[int]:
    return sorted(test_df["Store"].unique().tolist())


def _build_top_store_list(predictions: pd.DataFrame, model_key: str, limit: int) -> List[int]:
    model_rows = predictions[predictions["model_key"] == model_key]
    if model_rows.empty:
        return []
    top_rows = model_rows.nlargest(limit, "PredictedSales")
    return top_rows["Store"].astype(int).tolist()


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")
    st.title(PAGE_TITLE)
    st.caption("Demo dashboard comparing GPU-trained deep learning models on the shared test.csv slice.")

    test_df = load_test_data()
    prediction_df = load_all_predictions()
    store_options = _select_store_options(test_df)

    summary = _summary_metrics(test_df)
    components.render_summary_cards(summary)

    sidebar = st.sidebar
    sidebar.header("Controls")

    primary_model_key = sidebar.selectbox(
        "Primary model",
        options=MODEL_KEYS,
        format_func=lambda key: MODEL_REGISTRY[key].display_name,
    )

    comparison_candidates = [key for key in MODEL_KEYS if key != primary_model_key]
    default_compare = comparison_candidates[:2]
    comparison_models = sidebar.multiselect(
        "Compare with",
        options=comparison_candidates,
        default=default_compare,
        format_func=lambda key: MODEL_REGISTRY[key].display_name,
    )

    max_top_candidate = max(1, min(MAX_TOP_N, len(store_options)))
    min_top = 1 if max_top_candidate < 5 else 5
    step = 1 if max_top_candidate < 5 else 5
    top_n = sidebar.slider(
        "Top N stores",
        min_value=min_top,
        max_value=max_top_candidate,
        value=min(DEFAULT_TOP_N, max_top_candidate),
        step=step,
    )

    focus_store = sidebar.selectbox(
        "Focus store",
        options=store_options,
        format_func=lambda value: f"Store {value}",
    )

    show_raw = sidebar.toggle("Show raw data sample", value=False)

    chart_model_keys = list(dict.fromkeys([primary_model_key] + comparison_models))
    top_store_ids = _build_top_store_list(prediction_df, primary_model_key, top_n)

    components.render_model_overview(
        primary=get_model(primary_model_key),
        others=list(iter_models(chart_model_keys[1:])),
    )

    chart_df = prediction_df[
        prediction_df["model_key"].isin(chart_model_keys)
        & prediction_df["Store"].isin(top_store_ids)
    ].copy()

    components.render_predictions_chart(chart_df, store_order=top_store_ids)

    st.subheader("Top stores table")
    merged_primary = merge_predictions_with_test(primary_model_key)
    table_view = merged_primary[merged_primary["Store"].isin(top_store_ids)].copy()
    if table_view.empty:
        st.info("No rows available for the selected configuration.")
    else:
        table_view = table_view[
            [
                "Store",
                "Date",
                "DayOfWeek",
                "Customers",
                "Open",
                "Promo",
                "StateHoliday",
                "SchoolHoliday",
                "PredictedSales",
            ]
        ].sort_values("PredictedSales", ascending=False)
        table_view["Date"] = table_view["Date"].dt.date
        st.dataframe(table_view, hide_index=True, use_container_width=True)

    store_row = get_store_snapshot(focus_store)
    store_predictions = get_store_predictions(focus_store)
    components.render_store_details(store_row, store_predictions)

    if show_raw:
        components.render_data_sample(test_df)


if __name__ == "__main__":
    main()
