from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

from config import RESULTS_DIR


@dataclass(frozen=True)
class ModelInfo:
    key: str
    display_name: str
    description: str
    result_path: Path
    color: str
    trained_on_gpu: bool = True


MODEL_REGISTRY: Dict[str, ModelInfo] = {
    "gru": ModelInfo(
        key="gru",
        display_name="GRU",
        description="Gated Recurrent Unit sequence model trained on GPU",
        result_path=RESULTS_DIR / "gru_predictions.csv",
        color="#2E7D32",
    ),
    "lstm": ModelInfo(
        key="lstm",
        display_name="LSTM",
        description="Long Short-Term Memory network with attention",
        result_path=RESULTS_DIR / "lstm_predictions.csv",
        color="#1565C0",
    ),
    "nbeats": ModelInfo(
        key="nbeats",
        display_name="N-BEATS",
        description="Interpretable N-BEATS forecast architecture",
        result_path=RESULTS_DIR / "nbeats_predictions.csv",
        color="#6A1B9A",
    ),
    "tcn": ModelInfo(
        key="tcn",
        display_name="TCN",
        description="Temporal Convolutional Network trained with dilated blocks",
        result_path=RESULTS_DIR / "tcn_predictions.csv",
        color="#EF6C00",
    ),
}

MODEL_KEYS: Tuple[str, ...] = tuple(MODEL_REGISTRY.keys())


def get_model(key: str) -> ModelInfo:
    return MODEL_REGISTRY[key]


def iter_models(keys: Iterable[str] | None = None):
    ordered_keys = MODEL_KEYS if keys is None else tuple(keys)
    for model_key in ordered_keys:
        if model_key in MODEL_REGISTRY:
            yield MODEL_REGISTRY[model_key]
