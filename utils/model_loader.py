"""Load the selected model with caches keyed to the actual model files."""

import json
from pathlib import Path

import streamlit as st
import tensorflow as tf

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
ACTIVE_MODEL_PATH = MODELS_DIR / "TreeCategory_EfficientNetB0.h5"
ACTIVE_METADATA_PATH = MODELS_DIR / "TreeCategory_EfficientNetB0.metadata.json"
LEGACY_MODEL_PATH = MODELS_DIR / "Real_TreeClassifier.keras"
LEGACY_CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"


def _model_path() -> Path:
    path = ACTIVE_MODEL_PATH if ACTIVE_MODEL_PATH.exists() else LEGACY_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at {path}")
    return path


@st.cache_resource
def _load_model_cached(path: str, modified_time_ns: int):
    return tf.keras.models.load_model(path, compile=False)


def load_tree_model():
    path = _model_path()
    return _load_model_cached(str(path), path.stat().st_mtime_ns)


@st.cache_data
def _read_metadata(path: str, modified_time_ns: int):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_model_metadata():
    if ACTIVE_MODEL_PATH.exists() and ACTIVE_METADATA_PATH.exists():
        return _read_metadata(str(ACTIVE_METADATA_PATH), ACTIVE_METADATA_PATH.stat().st_mtime_ns)
    return {"uncertainty_threshold": 0.45}


@st.cache_data
def _read_class_names(path: str, modified_time_ns: int):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_class_names():
    if ACTIVE_MODEL_PATH.exists() and ACTIVE_METADATA_PATH.exists():
        metadata = _read_metadata(str(ACTIVE_METADATA_PATH), ACTIVE_METADATA_PATH.stat().st_mtime_ns)
        return metadata.get("class_names", [])
    if not LEGACY_CLASS_NAMES_PATH.exists():
        return ["leafs", "saplings", "trees", "unclear"]
    return _read_class_names(str(LEGACY_CLASS_NAMES_PATH), LEGACY_CLASS_NAMES_PATH.stat().st_mtime_ns)
