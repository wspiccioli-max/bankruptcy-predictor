import json
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent.parent
METADATA_PATH = ROOT / "data/processed/training_metadata.json"


@st.cache_data
def load_training_metadata() -> dict:
    if not METADATA_PATH.exists():
        return {}
    return json.loads(METADATA_PATH.read_text())


def methodology_caption() -> str:
    meta = load_training_metadata()
    if not meta:
        return "Training metadata unavailable. S&P 500 scoring model: Logistic Regression."
    return (
        f"Training sample: {meta['training_sample_size']} "
        f"({meta['filing_positive_count']} filing-positive, "
        f"{meta['control_count']} controls) · "
        f"Validation groups: {meta.get('unique_validation_groups', 'n/a')} "
        f"by {meta.get('validation_group_column', 'ticker')} · "
        f"S&P 500 scoring model: {meta['scoring_model']} · "
        f"Trained: {meta['trained_at_utc']}"
    )
