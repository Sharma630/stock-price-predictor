"""
Shared helpers for the Streamlit pages: cached data loading + a consistent
way to surface friendly error messages instead of raw tracebacks.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Tuple

import pandas as pd
import streamlit as st

from config import settings
from src.data_loader import load_market_data
from src.preprocessing import clean_ohlcv
from src.feature_engineering import build_feature_dataframe
from src.utils import (
    DataUnavailableError,
    InsufficientDataError,
    InvalidConfigurationError,
    ModelNotFoundError,
)


def init_session_state() -> None:
    """
    Initialize shared selections in session_state so every page has valid
    defaults, regardless of which page the user lands on first. Streamlit
    runs each page as an independent script, so this must be called at the
    top of every page (not just the home page) before any widget or lookup
    depends on these keys.
    """
    defaults = {
        "index_key": settings.DEFAULT_INDEX_KEY,
        "start_date": date.today() - timedelta(days=365 * settings.DEFAULT_HISTORY_YEARS),
        "end_date": date.today(),
        "lookback": settings.DEFAULT_LOOKBACK,
        "epochs": settings.DEFAULT_EPOCHS,
        "batch_size": settings.DEFAULT_BATCH_SIZE,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar() -> None:
    """
    Render the shared sidebar controls (index / date range / lookback).
    Must be called on every page (after init_session_state()) so the
    selection widgets and their session_state values are always available,
    no matter which page the user opens first.
    """
    init_session_state()

    st.sidebar.title("📈 Forecasting Controls")

    labels = settings.get_index_options_labels()
    index_keys = list(labels.keys())
    selected_label = st.sidebar.selectbox(
        "Select Market Index",
        options=[labels[k] for k in index_keys],
        index=index_keys.index(st.session_state["index_key"]),
    )
    st.session_state["index_key"] = index_keys[[labels[k] for k in index_keys].index(selected_label)]

    st.sidebar.date_input("Start Date", key="start_date")
    st.sidebar.date_input("End Date", key="end_date")

    st.sidebar.slider(
        "Lookback Window (trading days)",
        min_value=20,
        max_value=120,
        step=5,
        key="lookback",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Navigation**\n\n"
        "Use the pages in the left menu:\n"
        "1. Dashboard\n"
        "2. Technical Analysis\n"
        "3. Model Training\n"
        "4. Forecasting\n"
        "5. Model Performance\n"
        "6. About Project"
    )
    st.sidebar.markdown("---")
    st.sidebar.caption(settings.EDUCATIONAL_DISCLAIMER)


@st.cache_data(show_spinner="Downloading / loading market data...")
def cached_load_raw_data(index_key: str, start_date: date, end_date: date):
    """
    Cached wrapper around load_market_data so Streamlit reruns (e.g. from
    widget interaction) don't re-download the full dataset every time.
    Returns (dataframe, used_sample_data, source_message).
    """
    result = load_market_data(index_key, start_date, end_date)
    return result.data, result.used_sample_data, result.source_message


@st.cache_data(show_spinner="Calculating technical indicators...")
def cached_feature_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    clean_df = clean_ohlcv(raw_df)
    return build_feature_dataframe(clean_df)


def require_selection() -> Tuple[str, date, date, int]:
    """
    Read the shared selections from session_state. Callers must have
    already called render_sidebar() earlier on the same page so these
    keys are guaranteed to exist.
    """
    index_key = st.session_state.get("index_key", settings.DEFAULT_INDEX_KEY)
    start_date = st.session_state.get("start_date")
    end_date = st.session_state.get("end_date")
    lookback = st.session_state.get("lookback", settings.DEFAULT_LOOKBACK)
    return index_key, start_date, end_date, lookback


def load_data_with_feedback(index_key: str, start_date: date, end_date: date):
    """
    Load + validate market data, showing a friendly Streamlit error (instead
    of a raw traceback) and stopping the page on failure. Returns the feature
    dataframe (OHLCV + all indicators) on success.
    """
    try:
        raw_df, used_sample, source_message = cached_load_raw_data(index_key, start_date, end_date)
        if used_sample:
            st.warning(f"⚠️ {source_message} Live download was unavailable, so results below use bundled demo data.")
        feature_df = cached_feature_dataframe(raw_df)
        return raw_df, feature_df
    except InvalidConfigurationError as exc:
        st.error(f"Invalid configuration: {exc}")
        st.stop()
    except DataUnavailableError as exc:
        st.error(f"Data unavailable: {exc}")
        st.stop()
    except InsufficientDataError as exc:
        st.error(f"Not enough data: {exc}")
        st.stop()
    except Exception as exc:  # last-resort guard so users never see a raw traceback
        st.error(f"An unexpected error occurred while loading data: {exc}")
        st.stop()


def guarded(fn, *args, error_prefix: str = "An error occurred", **kwargs):
    """Run fn(*args, **kwargs) and convert known exceptions into friendly Streamlit errors."""
    try:
        return fn(*args, **kwargs)
    except ModelNotFoundError as exc:
        st.warning(str(exc))
        st.stop()
    except (InsufficientDataError, InvalidConfigurationError, DataUnavailableError) as exc:
        st.error(f"{error_prefix}: {exc}")
        st.stop()
    except Exception as exc:
        st.error(f"{error_prefix}: {exc}")
        st.stop()
