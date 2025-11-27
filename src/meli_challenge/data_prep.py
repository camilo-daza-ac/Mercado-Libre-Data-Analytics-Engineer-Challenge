"""Data preparation utilities for the Mercado Libre challenge.

This module centralizes every transformation originally prototyped inside
`notebooks/EDA.ipynb`, so that both notebooks and scripts can reuse the same
logic when cleaning the raw CSV.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_raw_dataset(filename: str = "df_challenge_meli.csv") -> pd.DataFrame:
    """Load the raw CSV shipped with the challenge."""

    path = RAW_DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {path}")
    return pd.read_csv(path)


def clean_price_and_stock(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the business rules for price/stock cleaning.

    Returns a tuple with (clean_df, outliers_df).
    """

    df_price = df.dropna(subset=["price"]).copy()
    price_p99 = df_price["price"].quantile(0.99)
    price_mask = (df_price["price"] > 0) & (df_price["price"] <= price_p99)

    outliers = df_price[~price_mask].copy()
    df_clean = df_price[price_mask].copy()

    # Stock tail normalization (p95)
    stock_p95 = df_clean["stock"].quantile(0.95)
    stock_max = df_clean["stock"].max()

    def normalize_tail(val: float) -> float:
        if val <= stock_p95:
            return val
        return stock_p95 + ((val - stock_p95) / (stock_max - stock_p95)) * stock_p95

    df_clean["stock_norm"] = df_clean["stock"].apply(normalize_tail)
    return df_clean, outliers


def impute_seller_reputation(df: pd.DataFrame) -> pd.DataFrame:
    """Fill seller_reputation nulls using the most common value per nickname."""

    rep_map = (
        df[df["seller_reputation"].notna()]
        .groupby("seller_nickname")["seller_reputation"]
        .agg(lambda x: x.mode().iloc[0])
    )

    df["seller_reputation"] = df["seller_reputation"].fillna(
        df["seller_nickname"].map(rep_map)
    )
    df["seller_reputation"] = df["seller_reputation"].fillna("unknown")
    return df


def save_processed(df: pd.DataFrame, outliers: pd.DataFrame) -> None:
    """Persist the curated dataset and outliers to the processed directory."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIR / "df_curated.csv", index=False)
    outliers.to_csv(PROCESSED_DIR / "outliers_price.csv", index=False)

def save_segmented_dataset(df: pd.DataFrame, filename: str = "df_clustered.csv") -> None:
    """Persist the curated dataset to the processed directory."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    # df = df[["seller_nickname", "performance_level", "performance_segment"]]
    df.to_csv(PROCESSED_DIR / filename, index=False)


def run_full_preparation() -> pd.DataFrame:
    """Convenience wrapper used by scripts/notebooks."""

    df_raw = load_raw_dataset()
    df_clean, outliers = clean_price_and_stock(df_raw)
    df_clean = impute_seller_reputation(df_clean)
    save_processed(df_clean, outliers)
    return df_clean

