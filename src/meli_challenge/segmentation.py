"""Seller-level feature engineering and clustering utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_curated_dataset(filename: str = "df_curated.csv") -> pd.DataFrame:
    """Load the cleaned dataset produced by ``data_prep``."""

    path = PROCESSED_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Curated dataset not found at {path}")
    return pd.read_csv(path)


import pandas as pd


def build_seller_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye una tabla agregada a nivel seller (`seller_table`) a partir del
    DataFrame de ítems.

    Espera al menos las columnas:
    - seller_nickname
    - titulo
    - stock_norm
    - logistic_type
    - price
    - category_id
    - condition
    - seller_reputation
    """

    df = df.copy()

    # Valor por ítem para evitar depender del df externo en el groupby
    df["item_value"] = df["price"] * df["stock_norm"]

    # 1. Tamaño e intensidad
    seller_metrics = (
        df.groupby("seller_nickname")
        .agg(
            n_items=("titulo", "count"),
            total_stock=("stock_norm", "sum"),
            logistic_type=("logistic_type", "first"),
            total_value=("item_value", "sum"),
        )
        .reset_index()
    )

    # 2. Diversificación vs especialización
    category_agg = (
        df.groupby("seller_nickname")
        .agg(
            n_categories=("category_id", pd.Series.nunique),
        )
        .reset_index()
    )

    def main_cat(gr: pd.DataFrame) -> pd.Series:
        most_common = gr["category_id"].value_counts().idxmax()
        pct_main = gr["category_id"].value_counts(normalize=True).max()
        return pd.Series(
            {
                "main_category": most_common,
                "pct_main_category": pct_main,
            }
        )

    diversity = (
        df.groupby("seller_nickname")
        .apply(main_cat)
        .reset_index()
    )

    # 3. Estructura de condición (nuevo/usado/refurb)
    def cond_pct(gr: pd.DataFrame) -> pd.Series:
        n = len(gr)
        pct_new = (gr["condition"] == "new").sum() / n
        pct_used = (gr["condition"] == "used").sum() / n
        pct_refurb = (gr["condition"] == "refurbished").sum() / n
        return pd.Series(
            {
                "pct_new": pct_new,
                "pct_used": pct_used,
                "pct_refurbished": pct_refurb,
            }
        )

    cond = (
        df.groupby("seller_nickname")
        .apply(cond_pct)
        .reset_index()
    )

    # 4. Posicionamiento de precios
    def price_stats(gr: pd.DataFrame) -> pd.Series:
        avg_price = gr["price"].mean()
        median_price = gr["price"].median()
        return pd.Series(
            {
                "avg_price_regular": avg_price,
                "median_price_regular": median_price,
            }
        )

    prices = (
        df.groupby("seller_nickname")
        .apply(price_stats)
        .reset_index()
    )

    # 5. Reputación
    reputation_map = {
        "green_platinum": 5,
        "green_gold": 4,
        "green": 4,
        "green_silver": 3,
        "yellow": 2,
        "light_green": 2,
        "red": 1,
        "orange": 1,
        "newbie": 0,
        "unknown": 0,
    }

    def rep_score(gr: pd.DataFrame) -> pd.Series:
        rep = gr["seller_reputation"].mode()[0]
        score = reputation_map.get(rep, 0)
        return pd.Series(
            {
                "seller_reputation": rep,
                "seller_reputation_score": score,
            }
        )

    reputation = (
        df.groupby("seller_nickname")
        .apply(rep_score)
        .reset_index()
    )

    # 6. Stock por ítem (sobre la tabla agregada)
    seller_metrics["avg_stock_per_item"] = (
        seller_metrics["total_stock"] / seller_metrics["n_items"]
    )

    # Merge de todas las métricas
    seller_table = (
        seller_metrics.merge(category_agg, on="seller_nickname")
        .merge(diversity, on="seller_nickname")
        .merge(cond, on="seller_nickname")
        .merge(prices, on="seller_nickname")
        .merge(reputation, on="seller_nickname")
    )

    return seller_table



def add_seller_size(
    df: pd.DataFrame,
    value_col: str = "total_value",
    quantiles: tuple[float, float, float] = (0.30, 0.60, 0.90),
) -> pd.DataFrame:
    """
    Clasifica el tamaño del seller usando percentiles de `value_col` (por defecto total_value)
    en:
        - Key Account
        - Core Seller
        - Local Hero
        - Long Tail
    """
    out = df.copy()

    if value_col not in out.columns:
        raise ValueError(f"Columna '{value_col}' no encontrada en el DataFrame.")

    values = out[value_col].fillna(0)

    q30, q60, q90 = np.quantile(values, quantiles)

    def _seller_size(v: float) -> str:
        if v >= q90:
            return "Key Account"
        elif v >= q60:
            return "Core Seller"
        elif v >= q30:
            return "Local Hero"
        else:
            return "Long Tail"

    out["seller_size"] = values.apply(_seller_size)
    return out

def add_diversification(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade la columna `clasificacion_diversificacion` al DataFrame a nivel seller,
    usando las reglas:

        if n_cat == 1 and n_items == 1      -> "Superficial"
        elif n_cat == 1 and n_items > 1    -> "Especialista"
        elif n_cat == 2 and n_items >= 2   -> "Híbrido"
        elif n_cat >= 3 or (n_cat >= 2 and n_items <= 3) -> "Disperso"
        else                               -> "Sin clasificar"
    """

    out = df.copy()
    # print(out.columns)

    # Soporte para n_categories / n_categorias
    if "n_categories" in out.columns:
        n_cat_col = "n_categories"
        # print("n_categories column found")
    elif "n_categorias" in out.columns:
        n_cat_col = "n_categorias"
        # print("n_categorias column found")
    else:
        raise ValueError("Se requiere una columna 'n_categories' o 'n_categorias'.")

    if "n_items" not in out.columns:
        raise ValueError("Se requiere la columna 'n_items'.")

    def _clasificacion_diversificacion(row: pd.Series) -> str:
        n_cat = row[n_cat_col]
        n_items = row["n_items"]

        if n_cat == 1 and n_items == 1:
            return "Superficial"
        elif n_cat == 1 and n_items > 1:
            return "Especialista"
        elif n_cat == 2 and n_items >= 2:
            return "Híbrido"
        elif n_cat >= 3 or (n_cat >= 2 and n_items <= 3):
            return "Disperso"
        else:
            return "Sin clasificar"

    out["clasificacion_diversificacion"] = out.apply(_clasificacion_diversificacion, axis=1)
    return out



def add_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade la columna `clasificacion_calidad` al DataFrame a nivel seller,
    usando las reglas:

        - premium:
            pct_new == 1 y seller_reputation_score == 5
        - confiable_gold:
            reputation in [3, 4] y pct_new >= 0.8
        - alto_riesgo:
            reputation in [0, 1, 2, 3] y pct_new <= 0.8
        - standard:
            resto de casos

    Requiere columnas:
        - pct_new
        - seller_reputation_score
    """

    out = df.copy()

    if "pct_new" not in out.columns:
        raise ValueError("Se requiere la columna 'pct_new'.")
    if "seller_reputation_score" not in out.columns:
        raise ValueError("Se requiere la columna 'seller_reputation_score'.")

    def _clasificacion_calidad(row: pd.Series) -> str:
        pct_new = row.get("pct_new", 0)
        seller_reputation_score = row.get("seller_reputation_score", 0)

        if pct_new == 1 and seller_reputation_score == 5:
            return "premium"
        elif seller_reputation_score in [3, 4] and pct_new >= 0.8:
            return "confiable_gold"
        elif seller_reputation_score in [0, 1, 2, 3] and pct_new <= 0.8:
            return "alto_riesgo"
        else:
            return "standard"

    out["clasificacion_calidad"] = out.apply(_clasificacion_calidad, axis=1)
    return out


def run_full_segmentation() -> pd.DataFrame:
    """Convenience wrapper used by scripts/notebooks."""

    df_raw = load_curated_dataset()
    df_raw = build_seller_table(df_raw)
    df_raw = add_seller_size(df_raw)
    df_raw = add_diversification(df_raw)
    df_raw = add_quality(df_raw)
    # df_raw = add_axis_scores(df_raw)
    # print(df_raw.columns)
    # print(df_raw.head())
    return df_raw

def save_segmented_dataset(
    df: pd.DataFrame, filename: str = "seller_segmentation.csv"
) -> Path:
    """Persist the seller-level dataset into ``data/processed``."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / filename
    df.to_csv(output_path, index=False)
    return output_path

