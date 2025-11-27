from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# src/meli_challenge/performance.p

# Mapas de score por eje
DIV_SCORE_MAP = {
    "Especialista": 2,
    "Híbrido": 2,
    "Superficial": 1,
    "Disperso": 0,
}

QUAL_SCORE_MAP = {
    "premium": 2,
    "confiable_gold": 2,
    "standard": 1,
    "alto_riesgo": 0,
}

LOG_SCORE_MAP = {
    "XD": 2,      # full / integrada
    "DS": 2,
    "FLEX": 2,    # express / rápida
    "Otro": 1,    # intermedio
    "FBM": 0,     # autogestionada por el seller
}


def _classify_performance(row: pd.Series) -> str:
    """
    Lógica de negocio para asignar performance_level por seller,
    usando seller_size + scores de diversificación, calidad y logística.

    Niveles:
        - Diamante
        - Top performance
        - Expected performance
        - Low performance
    """
    size = row["seller_size"]
    div_score = row["div_score"]
    qual_score = row["qual_score"]
    log_score = row["log_score"]
    total = row["total_score"]

    has_low_quality = (row["clasificacion_calidad"] == "alto_riesgo")
    is_disperso = (row["clasificacion_diversificacion"] == "Disperso")
    is_fbm = (row["logistic_type"] == "FBM")

    # 0) Diamante – nivel global, por encima de Top
    if (
        total == 6
        and not has_low_quality
        and not is_disperso
        and not is_fbm
    ):
        return "Diamante"

    # 1) Reglas por seller_size

    # --- Key Account (muy exigente) ---
    if size == "Key Account":
        # TOP: perfil muy alto (≥5) sin riesgos fuertes
        if (total >= 5) and not has_low_quality and not is_disperso and not is_fbm:
            return "Top performance"
        # LOW: mala calidad, disperso o score muy bajo
        if has_low_quality or is_disperso or total <= 2:
            return "Low performance"
        # resto
        return "Expected performance"

    # --- Core Seller ---
    if size == "Core Seller":
        # TOP: score alto (≥5) y sin riesgo fuerte
        if (total >= 5) and not has_low_quality and not is_disperso:
            return "Top performance"
        if has_low_quality or total <= 1:
            return "Low performance"
        return "Expected performance"

    # --- Local Hero ---
    if size == "Local Hero":
        # TOP: buena calidad + score alto + no disperso
        if (qual_score == 2) and (total >= 5) and not is_disperso:
            return "Top performance"
        if has_low_quality or (total <= 2 and is_disperso):
            return "Low performance"
        return "Expected performance"

    # --- Long Tail ---
    if size == "Long Tail":
        # TOP: pequeño pero sólido en todo (ningún eje en 0) y total razonable
        if (
            total >= 4
            and div_score >= 1
            and qual_score >= 1
            and log_score >= 1
        ):
            return "Top performance"
        if has_low_quality or total <= 1:
            return "Low performance"
        return "Expected performance"

    # fallback
    return "Expected performance"


def add_performance_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade al DataFrame a nivel seller:

        - div_score
        - qual_score
        - log_score
        - total_score
        - performance_level
        - performance_segment

    Requiere columnas previas:
        - seller_size
        - clasificacion_diversificacion
        - clasificacion_calidad
        - logistic_type
    """
    out = df.copy()

    # Validaciones mínimas
    required_cols = [
        "seller_size",
        "clasificacion_diversificacion",
        "clasificacion_calidad",
        "logistic_type",
    ]
    missing = [c for c in required_cols if c not in out.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas para performance: {missing}")

    # Scores por eje
    out["div_score"] = out["clasificacion_diversificacion"].map(DIV_SCORE_MAP)
    out["qual_score"] = out["clasificacion_calidad"].map(QUAL_SCORE_MAP)
    out["log_score"] = out["logistic_type"].map(LOG_SCORE_MAP)

    out["total_score"] = out["div_score"] + out["qual_score"] + out["log_score"]

    # Clasificación final
    out["performance_level"] = out.apply(_classify_performance, axis=1)
    out["performance_segment"] = out["seller_size"] + " - " + out["performance_level"]

    return out

def export_full_dataset(seller_table: pd.DataFrame, filename: str = "seller_performance.csv") -> Path:
    """Save the full seller performance dataset to the processed folder."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / filename
    seller_table.to_csv(output_path, index=False)
    return output_path

