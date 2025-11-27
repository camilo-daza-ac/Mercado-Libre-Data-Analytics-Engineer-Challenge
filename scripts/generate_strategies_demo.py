# scripts/generate_strategies_demo.py

import sys
from pathlib import Path
import argparse
import logging
import pandas as pd
from typing import List, Optional


# Añadimos src/ al path para poder importar el paquete
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from meli_challenge.genai import generate_strategy  

PROFILE_PATH = ROOT / "data" / "processed" / "seller_profile.csv"
OUT_PATH = ROOT / "data" / "outputs" / "strategies_sample.csv"


def run_strategy_generation() -> None:
    df = pd.read_csv(PROFILE_PATH)
    cols = ["seller_nickname", "seller_size", "performance_level"]
    df = df[cols].copy()

    ejemplos = (
        df.query("performance_level in ['Diamante', 'Top performance', 'Low performance']")
          .groupby(['seller_size', 'performance_level'])
          .head(1)
          .reset_index(drop=True)
    )


    rows = []
    for _, row in ejemplos.iterrows():
        strategy_text = generate_strategy(row)
        rows.append(
            {
                "seller_nickname": row["seller_nickname"],
                "seller_size": row["seller_size"],
                "performance_level": row["performance_level"],
                "strategy": strategy_text,
            }
        )

    out_df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False)

    print(f"[OK] Estrategias de ejemplo guardadas en: {OUT_PATH}")

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run Mercado Libre strategy generation")
    parser.add_argument(
        "--strategies",
        action="store_true",
        help="Run only the data preparation stage",
    )
    args = parser.parse_args(argv)

    if not args.strategies:
        parser.error("For now you must pass --strategies to run the strategy generation.")

    run_strategy_generation()

if __name__ == "__main__":
    main()
    