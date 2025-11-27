"""Entry-point to execute the full data pipeline from the CLI.

For now this script only runs the data preparation stage. As the solution
evolves we can plug in the segmentation and clustering modules from
``src/meli_challenge``.
"""

from __future__ import annotations

# scripts/run_pipeline.py
import sys
from pathlib import Path

# Añadir src/ al sys.path para poder importar meli_challenge
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import argparse
import logging
from pathlib import Path

from meli_challenge import data_prep
from meli_challenge import segmentation
from meli_challenge import performance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def run_data_stage() -> None:
    """Execute the data preparation stage and report basic stats."""

    logging.info("Starting data preparation stage…")
    df_clean = data_prep.run_full_preparation()
    # logging.info("Finished! Curated dataset shape: %s", df_clean.shape)
    df_segmented = segmentation.run_full_segmentation()
    # logging.info("Finished! Segmented dataset shape: %s", df_segmented.shape)
    df_segmented = performance.add_performance_level(df_segmented)
    logging.info("Finished! Performance dataset shape: %s", df_segmented.shape)

    # performance_level_counts = df_segmented["performance_level"].value_counts().reset_index()
    # performance_level_counts.columns = ["performance_level", "count"]
    # performance_level_counts["percentage"] = (performance_level_counts["count"] / performance_level_counts["count"].sum()) * 100
    # print("Distribución por performance_level:")
    # print(performance_level_counts)

    data_prep.save_segmented_dataset(df_segmented, filename="seller_profile.csv") 
    logging.info("Segmented dataset saved in %s", data_prep.PROCESSED_DIR / "seller_profile.csv")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run Mercado Libre pipeline")
    parser.add_argument(
        "--data",
        action="store_true",
        help="Run only the data preparation stage",
    )
    args = parser.parse_args(argv)

    if not args.data:
        parser.error("For now you must pass --data to run the pipeline.")

    run_data_stage()


if __name__ == "__main__":
    main()

