# Mercado Libre – Data Analytics Engineer Challenge

Este repositorio contiene el desarrollo completo del desafío técnico para el rol de **Data Analytics Engineer**, incluyendo análisis exploratorio, clusterización de sellers, extensión con GenAI y presentación final.

---

1. **Caso base – Clusterización y segmentación de sellers**
   - EDA y limpieza del dataset original.
   - Construcción de métricas a nivel seller.
   - Segmentación de negocio (`seller_size`) y niveles de desempeño (`performance_level`).
2. **Extensión GenAI – Recomendador de estrategias comerciales (Opción B)**
   - Uso de un modelo generativo (OpenAI) para proponer estrategias comerciales personalizadas
     a partir de `seller_size` y `performance_level`.

## 📂 Estructura del Repositorio


- `data/raw/df_challenge_meli.csv`: dataset original.
- `data/processed/df_curated.csv`: datos limpios a nivel ítem.
- `data/processed/outliers_price.csv`: registro de outliers filtrados.
- `data/processed/seller_profile.csv`: perfil mínimo (size + performance).
- `data/outputs/strategies_sample.csv`: estrategias generadas por el demo GenAI.
- `notebooks/`:
  - `EDA.ipynb`: exploración, profiling y reglas de limpieza.
  - `clustering.ipynb`: features por seller, heurísticas.
  - `genai_recommender.ipynb`: prototipo de prompts/LLM.
- `src/meli_challenge/`:
  - `data_prep.py`: carga y limpieza (price p99, stock_norm, reputación).
  - `segmentation.py`: agregaciones seller, etiquetas (size, calidad, etc.).
  - `performance.py`: scoring y export.
  - `genai/`: playbook, prompts y generador.
- `scripts/run_pipeline.py`: ESTE ES EL PIPELINE DEL LA CLUESTERIZACION FINAL. orquesta limpieza+segmentación y guarda `seller_profile.csv`.
- `scripts/generate_strategies_demo.py`: ESTE ES EL DEMO DE GENERADOR DE ESTRATEGIAS. Usa `seller_profile.csv` para crear `strategies_sample.csv`.

---


## 🚀 Cómo correr todo

1. Crear entorno y dependencias
2. Colocar el CSV original en `data/raw/df_challenge_meli.csv`.
3. Ejecutar pipeline (Clusterizacion):
    PYTHONPATH=src python scripts/run_pipeline.py --data  
    Genera `df_curated.csv`, `outliers_price.csv`, `seller_profile.csv`.
4. Generar estrategias:
    PYTHONPATH=src python scripts/generate_strategies_demo.py --strategies
    Genera `strategies_sample.csv`

Requisitos:
    Python 3.9+
	Cuenta y API key de OpenAI (para la extensión GenAI).

---


