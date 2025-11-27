# src/meli_challenge/genai/strategy_generator.py

from __future__ import annotations

import pandas as pd
from openai import OpenAI
from .prompt_builder import build_prompt_for_seller
from dotenv import load_dotenv
import os

load_dotenv()

# Cliente global (usa OPENAI_API_KEY del entorno)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_strategy(row: pd.Series) -> str:
    """
    Genera una estrategia comercial usando la API de OpenAI
    a partir de:
      - seller_nickname
      - seller_size
      - performance_level
    """
    prompt = build_prompt_for_seller(row)

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",  # o el modelo que estés usando en el notebook
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Eres un analista comercial senior de Mercado Libre. "
                        "Tu tarea es diseñar estrategias comerciales claras, accionables "
                        "y alineadas a objetivos de negocio."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.4,
            max_tokens=800,
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"[ERROR al llamar a la API de OpenAI]: {e}"