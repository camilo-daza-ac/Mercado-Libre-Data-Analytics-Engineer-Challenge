from __future__ import annotations

import pandas as pd
from .playbook import PLAYBOOK


def build_prompt_for_seller(row: pd.Series) -> str:
    """
    Construye el prompt para el LLM a partir de:
      - seller_nickname
      - seller_size
      - performance_level
    y del playbook de negocio.
    """
    nickname = row["seller_nickname"]
    size = row["seller_size"]
    level = row["performance_level"]

    key = (size, level)
    base = PLAYBOOK.get(
        key,
        {
            "objetivo": "Definir una estrategia comercial básica acorde al tamaño y nivel de performance del seller.",
            "lineas": [
                "Revisar catálogo y ajustar oferta a la demanda.",
                "Optimizar precios y promociones según su contexto competitivo.",
                "Definir acciones mínimas de mejora en servicio/logística.",
            ],
        },
    )

    prompt = f"""
Eres un analista comercial senior de Mercado Libre.

Perfil del seller:
- seller_nickname: {nickname}
- seller_size: {size}
- performance_level: {level}

Playbook de referencia para este segmento:
- Objetivo sugerido: {base["objetivo"]}
- Líneas sugeridas: {", ".join(base["lineas"])}

Con esta información, genera una estrategia comercial personalizada con el siguiente formato:

1) Objetivo principal (1 párrafo).
2) 3–5 acciones concretas para el equipo comercial, separadas por viñetas.
3) 2–3 KPIs clave para evaluar el impacto de la estrategia.

La respuesta debe ser clara, accionable y escrita en un lenguaje orientado a negocio.
"""
    return prompt