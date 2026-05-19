"""
sentiment.py
------------
Classificação de sentimento dos comentários usando HuggingFace Transformers.

Modelo: cardiffnlp/twitter-xlm-roberta-base-sentiment
    - Treinado em dados de redes sociais (estilo próximo a comentários YouTube)
    - Suporte multilíngue incluindo Português (PT-BR)
    - Classifica em: Positive / Neutral / Negative
    - Retorna score de confiança (0.0 a 1.0)

Internamente o modelo usa embeddings de 768 dimensões para classificar —
esse processo é oculto (acontece dentro do transformer).

Entrada : pd.DataFrame com coluna "texto_limpo" (saída do preprocessor)
Saída   : mesmo DataFrame + colunas "sentimento" e "score"
"""

import logging
import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────────
MODEL_NAME  = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
MAX_TOKENS  = 512       # limite do modelo; textos maiores são truncados
BATCH_SIZE  = 32        # comentários processados por vez (ajuste se der OOM)

# Mapeamento dos labels do modelo para português
LABEL_MAP = {
    "Positive": "Positivo",
    "Neutral" : "Neutro",
    "Negative": "Negativo",
}

# ── Carregamento do modelo (singleton) ───────────────────────────────────────
_classifier = None

def _get_classifier():
    """
    Carrega o modelo e tokenizer uma única vez (singleton).
    Usa GPU se disponível, senão CPU.
    """
    global _classifier
    if _classifier is not None:
        return _classifier

    device = 0 if torch.cuda.is_available() else -1
    dispositivo = "GPU" if device == 0 else "CPU"
    logger.info("Carregando modelo '%s' no %s...", MODEL_NAME, dispositivo)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    _classifier = pipeline(
        task      = "text-classification",
        model     = model,
        tokenizer = tokenizer,
        device    = device,
        # trunca automaticamente textos acima de MAX_TOKENS
        truncation      = True,
        max_length      = MAX_TOKENS,
        top_k           = 1,   # retorna só o label com maior score
    )

    logger.info("Modelo carregado com sucesso.")
    return _classifier


# ── Classificação em lote ─────────────────────────────────────────────────────
def _classificar_lote(textos: list[str]) -> list[dict]:
    """
    Classifica uma lista de textos em lote.

    Returns:
        list[dict] com chaves "sentimento" e "score" para cada texto.
        Textos vazios recebem sentimento "Neutro" e score 0.0.
    """
    classifier = _get_classifier()
    resultados = []

    for i in range(0, len(textos), BATCH_SIZE):
        lote = textos[i : i + BATCH_SIZE]

        # Separa textos vazios (não podem entrar no modelo)
        indices_validos = [j for j, t in enumerate(lote) if t and t.strip()]
        indices_vazios  = [j for j, t in enumerate(lote) if not t or not t.strip()]

        lote_resultado = [None] * len(lote)

        # Classifica textos válidos
        if indices_validos:
            textos_validos = [lote[j] for j in indices_validos]
            try:
                saidas = classifier(textos_validos)
                for idx, saida in zip(indices_validos, saidas):
                    # saida é lista com 1 elemento (top_k=1)
                    item = saida[0] if isinstance(saida, list) else saida
                    label_original = item["label"]
                    lote_resultado[idx] = {
                        "sentimento": LABEL_MAP.get(label_original, label_original),
                        "score"     : round(item["score"], 4),
                    }
            except Exception as e:
                logger.error("Erro ao classificar lote: %s", e)
                for idx in indices_validos:
                    lote_resultado[idx] = {"sentimento": "Neutro", "score": 0.0}

        # Preenche vazios com Neutro
        for idx in indices_vazios:
            lote_resultado[idx] = {"sentimento": "Neutro", "score": 0.0}

        resultados.extend(lote_resultado)

        logger.info(
            "Sentimentos: %d/%d comentários classificados...",
            min(i + BATCH_SIZE, len(textos)),
            len(textos),
        )

    return resultados


# ── Entrada pública ───────────────────────────────────────────────────────────
def classificar_sentimentos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recebe o DataFrame do preprocessor e adiciona duas colunas:
        - "sentimento" : str  — "Positivo", "Neutro" ou "Negativo"
        - "score"      : float — confiança do modelo (0.0 a 1.0)

    Usa o campo "texto_limpo" para classificar.
    Se "texto_limpo" estiver vazio, classifica o "texto" original truncado.

    Args:
        df: pd.DataFrame — saída de preprocessor.preprocessar()

    Returns:
        pd.DataFrame com as colunas "sentimento" e "score" adicionadas.
    """
    if df.empty:
        logger.warning("DataFrame vazio. Nada a classificar.")
        return df

    # Usa texto_limpo se disponível, senão texto original
    if "texto_limpo" in df.columns:
        textos = df["texto_limpo"].fillna("").tolist()
    else:
        textos = df["texto"].fillna("").tolist()

    logger.info("Classificando sentimento de %d comentários...", len(textos))

    resultados = _classificar_lote(textos)

    df = df.copy()
    df["sentimento"] = [r["sentimento"] for r in resultados]
    df["score"]      = [r["score"]      for r in resultados]

    # Log resumo
    contagem = df["sentimento"].value_counts().to_dict()
    logger.info("Classificação concluída: %s", contagem)

    return df
