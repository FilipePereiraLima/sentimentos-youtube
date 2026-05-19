"""
preprocessor.py
---------------
Pipeline de pré-processamento de texto para os comentários coletados.

Etapas aplicadas em ordem:
    1. Limpeza bruta          : remove URLs, emojis, HTML entities, caracteres especiais
    2. Detecção de idioma     : langdetect → coluna "idioma" no DataFrame
    3. Tokenização            : quebra o texto em tokens (palavras)
    4. Remoção de stopwords   : remove palavras sem valor semântico por idioma
    5. Lematização            : reduz palavras à forma base via spaCy

Entrada : list[dict]  — saída do youtube_client.buscar_comentarios()
Saída   : pd.DataFrame com colunas originais + "idioma" + "texto_limpo"
"""

import re
import logging
import pandas as pd
import nltk
from langdetect import detect, LangDetectException
from nltk.corpus import stopwords

logger = logging.getLogger(__name__)

# ── Downloads NLTK (só na primeira execução) ──────────────────────────────────
nltk.download("stopwords", quiet=True)
nltk.download("punkt",     quiet=True)

# ── Stopwords por idioma (NLTK) ───────────────────────────────────────────────
_IDIOMAS_NLTK = {
    "pt": "portuguese",
    "en": "english",
    "es": "spanish",
    "fr": "french",
    "de": "german",
    "it": "italian",
}

def _get_stopwords(codigo_idioma: str) -> set:
    """Retorna set de stopwords para o idioma detectado. Fallback: inglês."""
    nome_nltk = _IDIOMAS_NLTK.get(codigo_idioma, "english")
    try:
        return set(stopwords.words(nome_nltk))
    except OSError:
        return set(stopwords.words("english"))


# ── spaCy: carrega modelos sob demanda ────────────────────────────────────────
_SPACY_MODELS = {
    "pt": "pt_core_news_sm",
    "en": "en_core_web_sm",
}
_spacy_cache: dict = {}

def _get_spacy(codigo_idioma: str):
    """
    Carrega e cacheia o modelo spaCy para o idioma.
    Retorna None se o modelo não estiver disponível (lematização pulada).
    """
    if codigo_idioma in _spacy_cache:
        return _spacy_cache[codigo_idioma]

    model_name = _SPACY_MODELS.get(codigo_idioma)
    if not model_name:
        _spacy_cache[codigo_idioma] = None
        return None

    try:
        import spacy
        nlp = spacy.load(model_name, disable=["parser", "ner"])
        _spacy_cache[codigo_idioma] = nlp
        return nlp
    except OSError:
        logger.warning(
            "Modelo spaCy '%s' não encontrado. "
            "Rode: python -m spacy download %s",
            model_name, model_name
        )
        _spacy_cache[codigo_idioma] = None
        return None


# ── Etapa 1: Limpeza bruta ────────────────────────────────────────────────────
def _limpar_texto(texto: str) -> str:
    """
    Remove ruído do texto mantendo o conteúdo semântico.

    Remove:
        - URLs (http/https/www)
        - Menções (@usuario)
        - Hashtags (#tag)
        - HTML entities (&amp; &lt; etc.)
        - Emojis e caracteres não-ASCII
        - Pontuação excessiva e números isolados
        - Espaços duplicados
    """
    if not isinstance(texto, str):
        return ""

    # URLs
    texto = re.sub(r"https?://\S+|www\.\S+", "", texto)

    # HTML entities
    texto = re.sub(r"&\w+;", " ", texto)

    # Menções e hashtags
    texto = re.sub(r"[@#]\w+", "", texto)

    # Emojis e caracteres não-ASCII (mantém acentuação latina)
    texto = texto.encode("ascii", "ignore").decode("ascii")

    # Pontuação (mantém espaços)
    texto = re.sub(r"[^\w\s]", " ", texto)

    # Números isolados
    texto = re.sub(r"\b\d+\b", "", texto)

    # Espaços múltiplos
    texto = re.sub(r"\s+", " ", texto).strip()

    return texto.lower()


# ── Etapa 2: Detecção de idioma ───────────────────────────────────────────────
def _detectar_idioma(texto: str) -> str:
    """
    Detecta o idioma do texto usando langdetect.
    Retorna 'und' (undetermined) se o texto for curto/ambíguo demais.
    """
    if len(texto.split()) < 3:
        return "und"
    try:
        return detect(texto)
    except LangDetectException:
        return "und"


# ── Etapa 3 + 4: Tokenização e remoção de stopwords ──────────────────────────
def _tokenizar_e_filtrar(texto: str, codigo_idioma: str) -> list[str]:
    """
    Quebra o texto em tokens e remove stopwords do idioma detectado.
    Também filtra tokens com menos de 2 caracteres.
    """
    sw = _get_stopwords(codigo_idioma)
    tokens = texto.split()
    return [t for t in tokens if t not in sw and len(t) > 1]


# ── Etapa 5: Lematização ──────────────────────────────────────────────────────
def _lematizar(tokens: list[str], codigo_idioma: str) -> str:
    """
    Reduz cada token à sua forma base (lema) via spaCy.
    Ex: "correndo" → "correr" | "melhores" → "bom"

    Se o modelo spaCy não estiver disponível, retorna os tokens sem lematização.
    """
    nlp = _get_spacy(codigo_idioma)
    if nlp is None:
        return " ".join(tokens)

    doc = nlp(" ".join(tokens))
    return " ".join([token.lemma_ for token in doc])


# ── Pipeline completo ─────────────────────────────────────────────────────────
def _processar_comentario(texto_original: str) -> tuple[str, str]:
    """
    Aplica o pipeline completo em um único comentário.

    Returns:
        (idioma, texto_limpo)
    """
    # Etapa 1: limpeza
    texto_limpo = _limpar_texto(texto_original)

    if not texto_limpo:
        return "und", ""

    # Etapa 2: idioma (detectado no texto original, antes de remover stopwords)
    idioma = _detectar_idioma(texto_limpo)

    # Etapa 3 + 4: tokenização + stopwords
    tokens = _tokenizar_e_filtrar(texto_limpo, idioma)

    if not tokens:
        return idioma, ""

    # Etapa 5: lematização
    texto_final = _lematizar(tokens, idioma)

    return idioma, texto_final


# ── Entrada pública ───────────────────────────────────────────────────────────
def preprocessar(comentarios: list[dict]) -> pd.DataFrame:
    """
    Recebe a lista de comentários do youtube_client e retorna um DataFrame
    com as colunas originais mais duas novas:
        - "idioma"      : código ISO do idioma detectado (pt, en, es, und...)
        - "texto_limpo" : texto após limpeza, tokenização, stopwords e lematização

    Comentários com texto_limpo vazio após o processamento são mantidos no
    DataFrame mas podem ser filtrados pelo app se necessário.

    Args:
        comentarios: list[dict] — saída de youtube_client.buscar_comentarios()

    Returns:
        pd.DataFrame
    """
    if not comentarios:
        logger.warning("Lista de comentários vazia. Retornando DataFrame vazio.")
        return pd.DataFrame()

    df = pd.DataFrame(comentarios)

    # Converte datas para datetime
    for col in ["publicado_em", "atualizado_em"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    logger.info("Pré-processando %d comentários...", len(df))

    resultados = df["texto"].apply(_processar_comentario)
    df["idioma"]      = resultados.apply(lambda x: x[0])
    df["texto_limpo"] = resultados.apply(lambda x: x[1])

    vazios = (df["texto_limpo"] == "").sum()
    if vazios:
        logger.info("%d comentários ficaram vazios após limpeza (spam, emojis, etc.)", vazios)

    logger.info("Pré-processamento concluído.")
    return df
