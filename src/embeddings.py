"""
embeddings.py
-------------
Geração de embeddings semânticos e clusterização dos comentários.

Pipeline:
    1. sentence-transformers  → vetor de 384 dimensões por comentário
    2. UMAP                   → comprime 384D → 2D (x_umap, y_umap)
    3. K-Means (scikit-learn) → agrupa os pontos 2D em K clusters

Por que este uso de embedding é diferente do sentiment.py?
    - Em sentiment.py: o transformer usa embeddings de 768D internamente
      para classificar — você não vê nem manipula esses vetores.
    - Aqui: os embeddings de 384D são gerados explicitamente e usados
      para calcular similaridade semântica entre comentários, alimentar
      o UMAP e colorir os clusters no gráfico.

Modelo: paraphrase-multilingual-MiniLM-L12-v2
    - Leve (~120MB), rápido, multilíngue (PT, EN, ES e outros)
    - Gera vetores de 384 dimensões que capturam significado semântico
    - Comentários com sentido parecido ficam matematicamente próximos

Entrada : pd.DataFrame com coluna "texto_limpo" (saída do sentiment)
Saída   : mesmo DataFrame + colunas "x_umap", "y_umap", "cluster"
"""

import logging
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import umap

logger = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL  = "paraphrase-multilingual-MiniLM-L12-v2"
UMAP_N_NEIGHBORS = 15       # vizinhos considerados pelo UMAP (estrutura local)
UMAP_MIN_DIST    = 0.1      # distância mínima entre pontos no espaço 2D
UMAP_METRIC      = "cosine" # distância semântica entre vetores
UMAP_RANDOM_STATE = 42
KMEANS_K_DEFAULT  = 5       # número de clusters padrão
KMEANS_RANDOM_STATE = 42
BATCH_SIZE        = 64      # comentários por lote no encode

# ── Carregamento do modelo (singleton) ───────────────────────────────────────
_embedding_model = None

def _get_embedding_model() -> SentenceTransformer:
    """Carrega o modelo sentence-transformers uma única vez."""
    global _embedding_model
    if _embedding_model is None:
        logger.info("Carregando modelo de embeddings '%s'...", EMBEDDING_MODEL)
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Modelo de embeddings carregado.")
    return _embedding_model


# ── Etapa 1: Geração de embeddings ───────────────────────────────────────────
def _gerar_embeddings(textos: list[str]) -> np.ndarray:
    """
    Converte cada texto em um vetor de 384 dimensões.

    Textos vazios recebem vetor zerado (não entram no modelo).

    Returns:
        np.ndarray de shape (n_comentarios, 384)
    """
    model = _get_embedding_model()

    # Substitui vazios por placeholder (será zerado depois)
    placeholder = "comentário vazio"
    indices_vazios = [i for i, t in enumerate(textos) if not t or not t.strip()]
    textos_proc = [t if t and t.strip() else placeholder for t in textos]

    logger.info("Gerando embeddings para %d comentários...", len(textos_proc))

    embeddings = model.encode(
        textos_proc,
        batch_size       = BATCH_SIZE,
        show_progress_bar= False,
        normalize_embeddings = True,  # normaliza para cosine similarity
    )

    # Zera vetores de comentários originalmente vazios
    for idx in indices_vazios:
        embeddings[idx] = np.zeros(embeddings.shape[1])

    logger.info("Embeddings gerados: shape %s", embeddings.shape)
    return embeddings


# ── Etapa 2: Redução dimensional com UMAP ────────────────────────────────────
def _reduzir_umap(embeddings: np.ndarray) -> np.ndarray:
    """
    Comprime os vetores de 384D para 2D usando UMAP.

    Por que UMAP em vez de PCA?
        - PCA é linear: preserva variância global mas perde estrutura local
        - UMAP é não-linear: preserva relações de vizinhança entre pontos
        - Para NLP, comentários similares ficam visualmente agrupados com UMAP

    Returns:
        np.ndarray de shape (n_comentarios, 2) com coordenadas (x, y)
    """
    n = len(embeddings)

    # UMAP precisa de pelo menos n_neighbors + 1 amostras
    n_neighbors = min(UMAP_N_NEIGHBORS, n - 1) if n > 1 else 1

    logger.info("Reduzindo %dD → 2D com UMAP...", embeddings.shape[1])

    reducer = umap.UMAP(
        n_components = 2,
        n_neighbors  = n_neighbors,
        min_dist     = UMAP_MIN_DIST,
        metric       = UMAP_METRIC,
        random_state = UMAP_RANDOM_STATE,
    )

    # Normaliza antes do UMAP para melhor separação
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    coords_2d = reducer.fit_transform(embeddings_scaled)
    logger.info("UMAP concluído: shape %s", coords_2d.shape)
    return coords_2d


# ── Etapa 3: Clusterização com K-Means ───────────────────────────────────────
def _clusterizar_kmeans(coords_2d: np.ndarray, k: int) -> np.ndarray:
    """
    Agrupa os pontos 2D em K clusters usando K-Means.

    O algoritmo:
        1. Inicializa K centroides aleatoriamente (k-means++)
        2. Atribui cada ponto ao centroide mais próximo
        3. Recalcula os centroides como média do cluster
        4. Repete até convergir

    Returns:
        np.ndarray de shape (n_comentarios,) com o label do cluster (0 a K-1)
    """
    n = len(coords_2d)
    k_efetivo = min(k, n)  # não pode ter mais clusters do que pontos

    if k_efetivo != k:
        logger.warning(
            "K ajustado de %d para %d (menos comentários que clusters).", k, k_efetivo
        )

    logger.info("Clusterizando com K-Means (K=%d)...", k_efetivo)

    kmeans = KMeans(
        n_clusters   = k_efetivo,
        init         = "k-means++",   # inicialização inteligente (mais estável)
        n_init       = 10,
        random_state = KMEANS_RANDOM_STATE,
    )
    labels = kmeans.fit_predict(coords_2d)
    logger.info("K-Means concluído.")
    return labels


# ── Entrada pública ───────────────────────────────────────────────────────────
def gerar_embeddings_e_clusters(
    df: pd.DataFrame,
    k: int = KMEANS_K_DEFAULT,
) -> pd.DataFrame:
    """
    Executa o pipeline completo: embeddings → UMAP → K-Means.

    Adiciona ao DataFrame as colunas:
        - "x_umap"  : float — coordenada X no espaço 2D
        - "y_umap"  : float — coordenada Y no espaço 2D
        - "cluster" : int   — label do cluster (0 a K-1)

    Args:
        df : pd.DataFrame — saída de sentiment.classificar_sentimentos()
        k  : int          — número de clusters desejados (default 5)

    Returns:
        pd.DataFrame com as três colunas novas adicionadas.
    """
    if df.empty:
        logger.warning("DataFrame vazio. Nada a processar.")
        return df

    # Usa texto_limpo se disponível, senão texto original
    coluna_texto = "texto_limpo" if "texto_limpo" in df.columns else "texto"
    textos = df[coluna_texto].fillna("").tolist()

    # Etapa 1: embeddings
    embeddings = _gerar_embeddings(textos)

    # Etapa 2: UMAP
    coords_2d = _reduzir_umap(embeddings)

    # Etapa 3: K-Means
    clusters = _clusterizar_kmeans(coords_2d, k)

    df = df.copy()
    df["x_umap"]  = coords_2d[:, 0]
    df["y_umap"]  = coords_2d[:, 1]
    df["cluster"] = clusters

    logger.info(
        "Pipeline de embeddings concluído. Colunas adicionadas: x_umap, y_umap, cluster."
    )
    return df
