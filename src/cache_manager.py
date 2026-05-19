"""
cache_manager.py
----------------
Gerenciamento de cache em disco para evitar reconsultar a API do YouTube
e reprocessar os modelos NLP para o mesmo vídeo.

Estratégia de cache:
    - DataFrame completo (comentários + sentimento + embeddings) → joblib (.pkl)
    - Metadados do vídeo (título, canal, views...)               → JSON  (.json)

Estrutura de arquivos em /cache/:
    cache/
    ├── VIDEO_ID.pkl        ← DataFrame serializado com joblib
    └── VIDEO_ID_meta.json  ← Metadados do vídeo em JSON

Por que joblib e não pickle diretamente?
    - joblib é otimizado para objetos com grandes arrays numpy (como DataFrames com embeddings)
    - Compressão nativa e mais rápido que pickle para dados numéricos

Entrada/Saída: funções independentes chamadas pelo app.py
"""

import json
import logging
import os
from pathlib import Path

import joblib
import pandas as pd

logger = logging.getLogger(__name__)

# ── Configuração do diretório de cache ────────────────────────────────────────
CACHE_DIR = Path("cache")


def _garantir_cache_dir() -> None:
    """Cria o diretório /cache/ se não existir."""
    CACHE_DIR.mkdir(exist_ok=True)


def _caminho_df(video_id: str) -> Path:
    return CACHE_DIR / f"{video_id}.pkl"


def _caminho_meta(video_id: str) -> Path:
    return CACHE_DIR / f"{video_id}_meta.json"


# ── Verificação ───────────────────────────────────────────────────────────────
def cache_existe(video_id: str) -> bool:
    """
    Verifica se existe cache completo (DataFrame + metadados) para o video_id.

    Returns:
        bool: True se ambos os arquivos existirem.
    """
    existe = _caminho_df(video_id).exists() and _caminho_meta(video_id).exists()
    if existe:
        logger.info("Cache encontrado para '%s'.", video_id)
    return existe


# ── Salvamento ────────────────────────────────────────────────────────────────
def salvar_cache(video_id: str, df: pd.DataFrame, metadados: dict) -> None:
    """
    Salva o DataFrame e os metadados em disco.

    Args:
        video_id  : str          — ID do vídeo YouTube
        df        : pd.DataFrame — resultado completo do pipeline
        metadados : dict         — saída de youtube_client.buscar_metadados()
    """
    _garantir_cache_dir()

    # DataFrame → joblib (eficiente para arrays numpy grandes)
    caminho_df = _caminho_df(video_id)
    joblib.dump(df, caminho_df, compress=3)
    logger.info("DataFrame salvo em '%s'.", caminho_df)

    # Metadados → JSON (legível e leve)
    caminho_meta = _caminho_meta(video_id)
    with open(caminho_meta, "w", encoding="utf-8") as f:
        json.dump(metadados, f, ensure_ascii=False, indent=2, default=str)
    logger.info("Metadados salvos em '%s'.", caminho_meta)


# ── Carregamento ──────────────────────────────────────────────────────────────
def carregar_cache(video_id: str) -> tuple[pd.DataFrame, dict]:
    """
    Carrega DataFrame e metadados do disco.

    Returns:
        (df, metadados) — prontos para uso no dashboard.

    Raises:
        FileNotFoundError: se o cache não existir (use cache_existe() antes).
    """
    caminho_df   = _caminho_df(video_id)
    caminho_meta = _caminho_meta(video_id)

    if not caminho_df.exists() or not caminho_meta.exists():
        raise FileNotFoundError(
            f"Cache não encontrado para '{video_id}'. "
            "Processe o vídeo antes de carregar do cache."
        )

    logger.info("Carregando cache de '%s'...", video_id)

    df = joblib.load(caminho_df)

    with open(caminho_meta, "r", encoding="utf-8") as f:
        metadados = json.load(f)

    logger.info("Cache carregado: %d comentários.", len(df))
    return df, metadados


# ── Limpeza ───────────────────────────────────────────────────────────────────
def limpar_cache(video_id: str) -> None:
    """
    Remove os arquivos de cache de um vídeo específico.
    Útil para forçar reprocessamento.
    """
    for caminho in [_caminho_df(video_id), _caminho_meta(video_id)]:
        if caminho.exists():
            os.remove(caminho)
            logger.info("Cache removido: '%s'.", caminho)


def limpar_cache_completo() -> int:
    """
    Remove todos os arquivos de cache.

    Returns:
        int: quantidade de arquivos removidos.
    """
    _garantir_cache_dir()
    arquivos = list(CACHE_DIR.glob("*"))
    for arquivo in arquivos:
        os.remove(arquivo)
    logger.info("%d arquivo(s) de cache removido(s).", len(arquivos))
    return len(arquivos)


def listar_cache() -> list[dict]:
    """
    Lista todos os vídeos em cache com suas informações básicas.

    Returns:
        list[dict] com chaves: video_id, titulo, canal, tamanho_mb
    """
    _garantir_cache_dir()
    resultado = []

    for meta_file in CACHE_DIR.glob("*_meta.json"):
        video_id = meta_file.stem.replace("_meta", "")
        pkl_file = _caminho_df(video_id)

        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)

            tamanho_mb = round(pkl_file.stat().st_size / (1024 * 1024), 2) if pkl_file.exists() else 0

            resultado.append({
                "video_id"  : video_id,
                "titulo"    : meta.get("titulo", ""),
                "canal"     : meta.get("canal", ""),
                "tamanho_mb": tamanho_mb,
            })
        except Exception as e:
            logger.warning("Erro ao ler cache de '%s': %s", video_id, e)

    return resultado
