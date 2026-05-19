"""
youtube_client.py
-----------------
Responsável por toda comunicação com a YouTube Data API v3.

Funções principais:
    - extrair_video_id()   : extrai o video_id de qualquer formato de URL do YouTube
    - buscar_metadados()   : retorna título, canal, views, likes, thumbnail
    - buscar_comentarios() : coleta comentários com paginação automática (nextPageToken)

Autenticação: via API Key lida do .env (YOUTUBE_API_KEY)
Limite de cota: ~1 unidade por 100 comentários coletados
"""

import os
import re
import time
import logging
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

load_dotenv()
logger = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────────
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
API_SERVICE     = "youtube"
API_VERSION     = "v3"
MAX_RESULTS_POR_PAGINA = 100   # máximo permitido pela API por requisição
SLEEP_ENTRE_PAGINAS    = 0.5   # segundos de espera entre requisições (rate limit)


# ── Cliente ───────────────────────────────────────────────────────────────────
def _get_cliente():
    """Instancia e retorna o cliente autenticado da YouTube API."""
    if not YOUTUBE_API_KEY:
        raise ValueError(
            "YOUTUBE_API_KEY não encontrada. "
            "Verifique se o arquivo .env existe e contém a chave."
        )
    return build(API_SERVICE, API_VERSION, developerKey=YOUTUBE_API_KEY)


# ── Extração de video_id ──────────────────────────────────────────────────────
def extrair_video_id(url: str) -> str:
    """
    Extrai o video_id de diferentes formatos de URL do YouTube.

    Suporta:
        https://www.youtube.com/watch?v=VIDEO_ID
        https://youtu.be/VIDEO_ID
        https://www.youtube.com/embed/VIDEO_ID
        https://youtube.com/shorts/VIDEO_ID

    Returns:
        str: video_id (11 caracteres)

    Raises:
        ValueError: se a URL não for reconhecida como YouTube válida
    """
    padroes = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",   # watch?v= e /embed/
        r"youtu\.be\/([0-9A-Za-z_-]{11})",    # youtu.be/
        r"shorts\/([0-9A-Za-z_-]{11})",        # /shorts/
    ]
    for padrao in padroes:
        match = re.search(padrao, url)
        if match:
            return match.group(1)

    raise ValueError(f"Não foi possível extrair o video_id da URL: {url}")


# ── Metadados do vídeo ────────────────────────────────────────────────────────
def buscar_metadados(video_id: str) -> dict:
    """
    Busca informações gerais do vídeo.

    Returns:
        dict com as chaves:
            video_id     : str
            titulo       : str
            canal        : str
            publicado_em : str  (ISO 8601)
            views        : int
            likes        : int
            comentarios  : int  (total reportado pelo YouTube)
            thumbnail    : str  (URL da imagem)

    Raises:
        ValueError: se o vídeo não for encontrado ou for privado
        HttpError: para erros de quota ou autenticação
    """
    youtube = _get_cliente()

    try:
        resposta = youtube.videos().list(
            part="snippet,statistics",
            id=video_id
        ).execute()
    except HttpError as e:
        logger.error("Erro ao buscar metadados: %s", e)
        raise

    items = resposta.get("items", [])
    if not items:
        raise ValueError(
            f"Vídeo '{video_id}' não encontrado. "
            "Verifique se o link está correto e se o vídeo é público."
        )

    item       = items[0]
    snippet    = item["snippet"]
    statistics = item.get("statistics", {})

    # thumbnail: prefere maxres, depois high, depois default
    thumbnails = snippet.get("thumbnails", {})
    thumbnail  = (
        thumbnails.get("maxres", {}).get("url")
        or thumbnails.get("high", {}).get("url")
        or thumbnails.get("default", {}).get("url")
        or ""
    )

    return {
        "video_id"    : video_id,
        "titulo"      : snippet.get("title", ""),
        "canal"       : snippet.get("channelTitle", ""),
        "publicado_em": snippet.get("publishedAt", ""),
        "views"       : int(statistics.get("viewCount", 0)),
        "likes"       : int(statistics.get("likeCount", 0)),
        "comentarios" : int(statistics.get("commentCount", 0)),
        "thumbnail"   : thumbnail,
    }


# ── Comentários ───────────────────────────────────────────────────────────────
def buscar_comentarios(video_id: str, limite: int = 500) -> list[dict]:
    """
    Coleta comentários do vídeo com paginação automática via nextPageToken.

    Cada comentário retornado é um dict com:
        comentario_id : str
        texto         : str   (texto original, pode conter HTML entities)
        autor         : str
        likes         : int
        publicado_em  : str   (ISO 8601)
        atualizado_em : str   (ISO 8601)

    Args:
        video_id : str  — ID do vídeo YouTube
        limite   : int  — quantidade máxima de comentários a coletar (default 500)

    Returns:
        list[dict]: lista de comentários

    Raises:
        HttpError 403: vídeo com comentários desativados (tratado com fallback)
        HttpError 400/404: vídeo inválido ou privado
    """
    youtube    = _get_cliente()
    comentarios = []
    page_token  = None

    logger.info("Iniciando coleta de até %d comentários para %s", limite, video_id)

    while len(comentarios) < limite:
        # quantos pegar nesta página (respeita o limite total)
        max_esta_pagina = min(MAX_RESULTS_POR_PAGINA, limite - len(comentarios))

        try:
            resposta = youtube.commentThreads().list(
                part       ="snippet",
                videoId    = video_id,
                maxResults = max_esta_pagina,
                pageToken  = page_token,
                textFormat = "plainText",  # evita HTML nos textos
                order      = "relevance",  # mais relevantes primeiro
            ).execute()

        except HttpError as e:
            # Comentários desativados pelo criador do vídeo
            if e.resp.status == 403:
                logger.warning(
                    "Comentários desativados para %s. Retornando lista vazia.", video_id
                )
                return []
            logger.error("Erro ao buscar comentários: %s", e)
            raise

        for item in resposta.get("items", []):
            top_comment = item["snippet"]["topLevelComment"]["snippet"]
            comentarios.append({
                "comentario_id": item["id"],
                "texto"        : top_comment.get("textDisplay", ""),
                "autor"        : top_comment.get("authorDisplayName", ""),
                "likes"        : int(top_comment.get("likeCount", 0)),
                "publicado_em" : top_comment.get("publishedAt", ""),
                "atualizado_em": top_comment.get("updatedAt", ""),
            })

        # verifica se há próxima página
        page_token = resposta.get("nextPageToken")
        if not page_token:
            break  # acabaram as páginas disponíveis

        time.sleep(SLEEP_ENTRE_PAGINAS)  # evita rate limit

    logger.info("Coleta concluída: %d comentários obtidos.", len(comentarios))
    return comentarios[:limite]
