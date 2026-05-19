"""
app.py
------
Entry point do sistema de análise de sentimentos de comentários YouTube.

Fluxo completo:
    1. Usuário cola o link do YouTube + configura limite de comentários
    2. app.py verifica cache (joblib) — se existir, carrega direto
    3. Se não existir: youtube_client → preprocessor → sentiment → embeddings
    4. Salva resultado em cache
    5. Renderiza o dashboard com os 7 gráficos + tabela

Para rodar:
    streamlit run app.py
"""

import logging
import streamlit as st

from src.youtube_client  import extrair_video_id, buscar_metadados, buscar_comentarios
from src.preprocessor    import preprocessar
from src.sentiment       import classificar_sentimentos
from src.embeddings      import gerar_embeddings_e_clusters
from src.cache_manager   import cache_existe, salvar_cache, carregar_cache, limpar_cache

from components.video_card import exibir_video_card
from components.charts     import (
    grafico_donut,
    grafico_sentimento_tempo,
    grafico_histograma_score,
    grafico_scatter_clusters,
    grafico_likes_sentimento,
    grafico_volume_tempo,
    tabela_interativa,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Configuração da página ────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Análise de Sentimentos — YouTube",
    page_icon  = "🎯",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)


# ── Pipeline de processamento ─────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _rodar_pipeline(video_id: str, limite: int, k_clusters: int):
    """
    Executa o pipeline completo e salva em cache.
    O decorator st.cache_data evita reprocessamento durante a mesma sessão.

    Returns:
        (df, metadados)
    """
    # Etapa 1: metadados
    with st.spinner("📡 Buscando informações do vídeo..."):
        metadados = buscar_metadados(video_id)

    # Etapa 2: comentários
    with st.spinner(f"💬 Coletando até {limite} comentários..."):
        comentarios = buscar_comentarios(video_id, limite=limite)

    if not comentarios:
        st.warning(
            "⚠️ Este vídeo não possui comentários disponíveis "
            "(comentários desativados ou vídeo sem comentários)."
        )
        st.stop()

    # Etapa 3: pré-processamento NLP
    with st.spinner("🔤 Aplicando pipeline NLP (limpeza, tokenização, lematização)..."):
        df = preprocessar(comentarios)

    # Etapa 4: classificação de sentimentos
    with st.spinner("🤖 Classificando sentimentos com transformer..."):
        df = classificar_sentimentos(df)

    # Etapa 5: embeddings + UMAP + K-Means
    with st.spinner("📐 Gerando embeddings e clusterizando comentários..."):
        df = gerar_embeddings_e_clusters(df, k=k_clusters)

    # Etapa 6: cache
    salvar_cache(video_id, df, metadados)

    return df, metadados


# ── Sidebar ───────────────────────────────────────────────────────────────────
def _render_sidebar() -> tuple[str, int, int, str, bool]:
    """
    Renderiza a sidebar de configuração e retorna os parâmetros escolhidos.

    Returns:
        (url, limite, k_clusters, granularidade, forcar_reprocessar)
    """
    st.sidebar.title("⚙️ Configurações")
    st.sidebar.markdown("---")

    url = st.sidebar.text_input(
        "🔗 Link do vídeo YouTube",
        placeholder="https://www.youtube.com/watch?v=...",
    )

    limite = st.sidebar.select_slider(
        "💬 Limite de comentários",
        options=[100, 500, 1000],
        value=500,
    )

    k_clusters = st.sidebar.slider(
        "🔵 Número de clusters (K-Means)",
        min_value=2, max_value=10, value=5,
        help="Quantos grupos temáticos o algoritmo deve encontrar nos comentários.",
    )

    granularidade = st.sidebar.radio(
        "📅 Granularidade temporal",
        options=["D", "W", "ME"],
        format_func=lambda x: {"D": "Dia", "W": "Semana", "ME": "Mês"}[x],
        horizontal=True,
    )

    st.sidebar.markdown("---")
    forcar_reprocessar = st.sidebar.checkbox(
        "🔄 Forçar reprocessamento",
        help="Ignora o cache e reanalisa o vídeo do zero.",
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "📚 Projeto acadêmico — Introdução à IA\n\n"
        "**Stack:** Streamlit · HuggingFace · sentence-transformers · "
        "UMAP · K-Means · Plotly"
    )

    return url, limite, k_clusters, granularidade, forcar_reprocessar


# ── Dashboard ─────────────────────────────────────────────────────────────────
def _render_dashboard(df, metadados, granularidade: str) -> None:
    """Renderiza todos os componentes do dashboard."""

    # Card de metadados
    exibir_video_card(metadados, total_coletados=len(df))

    # ── Linha 1: Donut + Histograma ───────────────────────────────────────────
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.plotly_chart(grafico_donut(df), use_container_width=True)
    with col2:
        st.plotly_chart(grafico_histograma_score(df), use_container_width=True)

    # ── Linha 2: Sentimento × Tempo ───────────────────────────────────────────
    st.plotly_chart(
        grafico_sentimento_tempo(df, granularidade),
        use_container_width=True,
    )

    # ── Linha 3: Volume × Tempo ───────────────────────────────────────────────
    st.plotly_chart(
        grafico_volume_tempo(df, granularidade),
        use_container_width=True,
    )

    # ── Linha 4: Scatter Clusters + Box Likes ─────────────────────────────────
    col3, col4 = st.columns([1.6, 1], gap="large")
    with col3:
        st.plotly_chart(grafico_scatter_clusters(df), use_container_width=True)
    with col4:
        st.plotly_chart(grafico_likes_sentimento(df), use_container_width=True)

    # ── Linha 5: Tabela interativa ────────────────────────────────────────────
    st.markdown("---")
    tabela_interativa(df)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Header
    st.title("🎯 Análise de Sentimentos — Comentários YouTube")
    st.markdown(
        "Cole o link de um vídeo, configure o limite de comentários e clique em **Analisar**. "
        "O sistema aplica NLP completo e exibe os resultados em um dashboard interativo."
    )

    # Sidebar
    url, limite, k_clusters, granularidade, forcar_reprocessar = _render_sidebar()

    # Botão de análise
    analisar = st.button("🚀 Analisar", type="primary", use_container_width=True)

    # Recupera da sessão se já foi processado (evita reset ao usar filtros)
    if not analisar and "df" in st.session_state:
        _render_dashboard(
            st.session_state["df"],
            st.session_state["metadados"],
            granularidade,
        )
        return

    if not analisar:
        # Tela de boas-vindas
        st.info(
            "👈 Cole o link de um vídeo na sidebar e clique em **Analisar** para começar."
        )
        _exibir_sobre()
        return

    # Validação da URL
    if not url or not url.strip():
        st.error("❌ Por favor, insira o link de um vídeo YouTube.")
        return

    try:
        video_id = extrair_video_id(url.strip())
    except ValueError as e:
        st.error(f"❌ URL inválida: {e}")
        return

    # Limpa cache se solicitado
    if forcar_reprocessar:
        limpar_cache(video_id)
        # Limpa cache do st.cache_data também
        _rodar_pipeline.clear()

    # ── Carrega do cache ou processa ──────────────────────────────────────────
    if cache_existe(video_id) and not forcar_reprocessar:
        st.success("⚡ Resultado carregado do cache! (processamento instantâneo)")
        df, metadados = carregar_cache(video_id)
    else:
        st.info("🔄 Processando vídeo pela primeira vez (pode levar alguns minutos)...")
        try:
            df, metadados = _rodar_pipeline(video_id, limite, k_clusters)
            st.success("✅ Análise concluída!")
        except ValueError as e:
            st.error(f"❌ Erro ao buscar vídeo: {e}")
            return
        except Exception as e:
            st.error(f"❌ Erro inesperado: {e}")
            logger.exception("Erro no pipeline:")
            return

    # Salva na sessão para os filtros não resetarem
    st.session_state["df"]        = df
    st.session_state["metadados"] = metadados

    # ── Renderiza dashboard ───────────────────────────────────────────────────
    _render_dashboard(df, metadados, granularidade)


# ── Tela de boas-vindas ───────────────────────────────────────────────────────
def _exibir_sobre():
    """Exibe informações sobre o projeto na tela inicial."""
    st.markdown("---")
    st.subheader("ℹ️ Como funciona")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 📥 Coleta")
        st.markdown(
            "Os comentários são coletados via **YouTube Data API v3**. "
            "O sistema suporta paginação automática para coletar até 1.000 comentários por vídeo."
        )

    with col2:
        st.markdown("#### 🧠 Análise NLP")
        st.markdown(
            "Cada comentário passa por **limpeza**, **tokenização**, **remoção de stopwords** "
            "e **lematização**. A classificação usa o modelo transformer "
            "`cardiffnlp/twitter-xlm-roberta-base-sentiment`, multilíngue e treinado "
            "em dados de redes sociais."
        )

    with col3:
        st.markdown("#### 📊 Visualização")
        st.markdown(
            "Os embeddings semânticos (384D) são gerados com **sentence-transformers**, "
            "comprimidos para 2D com **UMAP** e agrupados com **K-Means**. "
            "Os resultados são exibidos em 7 gráficos interativos com **Plotly**."
        )

    st.markdown("---")
    st.subheader("📐 Pipeline técnico")
    st.code(
        """
YouTube URL
    └── youtube_client.py   → coleta metadados + comentários (YouTube API v3)
        └── preprocessor.py → limpeza + tokenização + stopwords + lematização
            └── sentiment.py → classificação transformer (Positivo / Neutro / Negativo)
                └── embeddings.py → sentence-transformers (384D) → UMAP (2D) → K-Means
                    └── cache_manager.py → salva resultado em disco (joblib)
                        └── dashboard → 7 gráficos Plotly + tabela interativa
        """,
        language="",
    )


if __name__ == "__main__":
    main()
