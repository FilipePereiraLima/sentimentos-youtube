"""
charts.py
---------
Todos os gráficos Plotly do dashboard de análise de sentimentos.

Gráficos implementados:
    1. grafico_donut()            — distribuição geral de sentimentos
    2. grafico_sentimento_tempo() — sentimento × tempo (barras empilhadas)
    3. grafico_histograma_score() — distribuição de confiança do modelo
    4. grafico_scatter_clusters() — mapa 2D de embeddings clusterizados
    5. grafico_likes_sentimento() — likes × sentimento (box plot)
    6. grafico_volume_tempo()     — volume de comentários por dia/semana
    7. tabela_interativa()        — tabela com filtros e exportação CSV
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Paleta de cores por sentimento ────────────────────────────────────────────
CORES_SENTIMENTO = {
    "Positivo": "#2ecc71",
    "Neutro"  : "#95a5a6",
    "Negativo": "#e74c3c",
}

# Paleta para clusters (até 10 clusters)
CORES_CLUSTERS = px.colors.qualitative.Set2


# ── 1. Donut de sentimentos ───────────────────────────────────────────────────
def grafico_donut(df: pd.DataFrame) -> go.Figure:
    """
    Pizza/Donut mostrando a proporção de comentários por sentimento.
    Visão geral imediata da recepção do vídeo.
    """
    contagem = df["sentimento"].value_counts().reset_index()
    contagem.columns = ["sentimento", "quantidade"]

    fig = px.pie(
        contagem,
        names  = "sentimento",
        values = "quantidade",
        hole   = 0.45,
        color  = "sentimento",
        color_discrete_map = CORES_SENTIMENTO,
        title  = "Distribuição de Sentimentos",
    )
    fig.update_traces(
        textposition = "outside",
        textinfo     = "percent+label",
        hovertemplate = "<b>%{label}</b><br>Comentários: %{value}<br>Porcentagem: %{percent}<extra></extra>",
    )
    fig.update_layout(
        showlegend   = True,
        legend_title = "Sentimento",
        margin       = dict(t=60, b=20, l=20, r=20),
    )
    return fig


# ── 2. Sentimento ao longo do tempo ──────────────────────────────────────────
def grafico_sentimento_tempo(df: pd.DataFrame, granularidade: str = "D") -> go.Figure:
    """
    Barras empilhadas mostrando evolução dos sentimentos ao longo do tempo.

    Args:
        granularidade: "D" = dia, "W" = semana, "ME" = mês
    """
    if "publicado_em" not in df.columns:
        return go.Figure().update_layout(title="Dados de data indisponíveis")

    df_temp = df.copy()
    df_temp["data"] = pd.to_datetime(df_temp["publicado_em"], utc=True, errors="coerce")
    df_temp = df_temp.dropna(subset=["data"])

    label_granularidade = {"D": "Dia", "W": "Semana", "ME": "Mês"}.get(granularidade, "Dia")
    df_temp["periodo"] = df_temp["data"].dt.to_period(granularidade).dt.to_timestamp()

    agrupado = (
        df_temp.groupby(["periodo", "sentimento"])
        .size()
        .reset_index(name="quantidade")
    )

    fig = px.bar(
        agrupado,
        x      = "periodo",
        y      = "quantidade",
        color  = "sentimento",
        color_discrete_map = CORES_SENTIMENTO,
        barmode= "stack",
        title  = f"Sentimento ao Longo do Tempo (por {label_granularidade})",
        labels = {"periodo": "Data", "quantidade": "Comentários", "sentimento": "Sentimento"},
    )
    fig.update_layout(
        xaxis_title  = "Data",
        yaxis_title  = "Número de Comentários",
        hovermode    = "x unified",
        legend_title = "Sentimento",
    )
    return fig


# ── 3. Histograma de scores de confiança ─────────────────────────────────────
def grafico_histograma_score(df: pd.DataFrame) -> go.Figure:
    """
    Histograma mostrando a distribuição dos scores de confiança do modelo.

    Score alto (0.9+) = modelo muito confiante.
    Score baixo (0.5~0.6) = comentário ambíguo ou irônico.
    """
    fig = px.histogram(
        df,
        x      = "score",
        color  = "sentimento",
        color_discrete_map = CORES_SENTIMENTO,
        nbins  = 20,
        barmode= "overlay",
        opacity= 0.75,
        title  = "Distribuição de Confiança do Modelo",
        labels = {"score": "Score de Confiança", "count": "Comentários"},
        range_x= [0.3, 1.0],
    )
    fig.update_layout(
        xaxis_title  = "Score de Confiança",
        yaxis_title  = "Número de Comentários",
        legend_title = "Sentimento",
        bargap       = 0.05,
    )
    fig.add_vline(
        x=0.5, line_dash="dash", line_color="gray",
        annotation_text="Limiar mínimo",
        annotation_position="top right",
    )
    return fig


# ── 4. Scatter 2D de embeddings clusterizados ─────────────────────────────────
def grafico_scatter_clusters(df: pd.DataFrame) -> go.Figure:
    """
    Scatter plot 2D onde cada ponto é um comentário, colorido pelo cluster K-Means.
    O hover mostra o texto original do comentário.

    Revela os grandes temas dos comentários visualmente.
    """
    if "x_umap" not in df.columns:
        return go.Figure().update_layout(title="Embeddings não disponíveis")

    df_plot = df.copy()
    df_plot["cluster_str"] = "Cluster " + df_plot["cluster"].astype(str)

    # Trunca texto para o hover (evita tooltip gigante)
    df_plot["texto_hover"] = df_plot["texto"].str[:150] + "..."

    fig = px.scatter(
        df_plot,
        x      = "x_umap",
        y      = "y_umap",
        color  = "cluster_str",
        symbol = "sentimento",
        hover_data = {
            "texto_hover" : True,
            "sentimento"  : True,
            "score"       : ":.2f",
            "x_umap"      : False,
            "y_umap"      : False,
            "cluster_str" : False,
        },
        color_discrete_sequence = CORES_CLUSTERS,
        title  = "Mapa de Embeddings — Clusters Semânticos (UMAP + K-Means)",
        labels = {"cluster_str": "Cluster", "sentimento": "Sentimento"},
    )
    fig.update_traces(marker=dict(size=6, opacity=0.7))
    fig.update_layout(
        xaxis_title  = "UMAP Dimensão 1",
        yaxis_title  = "UMAP Dimensão 2",
        legend_title = "Cluster",
        hovermode    = "closest",
    )
    return fig


# ── 5. Likes × Sentimento ─────────────────────────────────────────────────────
def grafico_likes_sentimento(df: pd.DataFrame) -> go.Figure:
    """
    Box plot mostrando a distribuição de likes por sentimento.
    Revela se comentários positivos ou negativos tendem a receber mais likes.
    """
    fig = px.box(
        df,
        x      = "sentimento",
        y      = "likes",
        color  = "sentimento",
        color_discrete_map = CORES_SENTIMENTO,
        points = "outliers",
        title  = "Distribuição de Likes por Sentimento",
        labels = {"sentimento": "Sentimento", "likes": "Número de Likes"},
    )
    fig.update_layout(
        xaxis_title  = "Sentimento",
        yaxis_title  = "Likes",
        showlegend   = False,
        yaxis_type   = "log" if df["likes"].max() > 1000 else "linear",
    )
    return fig


# ── 6. Volume de comentários ao longo do tempo ───────────────────────────────
def grafico_volume_tempo(df: pd.DataFrame, granularidade: str = "D") -> go.Figure:
    """
    Linha do tempo mostrando o volume total de comentários por período.
    Útil para ver picos de engajamento (ex: após upload, polêmicas).
    """
    if "publicado_em" not in df.columns:
        return go.Figure().update_layout(title="Dados de data indisponíveis")

    df_temp = df.copy()
    df_temp["data"] = pd.to_datetime(df_temp["publicado_em"], utc=True, errors="coerce")
    df_temp = df_temp.dropna(subset=["data"])

    label_granularidade = {"D": "Dia", "W": "Semana", "ME": "Mês"}.get(granularidade, "Dia")
    df_temp["periodo"] = df_temp["data"].dt.to_period(granularidade).dt.to_timestamp()

    volume = df_temp.groupby("periodo").size().reset_index(name="quantidade")

    fig = px.area(
        volume,
        x     = "periodo",
        y     = "quantidade",
        title = f"Volume de Comentários ao Longo do Tempo (por {label_granularidade})",
        labels= {"periodo": "Data", "quantidade": "Comentários"},
        color_discrete_sequence=["#3498db"],
    )
    fig.update_traces(
        line_color = "#2980b9",
        fillcolor  = "rgba(52, 152, 219, 0.2)",
        hovertemplate = "Data: %{x}<br>Comentários: %{y}<extra></extra>",
    )
    fig.update_layout(
        xaxis_title = "Data",
        yaxis_title = "Número de Comentários",
        hovermode   = "x unified",
    )
    return fig


# ── 7. Tabela interativa ──────────────────────────────────────────────────────
def tabela_interativa(df: pd.DataFrame) -> None:
    """
    Exibe tabela interativa com filtros por sentimento e idioma,
    e botão de exportação para CSV.

    Renderiza diretamente no Streamlit (não retorna Figure).
    """
    st.subheader("📋 Tabela de Comentários")

    # ── Filtros ──────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        sentimentos_disponiveis = ["Todos"] + sorted(df["sentimento"].unique().tolist())
        filtro_sentimento = st.selectbox("Filtrar por sentimento", sentimentos_disponiveis)

    with col2:
        if "idioma" in df.columns:
            idiomas_disponiveis = ["Todos"] + sorted(df["idioma"].unique().tolist())
            filtro_idioma = st.selectbox("Filtrar por idioma", idiomas_disponiveis)
        else:
            filtro_idioma = "Todos"

    with col3:
        score_minimo = st.slider(
            "Score mínimo de confiança",
            min_value=0.0, max_value=1.0,
            value=0.0, step=0.05,
        )

    # ── Aplicar filtros ───────────────────────────────────────────────────────
    df_filtrado = df.copy()

    if filtro_sentimento != "Todos":
        df_filtrado = df_filtrado[df_filtrado["sentimento"] == filtro_sentimento]

    if filtro_idioma != "Todos" and "idioma" in df_filtrado.columns:
        df_filtrado = df_filtrado[df_filtrado["idioma"] == filtro_idioma]

    if score_minimo > 0:
        df_filtrado = df_filtrado[df_filtrado["score"] >= score_minimo]

    # ── Colunas a exibir ──────────────────────────────────────────────────────
    colunas_exibir = ["texto", "autor", "sentimento", "score", "likes", "publicado_em"]
    if "idioma" in df_filtrado.columns:
        colunas_exibir.insert(4, "idioma")

    colunas_existentes = [c for c in colunas_exibir if c in df_filtrado.columns]
    df_exibir = df_filtrado[colunas_existentes].copy()

    # Formata data para leitura
    if "publicado_em" in df_exibir.columns:
        df_exibir["publicado_em"] = pd.to_datetime(
            df_exibir["publicado_em"], utc=True, errors="coerce"
        ).dt.strftime("%d/%m/%Y %H:%M")

    st.caption(f"Exibindo **{len(df_exibir)}** de **{len(df)}** comentários")

    st.dataframe(
        df_exibir,
        use_container_width = True,
        hide_index          = True,
        column_config       = {
            "texto"       : st.column_config.TextColumn("Comentário", width="large"),
            "autor"       : st.column_config.TextColumn("Autor", width="medium"),
            "sentimento"  : st.column_config.TextColumn("Sentimento", width="small"),
            "score"       : st.column_config.NumberColumn("Confiança", format="%.2f", width="small"),
            "likes"       : st.column_config.NumberColumn("Likes", width="small"),
            "idioma"      : st.column_config.TextColumn("Idioma", width="small"),
            "publicado_em": st.column_config.TextColumn("Data", width="medium"),
        },
    )

    # ── Exportar CSV ──────────────────────────────────────────────────────────
    csv = df_exibir.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        label     = "⬇️ Exportar CSV",
        data      = csv,
        file_name = "comentarios_analisados.csv",
        mime      = "text/csv",
    )
