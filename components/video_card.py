"""
video_card.py
-------------
Componente Streamlit que exibe o card de metadados do vídeo no topo do dashboard.

Exibe:
    - Thumbnail do vídeo
    - Título, canal e data de publicação
    - Métricas: views, likes, total de comentários coletados
"""

import streamlit as st
from datetime import datetime, timezone


def _formatar_numero(n: int) -> str:
    """Formata números grandes com separador: 1234567 → '1.234.567'"""
    return f"{n:,}".replace(",", ".")


def _formatar_data(iso_str: str) -> str:
    """Converte string ISO 8601 para formato legível: '2024-03-15'"""
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.strftime("%d/%m/%Y")
    except Exception:
        return iso_str


def exibir_video_card(metadados: dict, total_coletados: int) -> None:
    """
    Renderiza o card de metadados do vídeo.

    Args:
        metadados       : dict — saída de youtube_client.buscar_metadados()
        total_coletados : int  — quantidade de comentários coletados (pode ser
                                 menor que metadados["comentarios"] por causa do limite)
    """
    st.divider()

    col_thumb, col_info = st.columns([1, 2.5], gap="large")

    # ── Thumbnail ─────────────────────────────────────────────────────────────
    with col_thumb:
        if metadados.get("thumbnail"):
            st.image(metadados["thumbnail"], use_container_width=True)
        else:
            st.markdown("🎬 *Thumbnail indisponível*")

    # ── Informações textuais ──────────────────────────────────────────────────
    with col_info:
        st.markdown(f"### {metadados.get('titulo', 'Título indisponível')}")
        st.markdown(
            f"📺 **{metadados.get('canal', '')}** &nbsp;·&nbsp; "
            f"🗓️ {_formatar_data(metadados.get('publicado_em', ''))}"
        )

        st.markdown("")  # espaçamento

        # Métricas em linha
        m1, m2, m3 = st.columns(3)
        m1.metric("👁️ Views",     _formatar_numero(metadados.get("views", 0)))
        m2.metric("👍 Likes",     _formatar_numero(metadados.get("likes", 0)))
        m3.metric("💬 Comentários coletados", _formatar_numero(total_coletados))

        # Aviso se coletou menos que o total real
        total_real = metadados.get("comentarios", 0)
        if total_real > total_coletados:
            st.caption(
                f"ℹ️ O vídeo tem **{_formatar_numero(total_real)}** comentários no total. "
                f"Foram coletados **{_formatar_numero(total_coletados)}** conforme o limite configurado."
            )

    st.divider()
