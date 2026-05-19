# 🎯 Análise de Sentimentos — Comentários YouTube

Sistema web que coleta comentários de qualquer vídeo do YouTube e aplica um pipeline completo de **Processamento de Linguagem Natural (NLP)** para classificar o sentimento de cada comentário (Positivo, Neutro ou Negativo), exibindo os resultados em um dashboard interativo com 7 gráficos.

> Projeto acadêmico desenvolvido para a disciplina de **Introdução à Inteligência Artificial**.

---

## 📸 Funcionalidades

- Cole qualquer link do YouTube e receba uma análise completa dos comentários
- Classificação automática de sentimentos com modelo transformer multilíngue (suporte a PT-BR)
- Dashboard interativo com 7 gráficos Plotly (donut, barras, histograma, scatter, box plot, área e tabela)
- Mapa visual de clusters semânticos via UMAP + K-Means
- Cache inteligente por vídeo — evita reprocessamento desnecessário
- Exportação dos resultados em CSV

---

## 🧠 Conceitos de IA Demonstrados

| Conceito | Como aparece no projeto |
|---|---|
| **Compreensão de linguagem** | Transformer lê contexto, ironia e gírias — não apenas palavras-chave |
| **NLP clássico** | Pipeline: tokenização → stopwords → lematização (spaCy + NLTK) |
| **Embeddings (uso interno)** | Transformer usa vetores de 768D internamente para classificar |
| **Embeddings (uso explícito)** | sentence-transformers gera vetores de 384D para o gráfico de clusters |
| **Redução dimensional** | UMAP comprime 384D → 2D preservando similaridade semântica |
| **Clusterização** | K-Means agrupa comentários em N grupos temáticos |
| **Classificação de texto** | Saída do transformer: label (Pos/Neu/Neg) + score de confiança |
| **Tabulação** | DataFrame pandas com filtros, exportação CSV e visualização interativa |

---

## 🗂️ Estrutura do Projeto

```
sentimentos_youtube/
│
├── app.py                    # Entry point — Streamlit UI + orquestração do pipeline
├── .env                      # API Key do YouTube (NÃO versionar!)
├── .env.example              # Modelo do .env
├── .gitignore
├── requirements.txt
├── README.md
│
├── src/
│   ├── youtube_client.py     # Coleta metadados e comentários via YouTube Data API v3
│   ├── preprocessor.py       # Limpeza, tokenização, stopwords, lematização
│   ├── sentiment.py          # Classificação de sentimento com HuggingFace transformer
│   ├── embeddings.py         # sentence-transformers + UMAP + K-Means
│   └── cache_manager.py      # Cache em disco com joblib
│
├── components/
│   ├── video_card.py         # Card de metadados do vídeo (Streamlit)
│   └── charts.py             # 7 gráficos Plotly do dashboard
│
└── cache/                    # Gerado automaticamente — resultados por video_id
```

---

## ⚙️ Stack Tecnológica

| Camada | Tecnologia | Por quê? |
|---|---|---|
| Frontend/UI | Streamlit | Rápido de montar, nativo para dashboards Python |
| Linguagem | Python 3.11 | Ecossistema NLP mais maduro disponível |
| Fonte de dados | YouTube Data API v3 | API oficial do Google, estável e bem documentada |
| Gráficos | Plotly | Gráficos interativos com zoom, hover e filtros |
| Manipulação | Pandas | Tabulação, filtragem e exportação dos dados |
| NLP (classificação) | HuggingFace Transformers | Modelo pré-treinado multilíngue de alta acurácia |
| NLP (embeddings) | sentence-transformers | Embeddings semânticos explícitos para clusterização |
| NLP (clássico) | spaCy + NLTK | Tokenização, stopwords, lematização |
| Redução dimensional | UMAP | Preserva relações locais melhor que PCA para NLP |
| Clusterização | scikit-learn (K-Means) | Agrupamento não supervisionado dos comentários |
| Cache | joblib | Serialização eficiente de DataFrames com arrays numpy |

---

## 🔄 Pipeline Técnico

```
URL do YouTube
    │
    ▼
youtube_client.py
    ├── extrair_video_id()      → suporta watch, youtu.be, shorts, embed
    ├── buscar_metadados()      → título, canal, views, likes, thumbnail
    └── buscar_comentarios()    → paginação automática via nextPageToken
    │
    ▼
preprocessor.py
    ├── limpeza bruta           → remove URLs, emojis, HTML entities, menções
    ├── detecção de idioma      → langdetect (pt, en, es, fr, de, it...)
    ├── tokenização             → quebra o texto em tokens individuais
    ├── remoção de stopwords    → por idioma via NLTK
    └── lematização             → reduz à forma base via spaCy
    │
    ▼
sentiment.py
    └── transformer             → cardiffnlp/twitter-xlm-roberta-base-sentiment
                                   Positivo / Neutro / Negativo + score (0.0–1.0)
                                   (usa embeddings de 768D internamente)
    │
    ▼
embeddings.py
    ├── sentence-transformers   → vetor de 384D por comentário (explícito)
    ├── UMAP                    → 384D → 2D (x_umap, y_umap)
    └── K-Means                 → agrupa pontos 2D em K clusters
    │
    ▼
cache_manager.py
    ├── DataFrame → joblib (.pkl)
    └── Metadados → JSON
    │
    ▼
Dashboard Streamlit
    ├── Gráfico 1: Donut — distribuição de sentimentos
    ├── Gráfico 2: Barras empilhadas — sentimento × tempo
    ├── Gráfico 3: Histograma — scores de confiança do modelo
    ├── Gráfico 4: Scatter 2D — clusters semânticos (UMAP + K-Means)
    ├── Gráfico 5: Box plot — likes × sentimento
    ├── Gráfico 6: Área — volume de comentários ao longo do tempo
    └── Gráfico 7: Tabela interativa — filtros + exportação CSV
```

---

## 🚀 Como Rodar

### Pré-requisitos

- Python 3.11+
- Conta Google com acesso ao [Google Cloud Console](https://console.cloud.google.com)

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/sentimentos-youtube.git
cd sentimentos-youtube
```

### 2. Crie e ative o ambiente virtual

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / Mac
source .venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Baixe os modelos do spaCy

```bash
python -m spacy download pt_core_news_sm
python -m spacy download en_core_web_sm
```

### 5. Configure a API Key do YouTube

```bash
# Copie o arquivo de exemplo
cp .env.example .env
```

Edite o `.env` e cole sua chave:

```env
YOUTUBE_API_KEY=sua_chave_aqui
```

**Como obter a API Key:**
1. Acesse [console.cloud.google.com](https://console.cloud.google.com)
2. Crie um projeto
3. Ative a **YouTube Data API v3**
4. Vá em **Credenciais → Criar credencial → Chave de API**
5. Cole no `.env`

> **Cota gratuita:** 10.000 unidades/dia — cada requisição de comentários custa ~1 unidade por 100 comentários.

### 6. Rode o app

```bash
streamlit run app.py
```

O app abrirá automaticamente em `http://localhost:8501`.

---

## 📊 Dashboard — O que cada gráfico mostra

### 1. 🍩 Donut de Sentimentos
Proporção geral de comentários Positivos, Neutros e Negativos. Visão imediata da recepção do vídeo.

### 2. 📊 Sentimento ao Longo do Tempo
Barras empilhadas mostrando como os sentimentos evoluíram dia a dia (ou semana/mês). Útil para detectar mudanças após polêmicas ou respostas do criador.

### 3. 📈 Distribuição de Confiança do Modelo
Histograma dos scores de confiança (0.0 a 1.0):
- **Score alto (0.9+):** modelo muito confiante na classificação
- **Score baixo (0.5–0.6):** comentário ambíguo, irônico ou difícil de classificar

### 4. 🗺️ Mapa de Embeddings — Clusters Semânticos
Scatter plot 2D onde cada ponto é um comentário, colorido pelo cluster K-Means. O hover mostra o texto original. Revela os grandes temas dos comentários sem precisar ler nenhum.

### 5. 📦 Likes × Sentimento
Box plot da distribuição de likes por sentimento. Revela se comentários positivos ou negativos tendem a receber mais engajamento.

### 6. 📉 Volume de Comentários ao Longo do Tempo
Gráfico de área mostrando picos de engajamento ao longo do tempo.

### 7. 📋 Tabela Interativa
Tabela completa com filtros por sentimento, idioma e score mínimo, mais exportação em CSV.

---

## 🧩 Detalhamento dos Módulos

### `src/youtube_client.py`
Responsável por toda a comunicação com a YouTube Data API v3. Suporta todos os formatos de URL do YouTube (`watch?v=`, `youtu.be/`, `shorts/`, `embed/`). Implementa paginação automática via `nextPageToken` e sleep entre requisições para respeitar o rate limit. Possui fallback para vídeos com comentários desativados.

### `src/preprocessor.py`
Pipeline de NLP clássico aplicado a cada comentário:
- **Limpeza:** remove URLs, emojis, HTML entities, menções (`@`), hashtags (`#`) e números isolados
- **Detecção de idioma:** via `langdetect`, com fallback `"und"` para textos curtos
- **Tokenização:** divisão do texto em tokens individuais
- **Stopwords:** remoção por idioma via NLTK (pt, en, es, fr, de, it)
- **Lematização:** redução à forma base via spaCy (`correndo → correr`, `melhores → bom`)

### `src/sentiment.py`
Usa o modelo `cardiffnlp/twitter-xlm-roberta-base-sentiment` da HuggingFace:
- Treinado em dados de redes sociais (estilo próximo a comentários YouTube)
- Suporte multilíngue incluindo Português (PT-BR)
- Truncamento automático em 512 tokens
- Processamento em lotes (`batch_size=32`) para eficiência de memória
- GPU automático se disponível (CUDA), senão CPU

### `src/embeddings.py`
Pipeline de três etapas para geração e visualização de clusters:

**1. sentence-transformers** (`paraphrase-multilingual-MiniLM-L12-v2`)
- Gera vetores de 384 dimensões por comentário
- Comentários com significado parecido ficam matematicamente próximos

**2. UMAP** (Uniform Manifold Approximation and Projection)
- Comprime 384D → 2D preservando relações de vizinhança locais
- Não-linear: superior ao PCA para dados NLP

**3. K-Means** (scikit-learn)
- Agrupa os pontos 2D em K clusters (configurável: 2 a 10)
- Inicialização `k-means++` para maior estabilidade

### `src/cache_manager.py`
Sistema de cache em dois formatos:
- **DataFrame completo** → joblib (`.pkl`) com compressão nível 3, otimizado para arrays numpy
- **Metadados do vídeo** → JSON legível

Funções disponíveis: `cache_existe()`, `salvar_cache()`, `carregar_cache()`, `limpar_cache()`, `limpar_cache_completo()`, `listar_cache()`.

---

## 📦 Dependências

```
streamlit==1.35.0
google-api-python-client==2.131.0
youtube-transcript-api==0.6.2
transformers==4.41.2
sentence-transformers==3.0.1
torch==2.3.1
langdetect==1.0.9
nltk==3.8.1
spacy==3.7.5
plotly==5.22.0
pandas==2.2.2
scikit-learn==1.5.0
umap-learn==0.5.6
python-dotenv==1.0.1
joblib==1.4.2
```

---

## 🔐 Segurança

- A API Key do YouTube fica **exclusivamente** no arquivo `.env`, que está no `.gitignore`
- O arquivo `.env.example` serve como template sem dados sensíveis
- O diretório `cache/` também está no `.gitignore` (evita subir dados de análises)

---

## 📝 Observações

- A primeira execução para um vídeo pode levar alguns minutos (download dos modelos HuggingFace ~500MB + processamento NLP)
- Execuções subsequentes do mesmo vídeo são instantâneas (cache em disco)
- O modelo de embeddings (`MiniLM`) ocupa ~120MB; o transformer de sentimentos ~300MB
- Para vídeos com comentários em inglês, espanhol ou outros idiomas, o sistema funciona normalmente (modelos multilíngues)

---

*Desenvolvido como projeto acadêmico para a disciplina de Introdução à Inteligência Artificial.*
