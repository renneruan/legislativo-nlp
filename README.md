# Análise e Busca Semântica de Documentos Legislativos

Este repositório contém o desenvolvimento do projeto final de pós-graduação do curso de Ciência de Dados e Machine Learning do CEUB, focado na aplicação de técnicas de Processamento de Linguagem Natural (PLN) para criar um sistema de busca dos Eventos e Requerimentos da Câmara dos Deputados. O objetivo é superar as limitações da busca manual, acessando a agenda semanalmente, aderindo a técnicas de coleta de dados e permitindo que os usuários encontrem documentos mais aderentes a um tema desejado. Isso possibilita um melhor e mais fácil acompanhamento das tramitações.

## Principais Funcionalidades

- **Coleta de Dados:** Scripts para extrair dados de eventos e requerimentos da API de Dados Abertos da Câmara, utilizando também técnicas de Webscrapping para dados quebrados ou indisponíveis.
- **Pré-processamento:** Limpeza de texto utilizando spaCy, incluindo lematização e uma lista de _stopwords_ customizada para o domínio legislativo.
- **Feature Engineering:** Geração de embeddings semânticos a partir do texto dos documentos (`ementa` e `texto completo`) utilizando modelos de linguagem baseados no BERT.
- **Busca Vetorial:** Indexação dos embeddings em banco de dados vetorial (ChromaDB) para permitir as buscas por similaridade.
- **LLM-as-a-Judge:** Utilização de Large Language Model (OpenAI GPT) para re-rankear e avaliar a relevância dos resultados retornados pela busca vetorial.
- **Pipeline de Avaliação:** Avaliação do desempenho do sistema de busca, utilizando métricas padrão de Recuperação de Informação como `Precision@K`, `Recall@K` e `MRR`, em um conjunto de testes manualmente criado.
- **API e Página de Utilização** Utilizado o framework Flask para a construção de uma página personalizada para a utilização do projeto, possibilitando resgatar os dados para um período específico e recuperar os eventos de acordo com o tema fornecido pelo usuário. Além da página, os endpoints também podem ser utilizados para este fim.

## Estrutura do Projeto

O projeto segue uma estrutura modular.

```
legislativo-nlp/
├── artifacts/              # Pasta de artefatos, irá armazenas os bancos de dados vetoriais criados pelo Chroma
├── data/
│   └── datasets/           # CSVs com dados brutos e processados
|       ├── cleaned/        # Dados limpos após pre-processamento
|       ├── embeddings/     # Dados após criação de embeddings
|       ├── evaluation/     # Dados necessários para avaliação (Únicos persistidos)
│       └── raw /           # Dados brutos colhidos da API e Webscrap
├── notebooks/              # Notebooks de interesse, seguindo as seções da metodologia CRISP-DM
|   ├── business_and_data_understanding.ipynb
|   ├── exploratory_data_analysis.ipynb
|   └── modeling_evaluation.ipynb
├── src/
|   ├── data_collect/       # Módulo de coleta de dados
│   ├── data_preprocess/    # Módulo para limpeza e feature engineering
│   ├── data_retrieval/     # Móduls para busca e interação com o ChromaDB
│   ├── data_evaluation/    # Script e lógica de avaliação de performance dos modelos de embedding
|   ├── build.py            # Script principal para resgatar os dados e construir o banco vetorial
|   ├── evaluation.py       # Script para executar as etapas de avaliação
|   └── retriever.py        # Script principal para orquestrar etapa de recuperação dos dados
├── static/                 # Pasta contendo imagens e outros arquivos estáticos
├── templates/              # Estrutura HTML a ser utilizada pelo flask
├── .env                    # Arquivo env contendo API para OpenAI
├── app.py                  # Script contendo endpoints da API
├── requirements.txt        # Dependências do projeto
└── README.md
```

Se torna necessário possuir um .env para utilização do LLM como Juiz.

## Metodologia

O desenvolvimento seguiu as fases do ciclo de vida **CRISP-DM (Cross-Industry Standard Process for Data Mining)**, desde o entendimento do problema até a avaliação formal dos resultados e a estruturação para um potencial deploy. Mesmo não seguindo uma abordagem de Machine Learning usual, em que se ocorre um treinamento de um modelo a partir de dados tabulares, temos a etapa de Modelagem atrelada a escolha do modelo de embedding utilizado. Foi uma abordagem de projeto iterativa e cíclica, onde foi preciso retornar em diversas etapas modificando e melhorando as estruturas.

## Tecnologias Utilizadas

- **Linguagem:** Python 3.11+
- **Bibliotecas Principais:** Pandas, Spacy, PyTorch
- **NLP & Embeddings:** Hugging Face `Transformers`, `Datasets`, Modelo baseado no XLM-RoBERTa com treinamento multilingual
- **Banco Vetorial:** ChromaDB
- **LLM-as-a-Judge:** OpenAI API
- **Servidor Web:** Flask

## Setup e Instalação

Siga os passos abaixo para configurar o ambiente e executar o projeto.

**Ativação de ambiente virtual**

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

**Instalação de Dependências**

```bash
pip install -r requirements.txt
```

## Como Executar o Projeto

É recomendado que os 2 primeiros passos sejam feitos utilizando a interface disposta pelo servidor web, onde poderá ser melhor visualizado os resultados.

**1. Resgate dos Dados e construção do Banco de Dados Vetorial**

Execute o script `src/build.py` para coletar, pré-processar e criar os dados no banco ChromaDB.

```bash
python -m src.build
```

**2. Avaliação de Performance**

Para rodar a avaliação dos diferentes modelos, execute o script de avaliação.

```bash
python -m src.evaluation
```

**3. Executando a API Web (Flask)**

O projeto inclui uma API web construída com Flask para servir o modelo de busca e permitir a reindexação dos dados.

**a) Iniciar o Servidor**

Para iniciar o servidor web, execute o arquivo `app.py` a partir do diretório raiz.

```bash
python app.py
```

A página HTML gerada estará disponível em `http://localhost:8080`.

A seguir temos um vídeo exemplificando o uso, assim como o resultado da busca. Algumas partes do vídeo foram recortadas para minimizar o tempo de espera para criação do banco por exemplo, é pensado que esta etapa seja feita por um orquestrador em um ambiente com GPU, uma vez que ela envolve coleta de dados e técnicas de Deep Learning, acelerando o processo e permitindo o uso facilitado após criação.

https://github.com/user-attachments/assets/900604d6-ec95-481a-90af-fa21cfdc2582

**b) Endpoints da API**

A API expõe os seguintes endpoints:

- **`GET /`**

  - **Descrição:** Renderiza a página principal da aplicação (o arquivo `index.html`).
  - **Uso:** Acesse `http://localhost:8080` em seu navegador.

- **`POST /build_data`**

  - **Descrição:** Dispara o processo completo de coleta, pré-processamento e indexação dos dados no ChromaDB. Pode levar vários minutos para ser concluído.
  - **Uso (via cURL):**
    ```bash
    curl -X POST http://localhost:8080/build_data
    ```

- **`POST /retrieve`**

  - **Descrição:** Realiza uma busca semântica com base em uma query enviada no corpo da requisição.
  - **Corpo da Requisição (JSON):**
    ```json
    {
      "query": "sua pergunta sobre o tema legislativo"
    }
    ```
  - **Uso (via cURL):**
    ```bash
    curl -X POST -H "Content-Type: application/json" \
    -d '{"query": "políticas públicas para o meio ambiente"}' \
    http://localhost:8080/retrieve
    ```

## Resultados e Conclusões

A fase de avaliação comparou quatro modelos de embedding (`e5-base`, `bge-m3`, `miniLM`, `portuguese-bert`) em quatro estratégias diferentes. As principais conclusões foram:

1.  **Melhor Modelo:** O modelo **`bge-m3`** apresentou o melhor desempenho geral, com uma `Precision@5` de **0.75** e `Recall@5` de **0.735**.
2.  **Melhor Fonte de Dados:** Indexar o **texto completo dos PDFs de Requerimentos** gerou resultados significativamente mais precisos do que usar apenas as ementas.
3.  **Impacto do LLM Juiz:** A camada experimental do LLM-as-a-Judge, na sua configuração atual, **reduziu a performance** da busca. Isso indica a necessidade de um refinamento no prompt, nos critérios de avaliação ou no modelo de LLM utilizado para a tarefa de julgamento.

## Próximos Passos

- Iterar sobre os prompts e testar modelos LLM mais avançados (ex: `gpt-4o`) para melhorar a capacidade de re-ranking.
- Buscar estratégias de classificação ou predição de votações baseando-se no contexto semântico.
- Granular para os tipos diferentes de eventos presentes na câmara.
