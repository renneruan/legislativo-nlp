import spacy
import spacy.cli

try:
    nlp = spacy.load("pt_core_news_sm")
except OSError:
    spacy.cli.download("pt_core_news_sm")
    nlp = spacy.load("pt_core_news_sm")

custom_motions_stopwords = [
    # Termos Formais e Estruturais
    "apresentação",
    "excelência",
    "gabinete",
    "inciso",
    "justificativa",
    "lexedit",
    "nº",
    "nobre",
    "pares",
    "presidente",
    "sala",
    "senhor",
    "senhora",
    "sr",
    "termos",
    "líder",
    "vossa",
    # Cargos e Títulos
    "dep",
    "deputada",
    "deputado",
    "deputados",
    "prefeito",
    "representante",
    "vereadores",
    # Verbos de Ação Legislativa
    "contar",
    "convidar",
    "ouvir",
    "requer",
    "requerer",
    "requeiro",
    "realização",
    # Termos Legislativos e Jurídicos
    "caput",
    "artigo",
    "artigos",
    "câmara",
    "deliberativa",
    "constituição",
    "fulcro",
    "interno",
    "oitiva",
    "plenário",
    "pública",
    "público",
    "regimento",
    "requerimento",
    "turno",
    "turnos",
    "comissão",
    "especial",
    "audiência",
    "sessão",
    "sessões",
    "âmbito",
    "regimento",
    "interno",
    # Entidades e Setores Genéricos
    "estado",
    "federal",
    "governo",
    "privado",
    # Ruído de Rodapé/Metadados
    "assinado",
    "eletronicamente",
]

for word in custom_motions_stopwords:
    nlp.vocab[word].is_stop = True


def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    doc = nlp(text.lower())
    tokens = [
        # Divisão (tokenização) e Lematização
        token.lemma_
        for token in doc
        # Verifica apenas alfabéticos, se não é stop word e se o tamanho é
        # maior que 2
        if token.is_alpha and not token.is_stop and len(token.text) > 2
    ]
    return " ".join(tokens)


def clean_motions(motions_df):
    """
    Limpa as colunas de texto dos requerimentos
    """
    motions_df["ementa_limpa"] = (
        motions_df["ementa"].fillna("").apply(preprocess_text)
    )
    motions_df["pdf_text_limpo"] = (
        motions_df["pdf_text"].fillna("").apply(preprocess_text)
    )

    return motions_df


def remove_empty_texts(motions_df):
    """
    Remove linhas onde as colunas de texto estão vazias
    """
    print("Tamanho inicial do DataFrame:", len(motions_df))

    motions_df = motions_df.dropna(subset=["ementa_limpa", "pdf_text_limpo"])

    motions_df = motions_df[
        (motions_df["ementa_limpa"] != "")
        & (motions_df["pdf_text_limpo"] != "")
    ].reset_index(drop=True)

    print("Tamanho após remoção de textos vazios:", len(motions_df))

    return motions_df
