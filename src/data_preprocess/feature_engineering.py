import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
)
from datasets import Dataset

model_ckpt = "neuralmind/bert-base-portuguese-cased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(model_ckpt).to(device)


def tokenize(batch, text_column):
    texts = [str(text) for text in batch[text_column]]

    # Caso acima de 512 (limite de entrada do distilBERT trunca-se os dados)
    encodings = tokenizer(texts, padding=True, truncation=True, max_length=512)

    return encodings


def extract_hidden_states(batch):
    # Repassa entradas do modelo para o dispositivo (GPU ou CPU)
    inputs = {
        k: v.to(device)
        for k, v in batch.items()
        if k in tokenizer.model_input_names
    }

    with torch.no_grad():  # Congela o modelo BERT, para que não seja treinado
        # Queremos apenas obter o resultado da última camada oculta
        # Pois ela apresenta informações mais ricas após a passagem pelas
        # camadas do modelo.
        last_hidden_state = model(**inputs).last_hidden_state

    # Retornamos os valores como um array Numpy
    return {"hidden_state": last_hidden_state[:, 0].cpu().numpy()}


def extract_semantic_features(motions_dataset, column_name):
    text_encoded = motions_dataset.map(
        tokenize,
        batched=True,
        fn_kwargs={"text_column": column_name},
    )

    text_encoded.set_format("torch", columns=["input_ids", "attention_mask"])
    hidden_features = text_encoded.map(extract_hidden_states, batched=True)

    df_embbeding = hidden_features.to_pandas()

    return df_embbeding[["motion_id", "hidden_state"]]


def create_embeddings(motions_df):
    motions_dataset = Dataset.from_pandas(motions_df)

    ementa_embeddigns = extract_semantic_features(
        motions_dataset, "ementa_limpa"
    )
    pdf_embeddings = extract_semantic_features(
        motions_dataset, "pdf_text_limpo"
    )

    motions_with_embeddings = motions_df.merge(
        ementa_embeddigns, on="motion_id", how="left"
    ).merge(
        pdf_embeddings,
        on="motion_id",
        suffixes=("_ementa", "_pdf"),
        how="left",
    )

    return motions_with_embeddings


def create_motion_id(motions_df):
    def transform_id(row):
        if row.get("titulo"):
            motion_id = row["titulo"].replace(" ", "_").replace("/", "_")
        else:
            motion_id = "sem_titulo"
        return motion_id

    motions_df["motion_id"] = motions_df.apply(transform_id, axis=1)

    return motions_df
