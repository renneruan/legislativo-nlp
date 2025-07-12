import os
import chromadb
import torch
import openai

from dotenv import load_dotenv

from transformers import (
    AutoModel,
    AutoTokenizer,
)

from src.data_retrieval.llm_judge import filter_and_rerank_with_llm


model_ckpt = "neuralmind/bert-base-portuguese-cased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(model_ckpt).to(device)


def create_query_embedding(text: str):
    """
    Gera o embedding para a consulta de texto.
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0].cpu().numpy()

    return embedding.tolist()


def search_motions(query_text: str, n_results: int):
    query_embedding = create_query_embedding(query_text)

    client = chromadb.PersistentClient(path="artifacts/chroma_db")

    ementas_collection = client.get_collection(name="ementas")
    pdf_collection = client.get_collection(name="pdfs")

    ementa_results = ementas_collection.query(
        query_embeddings=query_embedding, n_results=n_results
    )

    pdf_results = pdf_collection.query(
        query_embeddings=query_embedding, n_results=n_results
    )

    return ementa_results, pdf_results


def retrieve_by_query(query, n_results, use_llm_judge=True):
    load_dotenv()

    ementa_results, pdf_results = search_motions(query, n_results)

    if use_llm_judge and os.getenv("OPENAI_API_KEY"):
        client_openai = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        ementa_results_reranked = filter_and_rerank_with_llm(
            initial_results=ementa_results,
            query_text=query,
            client=client_openai,
        )

        # for item in ementa_results_reranked:
        #     print(f"\nID: {item['id']} | Score LLM: {item['llm_score']}")
        #     print(f"  Justificativa: {item['llm_justificativa']}")
        #     print(f"  Texto: {item['document']}")

        pdf_results_reranked = filter_and_rerank_with_llm(
            initial_results=pdf_results,
            query_text=query,
            client=client_openai,
        )

        # for item in pdf_results_reranked:
        #     print(f"\nID: {item['id']} | Score LLM: {item['llm_score']}")
        #     print(f"  Justificativa: {item['llm_justificativa']}")

        return ementa_results_reranked, pdf_results_reranked

    else:
        ids = ementa_results["ids"][0]
        docs = ementa_results["documents"][0]
        dists = ementa_results["distances"][0]

        # for i in range(len(ids)):
        #     print(f"\nID: {ids[i]}")
        #     print(f"Distância: {dists[i]:.2f}")
        #     print(f"Texto: {docs[i][:200]}...")

        ids = pdf_results["ids"][0]
        docs = pdf_results["documents"][0]
        dists = pdf_results["distances"][0]

        # for i in range(len(ids)):
        #     print(f"\nID: {ids[i]}")
        #     print(f"Distância: {dists[i]:.2f}")
        #     print(f"Texto: {docs[i][:200]}...")

        return ementa_results, pdf_results
