import os
import chromadb
import openai
import pandas as pd

from datasets import Dataset

from dotenv import load_dotenv

from src.data_retrieval.llm_judge import filter_and_rerank_with_llm
from src.data_preprocess.feature_engineering import extract_semantic_features


def create_query_embedding(text: str, model_cpkt: str):
    """
    Gera o embedding para a consulta de texto.
    """
    query_dataset = Dataset.from_pandas(pd.DataFrame([{"query": text}]))

    query_embedding_df = extract_semantic_features(
        motions_dataset=query_dataset,
        column_name="query",
        model_ckpt=model_cpkt,
    )
    query_embedding = query_embedding_df["hidden_state"].iloc[0].tolist()

    return query_embedding


def search_motions(
    query_text: str, n_results: int, model: str, model_cpkt: str
):
    query_embedding = create_query_embedding(query_text, model_cpkt)

    client = chromadb.PersistentClient(path=f"artifacts/chroma_db_{model}")

    ementas_collection = client.get_collection(name="ementas")
    pdf_collection = client.get_collection(name="pdfs")

    ementa_results = ementas_collection.query(
        query_embeddings=query_embedding, n_results=n_results
    )

    pdf_results = pdf_collection.query(
        query_embeddings=query_embedding, n_results=n_results
    )

    return ementa_results, pdf_results


def retrieve_by_query(
    query,
    n_results,
    use_llm_judge=True,
    model="portuguese-bert",
    model_cpkt="neuralmind/bert-base-portuguese-cased",
):
    load_dotenv()

    ementa_results, pdf_results = search_motions(
        query, n_results, model, model_cpkt
    )

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
