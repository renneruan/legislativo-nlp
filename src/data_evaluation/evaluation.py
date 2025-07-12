import pandas as pd
import numpy as np

from typing import List, Dict, Tuple

from src.data_retrieval.similarity_retriever import retrieve_by_query

K_FOR_METRICS = 5  # Vamos calcular a Precision@K e Recall@K para K=8


def calculate_precision_at_k(
    retrieved_ids: list, relevant_ids: list, k: int
) -> float:
    if k == 0:
        return 0.0
    top_k_retrieved = retrieved_ids[:k]
    hits = len(set(top_k_retrieved) & set(relevant_ids))
    return hits / k


def calculate_recall_at_k(
    retrieved_ids: list, relevant_ids: list, k: int
) -> float:
    if not relevant_ids:
        return 0.0
    top_k_retrieved = retrieved_ids[:k]
    hits = len(set(top_k_retrieved) & set(relevant_ids))
    return hits / len(relevant_ids)


def calculate_reciprocal_rank(
    retrieved_ids: list, relevant_ids: list
) -> float:
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def compute_metrics(
    retrieved_ids: List[str], relevant_ids: List[str], k: int
) -> Dict[str, float]:

    return {
        f"Precision@{k}": calculate_precision_at_k(
            retrieved_ids, relevant_ids, k
        ),
        f"Recall@{k}": calculate_recall_at_k(retrieved_ids, relevant_ids, k),
        "MRR": calculate_reciprocal_rank(retrieved_ids, relevant_ids),
    }


def evaluate_variant(
    query: str, relevant_ids: List[str], use_llm: bool, k: int
) -> Tuple[str, Dict[str, float]]:

    results_ementa, results_pdf = retrieve_by_query(
        query, n_results=k, use_llm_judge=use_llm
    )

    if use_llm:
        retrieved_ids_ementa = [result["id"] for result in results_ementa]
        retrieved_ids_pdf = [result["id"] for result in results_pdf]

    else:
        retrieved_ids_ementa = [
            meta["motion_id"] for meta in results_ementa["metadatas"][0]
        ]
        retrieved_ids_pdf = [
            meta["motion_id"] for meta in results_pdf["metadatas"][0]
        ]

    ementa_metrics = compute_metrics(retrieved_ids_ementa, relevant_ids, k)
    pdf_metrics = compute_metrics(retrieved_ids_pdf, relevant_ids, k)

    return [
        {
            "key": "ementa" + ("_llm" if use_llm else "_baseline"),
            "metrics": ementa_metrics,
        },
        {
            "key": "pdf" + ("_llm" if use_llm else "_baseline"),
            "metrics": pdf_metrics,
        },
    ]


def run_evaluation(
    golden_dataset: pd.DataFrame,
) -> Dict[str, Dict[str, float]]:
    """
    Executa o processo de avaliação completo em diferentes variantes:
    - ementa
    - pdf
    - ementa+pdf
    - com/sem LLM juiz
    """
    all_metrics = {}
    USE_JUDGE = [False, True]

    for _, row in golden_dataset.iterrows():
        query = row["query"]
        relevant_ids = eval(row["relevant_motion_ids"])
        print(f"\nAvaliando a Query: '{query}'")

        for use_llm in USE_JUDGE:
            metrics = evaluate_variant(
                query, relevant_ids, use_llm, K_FOR_METRICS
            )

            for obj in metrics:
                if obj["key"] not in all_metrics:
                    all_metrics[obj["key"]] = {k: [] for k in obj["metrics"]}

                for m_name, m_value in obj["metrics"].items():
                    all_metrics[obj["key"]][m_name].append(m_value)

            # print(all_metrics)

    # Agregação final (média)
    final_metrics = {
        variant: {
            metric: np.mean(values) for metric, values in metric_dict.items()
        }
        for variant, metric_dict in all_metrics.items()
    }

    return final_metrics


def display_results(metrics: dict):
    """Apresenta os resultados finais em um formato legível."""
    print("\n\n" + "=" * 60)
    print("           RESULTADO FINAL DA AVALIAÇÃO")
    print("=" * 60)

    df = pd.DataFrame(metrics).round(3)
    print(df)

    print("\n" + "=" * 60)
