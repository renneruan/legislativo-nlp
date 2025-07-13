import pandas as pd

from src.data_evaluation.evaluation import run_evaluation, display_results

GOLDEN_GROUND_TRUTH_PATH = "data/datasets/evaluation/golden_motions.csv"
GOLDEN_CLEANED_PATH = "data/datasets/evaluation/golden_motions_cleaned.csv"


EMBEDDING_MODELS_TO_BE_TESTED = {
    "e5-base": "intfloat/multilingual-e5-base",
    "bge-m3": "BAAI/bge-m3",
    "miniLM": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "portuguese-bert": "neuralmind/bert-base-portuguese-cased",
}


def evaluate_multiple_models():
    try:
        golden_ground_truth_df = pd.read_csv(GOLDEN_GROUND_TRUTH_PATH)
        golden_cleaned_motions_df = pd.read_csv(GOLDEN_CLEANED_PATH)
    except FileNotFoundError:
        print("AVISO: Arquivos de avaliação não encontrados.")

    for model in EMBEDDING_MODELS_TO_BE_TESTED:
        final_results = run_evaluation(
            golden_ground_truth_df,
            golden_cleaned_motions_df,
            model,
            EMBEDDING_MODELS_TO_BE_TESTED[model],
        )

        print("\n")
        print(f"Resultados de avaliação para {model}")
        display_results(final_results)


if __name__ == "__main__":
    evaluate_multiple_models()
