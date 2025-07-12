import pandas as pd

from src.data_evaluation.evaluation import run_evaluation, display_results

GOLDEN_DATASET_PATH = "data/datasets/evaluation/golden_motions.csv"
if __name__ == "__main__":
    try:
        golden_df = pd.read_csv(GOLDEN_DATASET_PATH)
    except FileNotFoundError:
        print(
            f"""
            AVISO: Arquivo de teste não encontrado em '{GOLDEN_DATASET_PATH}'.
              Usando dados de exemplo."""
        )

    final_results = run_evaluation(golden_df)
    display_results(final_results)
