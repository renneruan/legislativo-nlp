from tqdm import tqdm
import os

import pandas as pd

from src.data_collect.events import get_events_with_details
from src.data_collect.topics import get_event_topics
from src.data_collect.motions import get_events_motions

from src.data_preprocess.cleaning import clean_motions, remove_empty_texts
from src.data_preprocess.feature_engineering import (
    create_embeddings,
    create_motion_id,
)

from src.data_preprocess.embedding_store import save_embeddings_to_chroma

tqdm.pandas()


def main():
    # events_df = get_events_with_details()
    # events_df["topics"] = events_df["details"].progress_apply(get_event_topics)

    # os.makedirs("data/datasets/raw", exist_ok=True)
    # os.makedirs("data/datasets/cleaned", exist_ok=True)
    # os.makedirs("data/datasets/embeddings", exist_ok=True)

    # events_df.to_csv(
    #     "data/datasets/raw/events_with_topics.csv",
    #     index=False,
    #     encoding="utf-8-sig",
    # )

    # motions = get_events_motions(events_df)
    # motions.to_csv(
    #     "data/datasets/raw/motions.csv", index=False, encoding="utf-8-sig"
    # )

    motions = pd.read_csv(
        "data/datasets/raw/motions.csv", encoding="utf-8-sig"
    )

    clean_motions_df = clean_motions(motions)
    clean_motions_df = remove_empty_texts(clean_motions_df)
    clean_motions_df = create_motion_id(clean_motions_df)
    clean_motions_df.to_csv(
        "data/datasets/cleaned/motions_cleaned.csv",
        index=False,
        encoding="utf-8-sig",
    )

    motions_with_embeddings_df = create_embeddings(clean_motions_df)
    motions_with_embeddings_df.to_csv(
        "data/datasets/embeddings/motions_with_embeddings.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_embeddings_to_chroma(motions_with_embeddings_df)


if __name__ == "__main__":
    main()
