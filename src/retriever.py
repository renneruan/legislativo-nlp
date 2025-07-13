import pandas as pd
from src.data_retrieval.similarity_retriever import retrieve_by_query

events = pd.read_csv("data/datasets/raw/events_with_topics.csv")
motions = pd.read_csv("data/datasets/cleaned/motions_cleaned.csv")


def add_event_data(row):

    event = events[events["id"] == row["event_id"]].iloc[0]
    row["event_id"] = event["id"]
    row["event_url"] = event["uri"]
    row["event_description"] = event["descricao"]
    row["event_status"] = event["situacao"]

    return row


def retrieve_legislative_events(query, k=5, use_llm=True):
    results_ementa, results_pdf = retrieve_by_query(
        query, n_results=k, use_llm_judge=use_llm
    )

    return_dict = {}

    # Primeiro analisa os pdfs depois as ementas (a ordem importa)
    for result in results_pdf + results_ementa:
        if result["id"] not in return_dict:
            print(result["id"])
            motion_id = result["id"]

            filtered_motion = motions[motions["motion_id"] == motion_id].iloc[
                0
            ]
            filtered_motion = add_event_data(filtered_motion)

            return_dict[result["id"]] = filtered_motion.to_dict()
            # print(results_ementa, results_pdf)

    return return_dict
