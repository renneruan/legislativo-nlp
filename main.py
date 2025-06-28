from tqdm import tqdm

from src.data_collect.events import get_events_with_details
from src.data_collect.topics import get_event_topics
from src.data_collect.motions import get_events_motions

tqdm.pandas()


def main():
    events_df = get_events_with_details()
    events_df["topics"] = events_df["details"].progress_apply(get_event_topics)
    events_df.to_csv(
        "events_with_topics.csv", index=False, encoding="utf-8-sig"
    )

    motions = get_events_motions(events_df)
    motions.to_csv("motions.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
