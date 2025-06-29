import pandas as pd
import requests

from src.utils.date import get_weekly_dates
from src.utils.request_wrapper import simple_url_request


def get_events(start_date, end_date, itens_per_page=100):
    url = "https://dadosabertos.camara.leg.br/api/v2/eventos"
    headers = {"Accept": "application/json"}

    page = 1
    events = []
    while page < 2:
        params = {
            "dataInicio": start_date,
            "dataFim": end_date,
            "ordenarPor": "dataHoraInicio",
            "itens": itens_per_page,
            "ordem": "ASC",
            "pagina": page,
        }

        response = requests.get(url, params=params, headers=headers)

        if response.status_code != 200:
            print(f"Erro ao requisitar página {page}: {response.status_code}")
            break

        data = response.json()["dados"]

        if not data:
            break

        events.extend(data)
        print(f"Página {page} carregada com {len(data)} eventos.")
        page += 1

    return events


def get_event_detail(event_uri):
    detail = simple_url_request(event_uri)

    return detail["dados"] if detail is not None else None


def get_events_with_details():
    start_date, end_date = get_weekly_dates()

    print(f"Data de início: {start_date}")
    print(f"Data de fim: {end_date}")

    events = get_events(start_date, end_date)

    events_df = pd.DataFrame(events)
    events_df["details"] = events_df["uri"].apply(get_event_detail)

    print(f"Total de eventos coletados: {len(events_df)}")

    return events_df
