import time
from bs4 import BeautifulSoup

from urllib.parse import urlparse, parse_qs
from utils.request_wrapper import html_url_request


def get_topics_url(event_detail):

    if not event_detail or "id" not in event_detail:
        print("Falha: Detalhes do evento não disponíveis ou ID faltante.")
        return None

    event_id = event_detail["id"]
    event_url = f"https://www.camara.leg.br/evento-legislativo/{event_id}"

    event_html = html_url_request(event_url)

    print(f"Resgatando link das pautas do evento: {event_url}")
    if event_html:
        event_soup = BeautifulSoup(event_html, "html.parser")
        itens = event_soup.find_all(class_="links-adicionais__item")

        cleaned_links = {}
        for item in itens:
            texto = item.get_text(strip=True)
            link = item.find("a")["href"] if item.find("a") else None

            cleaned_links[texto] = link

        if "Pauta" in cleaned_links:
            print(f"Pauta base resgatada {cleaned_links["Pauta"]}")
            return cleaned_links["Pauta"]
        else:
            print(f"Falha: Evento de url {event_url} sem pauta")
            return None
    else:
        print(f"Falha: Evento de url {event_url} sem HTML tratável")
        return None


def get_true_topics_url(topics_url):
    if topics_url:
        parsed_topics_url = urlparse(topics_url)
        query_params = parse_qs(parsed_topics_url.query)

        codteor = query_params.get("codteor", [None])[0]

        if codteor:
            true_topics_url = (
                f"https://www.camara.leg.br/internet/ordemdodia/integras/"
                f"{codteor}.htm"
            )

            print(f"Link para pauta tratado: {true_topics_url}")
            return true_topics_url
        else:
            print(
                "Falha: Não foi possível tratar"
                " pauta código teor não encontrado."
            )
    return None


def get_topics_text(topics_url):
    topics_html = html_url_request(topics_url)

    if topics_html:
        print(f"Resgatando texto da pauta: {topics_url}")
        soup_topics = BeautifulSoup(topics_html, "html.parser")
        topics_text = soup_topics.get_text(separator="\n", strip=True)

        return topics_text
    else:
        print(f"Não foi possível resgatar o HTML da Pauta {topics_url}")


def get_event_topics(event):
    base_topics_url = get_topics_url(event)
    true_topics_url = get_true_topics_url(base_topics_url)

    topics_text = None
    if true_topics_url:
        topics_text = get_topics_text(true_topics_url)
    # time.sleep(0.5)  # Prevenir bloqueio por muitas requisições

    return topics_text
