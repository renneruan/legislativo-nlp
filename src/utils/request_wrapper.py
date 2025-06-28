import requests


def simple_url_request(url):
    try:
        response = requests.get(url, headers={"Accept": "application/json"})
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Erro na requisição para {url}: {response.status_code}")
            return None
    except Exception as e:
        print(f"Exceção ao acessar {url}: {e}")
        return None


def html_url_request(url):
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.text
        else:
            print(f"Erro na requisição para {url}: {response.status_code}")
            return None
    except Exception as e:
        print(f"Exceção ao acessar {url}: {e}")
        return None


def pdf_url_request(url):
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/pdf"}

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.content
        else:
            print(f"Erro ao baixar PDF de {url}: {response.status_code}")
            return None
    except Exception as e:
        print(f"Exceção ao baixar PDF de {url}: {e}")
        return None
