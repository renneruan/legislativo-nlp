import pandas as pd
import time
from tqdm import tqdm
import os

from pdfminer.high_level import extract_text

from utils.request_wrapper import simple_url_request, pdf_url_request

tqdm.pandas()


def get_motion_detail(uri):
    time.sleep(0.5)  # evitar sobrecarregar a API

    motion_details = simple_url_request(uri)

    return motion_details["dados"] if motion_details is not None else None
    # return motion_details


def get_motion_additional_data(motion):
    details = motion.get("details", {})

    status_description = None
    ementa = None
    pdf_url = None
    motion_id = None

    if isinstance(details, dict):
        status = details.get("statusProposicao", None)
        if status:
            status_description = status.get("descricaoSituacao")

        ementa = details.get("ementa", None)
        pdf_url = details.get("urlInteiroTeor", None)

        numero = details.get("numero", "")
        ano = details.get("ano", "")

        if numero and ano:
            motion_id = f"{numero}_{ano}"

    motion["status"] = status_description
    motion["ementa"] = ementa
    motion["pdf_url"] = pdf_url
    motion["id"] = motion_id

    return motion


def explode_events_motions(events_df):
    motions = []

    for _, row in events_df.iterrows():
        event_id = row["id"]
        reqs = row.get("details", {}).get("requerimentos", [])
        for r in reqs:
            r["event_id"] = event_id
            motions.append(r)

    motions_df = pd.DataFrame(motions)

    return motions_df


def download_motions(motions_df, output_dir="data/motions"):
    os.makedirs(output_dir, exist_ok=True)

    print("Iniciando o download dos requerimentos em PDF.")
    pdf_texts = []
    for _, row in tqdm(motions_df.iterrows(), total=len(motions_df)):
        pdf_url = row.get("pdf_url")
        motion_id = row.get("id")

        if not pdf_url or not isinstance(pdf_url, str):
            print(f"PDF URL inválido ou ausente para {motion_id}.")
            pdf_texts.append("")
            continue

        try:
            motion_name = os.path.join(output_dir, f"{motion_id}.pdf")

            if os.path.exists(motion_name):
                print(f"Arquivo {motion_name} já existe. Pulando download.")
            else:
                motion_pdf = pdf_url_request(pdf_url)
                if motion_pdf:
                    with open(motion_name, "wb") as f:
                        f.write(motion_pdf)

                    print(f"Baixado {motion_id} em {motion_name} com sucesso.")

                else:
                    print(f"Erro ao baixar {motion_id}")
                    pdf_texts.append("")
                    continue

            pdf_texts.append(extract_text(motion_name))
        except Exception as e:
            pdf_texts.append("")
            print(f"Erro em {motion_id}: {e}")

    motions_df["pdf_text"] = pdf_texts


def get_events_motions(events_df):
    motions_df = explode_events_motions(events_df)

    motions_df["details"] = motions_df["uri"].progress_apply(get_motion_detail)

    motions_df = get_motion_additional_data(motions_df)
    motions_df = motions_df.apply(get_motion_additional_data, axis=1)

    download_motions(motions_df)

    return motions_df
