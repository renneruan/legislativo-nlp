import chromadb
from chromadb.errors import NotFoundError


def save_embeddings_to_chroma(embeddings_df, embedding_key="portuguese-bert"):
    client = chromadb.PersistentClient(
        path=f"artifacts/chroma_db_{embedding_key}"
    )

    try:
        client.delete_collection(name="ementas")
        client.delete_collection(name="pdfs")
        print(f"INFO: Coleção limpa para ChromaDB '{embedding_key}'.")
    except NotFoundError:
        # Caso já exista, só ignora
        pass

    ementas_collection = client.get_or_create_collection("ementas")
    pdf_collection = client.get_or_create_collection("pdfs")

    for _, row in embeddings_df.iterrows():
        ementas_collection.upsert(
            documents=[row["ementa_limpa"]],
            embeddings=[row["hidden_state_ementa"].tolist()],
            metadatas={"motion_id": row["motion_id"], "source": "ementa"},
            ids=[str(row["motion_id"]) + "_ementa"],
        )

        pdf_collection.upsert(
            documents=[row["pdf_text_limpo"]],
            embeddings=[row["hidden_state_pdf"].tolist()],
            metadatas={"motion_id": row["motion_id"], "source": "pdf"},
            ids=[str(row["motion_id"]) + "_pdf"],
        )

    ementas_size = ementas_collection.count()
    pdf_size = pdf_collection.count()

    print(f"Ementas: {ementas_size} documentos")
    print(f"PDFs: {pdf_size} documentos")

    return ementas_size, pdf_size
