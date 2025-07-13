import json
import openai


def get_openai_judgment(
    query_text: str,
    document_text: str,
    client: openai.OpenAI,
    model_name: str = "gpt-3.5-turbo",
) -> dict:
    """
    Envia a query e o documento para a API da OpenAI e retorna uma avaliação.

    Args:
        query_text: O texto da busca original do usuário.
        document_text: O texto do documento recuperado (ex: a ementa).
        client: Uma instância do cliente da OpenAI.
        model_name: O nome do modelo a ser usado.

    Returns:
        Um dicionário com 'score' e 'justificativa'
    """

    system_prompt = """
    Você é um analista legislativo sênior, especialista em interpretar
    a relevância de documentos da Câmara dos Deputados. Sua tarefa é
    avaliar o quão bem um documento adere a um termo de pesquisa do usuário,
    fornecendo uma nota de 0 a 10.

    Tente também avaliar se o que é presente no documento está dentro do mesmo
    macro tema, ou tem intereseccionalidades com o termo pesquisado.

    Critérios de Pontuação:
    - 10: Relevância Direta: O documento tem aderência perfeita ao termo pesquisado.
    - 7-9: Relevância Alta: O documento trata do tópico principal
      que foi pesquisado, ou temáticas inerentes a ele.
    - 5-6: Relevância Parcial: O documento menciona o tópico,
      mas de forma secundária, e possui foco em outro contexto.
    - 2-4: Relevância Baixa: O documento contém poucas ou nenhuma palavras-chave
      relativo ao termo pesquisado.
    - 0-1: Irrelevante.
    """

    user_prompt = f"""
    Termos de pesquisa do Usuário:
    "{query_text}"

    Texto do Documento:
    "{document_text}"

    Sua Avaliação (responda APENAS em formato JSON com as chaves
      "score" e "justificativa"):
    """

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,  # Temperatura 0 para ser mais consistente e direto
            response_format={"type": "json_object"},
        )

        judgment = json.loads(response.choices[0].message.content)

        print(judgment)
        return judgment

    except Exception as e:
        print(f"[Juiz OpenAI] Erro ao avaliar: {e}")
        return {"score": 0, "justificativa": "Erro na avaliação do LLM."}


def filter_and_rerank_with_llm(
    initial_results: dict,
    query_text: str,
    client: openai.OpenAI,
    relevance_threshold: int = 7,
) -> list:
    """
    Recebe os resultados brutos do ChromaDB, usa o LLM para julgá-los,
    filtra e reordena a lista final.

    Args:
        initial_results: O dicionário de resultados da consulta ao ChromaDB.
        query_text: O texto da busca original do usuário.
        client: Uma instância do cliente da OpenAI.
        relevance_threshold: A nota de corte para considerar
          um resultado relevante.

    Returns:
        Uma lista de dicionários, contendo os resultados
          validados e reordenados.
    """

    validated_results = []
    if initial_results.get("ids") and initial_results["ids"][0]:
        # print(f"Analisando {len(initial_results['ids'][0])} resultados.")

        for i, doc_id in enumerate(initial_results["ids"][0]):
            document_text = initial_results["documents"][0][i]
            judgment = get_openai_judgment(query_text, document_text, client)

            # print(
            #     f"ID: {doc_id[:30]}... | Score do Juiz: {judgment.get(
            #         'score', 0
            #     )}"
            # )

            if judgment.get("score", 0) >= relevance_threshold:
                result_item = {
                    "id": initial_results["metadatas"][0][i]["motion_id"],
                    "document": document_text,
                    "metadata": initial_results["metadatas"][0][i],
                    "distance": initial_results["distances"][0][i],
                    "llm_score": judgment.get("score"),
                    "llm_justificativa": judgment.get("justificativa"),
                }
                validated_results.append(result_item)

    validated_results.sort(key=lambda x: x["llm_score"], reverse=True)

    # print(
    #     f"{len(validated_results)} resultados finais após filtro e re-ranking"
    # )
    return validated_results
