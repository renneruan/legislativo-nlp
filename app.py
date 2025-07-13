"""
Módulo com funções de endpoint a serem servidas pela API Flask.
"""

from flask import Flask, jsonify, render_template, request

from src.build import collect_and_build_data
from src.retriever import retrieve_legislative_events

app = Flask(__name__)


@app.route("/", methods=["GET"])
def home_page():
    """Home page a ser renderizada"""
    return render_template("index.html")


@app.route("/build_data", methods=["POST"])
def build_data():
    try:
        sizes = collect_and_build_data()

        return jsonify({"success": True, "data": sizes}), 200
    except ValueError as e:
        print(f"Não foi possível criar dados: {e}")
        return "Falha na montagem", 500


@app.route("/retrieve", methods=["POST"])
def retrieve():
    try:
        data = request.get_json()
        query = data.get("query")

        if not query:
            return jsonify({"error": "A query de busca é obrigatória."}), 400

        return_data = retrieve_legislative_events(query)
        print(len(return_data))
        return jsonify({"success": True, "data": return_data}), 200
    except Exception as e:
        print(f"Não foi possível recuperar dados: {e}")
        return "Falha na busca", 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
    # app.run(host="0.0.0.0", port=8080)
