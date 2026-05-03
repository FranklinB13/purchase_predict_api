import pandas as pd
from flask import Flask, jsonify, request

from src.model import Model

app = Flask(__name__)

# Le modèle est chargé une seule fois au démarrage de l'API
model = Model()


@app.route("/", methods=["GET"])
def home():
    return "OK !", 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Reçoit un DataFrame sérialisé en JSON et retourne les prédictions.

    Exemple de requête :
        requests.post("http://localhost:5000/predict", json=df.to_json())

    Les prédictions sont retournées sous forme de liste d'entiers (0 ou 1).
    On doit convertir en int car numpy int64 n'est pas sérialisable en JSON.
    """
    body = request.get_json()
    df = pd.read_json(body)
    results = [int(x) for x in model.predict(df).flatten()]
    return jsonify(results), 200


if __name__ == "__main__":
    app.run(port=5000)
