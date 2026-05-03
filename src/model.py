import os

import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

ENV = os.getenv("ENV")
MLFLOW_REGISTRY_NAME = os.getenv("MLFLOW_REGISTRY_NAME")

# La vérification des variables est déjà faite dans __init__.py
mlflow.set_tracking_uri(os.getenv("MLFLOW_SERVER"))


class Model:
    """
    Charge le modèle depuis MLflow et expose une méthode predict.

    Pourquoi une classe ? Parce qu'on veut charger le modèle UNE SEULE FOIS
    au démarrage de l'API (c'est lourd de le recharger à chaque requête),
    et le garder en mémoire pour toutes les requêtes suivantes.
    """

    def __init__(self):
        self.model = None
        self.transform_pipeline = None
        self.load_model()

    def load_model(self):
        """
        Récupère depuis MLflow :
        - le modèle correspondant à l'alias ENV (staging ou production)
        - le transform_pipeline.pkl pour réappliquer le même encodage qu'à l'entraînement
        """
        client = MlflowClient()
        alias = ENV

        # On récupère la version du modèle associée à l'alias (staging ou production)
        model_version = client.get_model_version_by_alias(
            MLFLOW_REGISTRY_NAME,
            alias,
        )

        # On télécharge le transform_pipeline.pkl depuis les artifacts du run d'entraînement
        artifact_uri = f"runs:/{model_version.run_id}/transform_pipeline.pkl"
        pipeline_path = mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri)

        if pipeline_path is None:
            raise RuntimeError(
                f"Impossible de télécharger transform_pipeline.pkl "
                f"pour run_id={model_version.run_id}. "
                "Vérifiez que le pipeline training a bien loggé cet artifact."
            )

        # Chargement du modèle et du pipeline de transformation
        self.model = mlflow.sklearn.load_model(
            f"models:/{MLFLOW_REGISTRY_NAME}@{alias}"
        )
        self.transform_pipeline = joblib.load(pipeline_path)

    def predict(self, X):
        """
        Applique le même encodage qu'à l'entraînement puis calcule les prédictions.

        On doit réutiliser les MÊMES LabelEncoders qu'à l'entraînement :
        si 'samsung' était encodé en 0 à l'entraînement, il doit l'être aussi ici.
        C'est pourquoi on a sauvegardé le transform_pipeline dans MLflow.
        """
        if self.model is None:
            return None

        if self.transform_pipeline:
            # transform_pipeline est un dict {colonne: LabelEncoder}
            for name, encoder in self.transform_pipeline.items():
                X[name] = X[name].fillna("unknown")
                X[name] = encoder.transform(X[name])

        # On supprime les colonnes qui ne sont pas des features
        for col in ["user_id", "user_session", "purchased"]:
            if col in X.columns:
                X = X.drop(col, axis=1)

        return self.model.predict(X)
