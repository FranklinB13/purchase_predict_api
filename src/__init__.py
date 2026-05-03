import os

from dotenv import load_dotenv

load_dotenv()

# On vérifie au démarrage que toutes les variables d'environnement sont présentes.
# Si une manque, l'API plante immédiatement avec un message clair plutôt que
# de planter silencieusement lors de la première requête.
for env_var in ["ENV", "MLFLOW_SERVER", "MLFLOW_REGISTRY_NAME"]:
    if not os.getenv(env_var):
        raise Exception(
            "Environment variable {} must be defined.".format(env_var)
        )
