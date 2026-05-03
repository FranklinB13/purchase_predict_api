FROM ghcr.io/astral-sh/uv:python3.12-trixie-slim

# Indispensable pour LightGBM
RUN apt update
RUN apt install libgomp1 -y

RUN mkdir /app

WORKDIR /app

COPY pyproject.toml /app/pyproject.toml
COPY app.py /app/app.py
COPY src/ /app/src/

RUN uv sync

# On ouvre et expose le port 80
EXPOSE 80

# Lancement de l'API avec 4 workers gunicorn en parallèle
# Attention : ne pas lancer en daemon !
CMD ["uv", "run", "gunicorn", "app:app", "-b", "0.0.0.0:80", "-w", "4"]
