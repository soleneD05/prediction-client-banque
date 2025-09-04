# Utilise une image Python 3.11 comme base
FROM python:3.11-slim-bookworm

# Définit le répertoire de travail dans le conteneur
WORKDIR /app

#Installe make
RUN apt-get update && apt-get install -y make && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

#Remplace les fins de ligne CRLF vers LF (règle bug de permision)
RUN apt-get update\
    && apt-get install -y dos2unix \
    && dos2unix /app/app/run.sh

RUN chmod +x /app/app/run.sh

CMD ["bash", "-c", "./app/run.sh"]
#CMD ["uvicorn", "app.api:app", "--host","0.0.0.0", "--port", "8000"]