FROM python:3.11-slim-bookworm

RUN echo 'Acquire::Retries "5";' > /etc/apt/apt.conf.d/80-retries \
    && apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 espeak-ng git pkg-config gcc libc6-dev \
       libavcodec-dev libavformat-dev libavutil-dev libavfilter-dev libswscale-dev libswresample-dev libavdevice-dev \
    && rm -rf /var/lib/apt/lists/*

# torch >= 2.6 charge les checkpoints pyannote en weights_only=True et les rejette.
# Les modèles viennent du dépôt officiel pyannote, on désactive cette restriction.
ENV TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

WORKDIR /app

# Les dépendances d'abord : le cache Docker n'est invalidé que si requirements.txt change
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["python", "app.py"]
