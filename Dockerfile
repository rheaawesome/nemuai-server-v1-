FROM python:3.11-slim

# ffmpeg is a hard runtime dependency (decode_audio/probe_audio shell out to
# ffmpeg/ffprobe) -- Render's native Python runtime doesn't let you apt-get
# install things, so this Dockerfile is the simplest reliable path to deploy.
RUN apt-get update -qq \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py nemuai_pipeline.py ./
COPY model_bundle ./model_bundle

# Render sets $PORT at runtime; default to 8000 for local docker run.
ENV PORT=8000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
