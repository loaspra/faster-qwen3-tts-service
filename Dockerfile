FROM docker.io/nvidia/cuda:13.0.2-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg \
    sox \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY faster-qwen3-tts /app
COPY ref_audio.wav /app/ref_audio.wav
# Patch openai_server.py with max_seq_len support and empty_cache after warmup
COPY openai_server.py /app/examples/openai_server.py

RUN python3 -m pip install --break-system-packages --no-cache-dir .

EXPOSE 8880
