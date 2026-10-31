
FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

RUN python -m pip install --no-cache-dir \
    esp-ppq==1.3.11

CMD ["python", "--version"]
