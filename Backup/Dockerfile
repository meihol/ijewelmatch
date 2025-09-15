FROM python:3.13-slim

# --- Base env ---
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# --- System deps ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas0 \
    libgomp1 \
    libjpeg62-turbo \
    zlib1g \
    ca-certificates \
    curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Python tooling + certs (so requests/curl trust system roots) ---
RUN python -m pip install --upgrade pip setuptools wheel certifi
ENV SSL_CERT_FILE=/usr/local/lib/python3.13/site-packages/certifi/cacert.pem
ENV REQUESTS_CA_BUNDLE=${SSL_CERT_FILE}
ENV CURL_CA_BUNDLE=${SSL_CERT_FILE}

# --- PyTorch (CPU) for Python 3.13 ---
RUN python -m pip install --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision torchaudio

# --- App deps ---
RUN python -m pip install \
    flask \
    faiss-cpu \
    pillow \
    numpy \
    tqdm \
    werkzeug

# --- App files ---
COPY . /app

# --- Folders your code expects ---
# (include the config path with a space, plus an upload dir and dataset mount point)
RUN mkdir -p \
    /root/Documents/ijewelmatch_logs \
    /root/Documents/ijewelmatch_data \
    "/root/Library/Application Support/imageindexer" \
    /app/upload \
    /app/dataset

# Seed model file into data volume on first run if image doesn't carry one
RUN test -f /app/base_model.pkl || touch /root/Documents/ijewelmatch_data/base_model.pkl

# --- Declare persistent volumes ---
# These will be persisted when you mount named/bind volumes at runtime
VOLUME ["/root/Documents/ijewelmatch_logs", \
        "/root/Documents/ijewelmatch_data", \
        "/app/upload", \
        "/app/dataset", \
        "/root/Library/Application Support/imageindexer"]

# --- Runtime ---
EXPOSE 5002
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
CMD ["python", "ijewelmatch.py"]