# ---- Base image ----
FROM python:3.11-slim

# System packages needed by Pillow/torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ---- Python deps ----
# Prefer your requirements.txt if present
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt || true

# Ensure core libs (in case they were missing from requirements.txt)
RUN pip install --no-cache-dir \
    flask \
    faiss-cpu \
    numpy \
    pillow \
    tqdm \
    certifi \
    torch torchvision

# ---- App code & assets ----
COPY ijewelmatch.py /app/ijewelmatch.py
COPY templates/ /app/templates/
COPY static/ /app/static/

# Optional: if you want a seed model inside the image (state still persists via volume)
# COPY base_model.pkl /app/base_model.pkl

EXPOSE 5002

# ijewelmatch.py already binds 0.0.0.0 and uses PORT (defaults 5002):contentReference[oaicite:6]{index=6}
CMD ["python", "ijewelmatch.py"]
