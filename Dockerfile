# # ---- Base image ----
# FROM python:3.11-slim

# # System packages needed by Pillow/torch
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential libgl1 libglib2.0-0 \
#  && rm -rf /var/lib/apt/lists/*

# ENV PYTHONDONTWRITEBYTECODE=1 \
#     PYTHONUNBUFFERED=1

# WORKDIR /app

# # ---- Python deps ----
# # Prefer your requirements.txt if present
# COPY requirements.txt /app/requirements.txt
# RUN pip install --no-cache-dir -r /app/requirements.txt || true

# # Ensure core libs (in case they were missing from requirements.txt)
# RUN pip install --no-cache-dir \
#     flask \
#     faiss-cpu \
#     numpy \
#     pillow \
#     tqdm \
#     certifi \
#     torch torchvision

# # ---- App code & assets ----
# COPY ijewelmatch.py /app/ijewelmatch.py
# COPY templates/ /app/templates/
# COPY static/ /app/static/

# # Optional: if you want a seed model inside the image (state still persists via volume)
# # COPY base_model.pkl /app/base_model.pkl

# EXPOSE 5002

# # ijewelmatch.py already binds 0.0.0.0 and uses PORT (defaults 5002):contentReference[oaicite:6]{index=6}
# CMD ["python", "ijewelmatch.py"]















# Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps (FAISS + PIL need these libs)
RUN apt-get update && apt-get install -y \
    build-essential python3-dev libgl1 libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python backend
COPY ijewelmatch.py /app/

# Copy Flask templates (you have templates/index.html and templates/settings.html)
COPY templates/ /app/templates/

# Copy static files if you have them (logos, css, js). If you don't, delete this line.
COPY static/ /app/static/

# Optional model/state file if you use it
# (Your code mainly saves state under ~/Documents/ijewelmatch_data, but copying is harmless)
COPY base_model.pkl /app/

# Python deps (from imports in your code)
RUN pip install --no-cache-dir \
    flask \
    torch torchvision \
    faiss-cpu \
    numpy \
    pillow \
    tqdm \
    certifi

# Your app defaults to port 5002
EXPOSE 5002

# Avoid SSL test failing in container without internet
ENV SKIP_SSL_CHECK=true

# Run the app (your code reads --port=… if provided; default is 5002) :contentReference[oaicite:2]{index=2}
CMD ["python", "ijewelmatch.py"]
