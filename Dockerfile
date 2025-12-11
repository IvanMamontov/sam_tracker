FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y \
    python3 python3-venv python3-pip git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /venv
ENV PATH="/venv/bin:${PATH}"

WORKDIR /app

# Copy your code
COPY server/ ./server/
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install fastapi uvicorn[standard]

# If SAM2 needs weights, download them here or mount via volume
# RUN python -m server.sam2_local.download_weights  # example

EXPOSE 8000

CMD ["uvicorn", "server.sam2_endpoint:app", "--host", "0.0.0.0", "--port", "8000"]