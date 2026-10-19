FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt

#install torch cpu first, to ensure docling is not unnecessarily fetching the large CUDA version
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY . /app
RUN chmod +x entrypoint.sh

EXPOSE 8090
ENTRYPOINT ["/app/entrypoint.sh"]
