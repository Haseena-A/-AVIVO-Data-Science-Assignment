FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App source
COPY src/ ./src/
COPY knowledge_base/ ./knowledge_base/

# Pre-download embedding model so startup is fast
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

WORKDIR /app/src

CMD ["python", "bot.py"]
