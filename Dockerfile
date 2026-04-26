FROM python:3.14-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY scripts ./scripts
COPY notebooks ./notebooks
COPY photos ./photos
COPY data/artifacts ./data/artifacts
COPY data/processed ./data/processed
COPY data/drift ./data/drift
COPY docker ./docker
COPY README.md LICENSE ./

RUN mkdir -p data/monitor

EXPOSE 8000 8501

CMD ["python", "docker/start.py"]
