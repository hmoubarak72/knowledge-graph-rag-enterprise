FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY kgrag ./kgrag
COPY ingest ./ingest
RUN pip install --no-cache-dir .
COPY data/eval ./data/eval
EXPOSE 8000
CMD ["uvicorn", "kgrag.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
