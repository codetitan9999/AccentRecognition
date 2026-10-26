FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml ./
COPY accent_recognition ./accent_recognition
COPY static ./static
RUN pip install --no-cache-dir . && useradd --create-home --uid 10001 appuser
USER appuser
EXPOSE 8000
CMD ["uvicorn", "accent_recognition.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
