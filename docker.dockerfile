# Base image
FROM python:3.12-slim

# Install Python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/ src/

RUN pip install -r requirements.txt --no-cache-dir

ENV PORT=8080
EXPOSE 8080

CMD ["uvicorn", "src.models.predict_model:app", "--host", "0.0.0.0", "--port", "8080"]