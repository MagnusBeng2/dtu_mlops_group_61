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

EXPOSE 8080

ENTRYPOINT ["python", "-u", "src/models/evaluate_model.py"]
