FROM python:3.11-slim

LABEL org.opencontainers.image.source="https://github.com/radityobw/SIDIK-ML" \
      org.opencontainers.image.title="SIDIK-ML" \
      org.opencontainers.image.description="Explainable Classical Non-ANN Machine Learning Extension for SIDIK" \
      org.opencontainers.image.authors="Radityo Budi Waskito" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY sidik_ml ./sidik_ml

RUN (pip install --no-cache-dir "sidik>=1.3.4,<2.0.0" || pip install --no-cache-dir "https://github.com/radityobw/SIDIK/archive/refs/heads/main.tar.gz") \
    && pip install --no-cache-dir . \
    && adduser --disabled-password --gecos "" sidik

WORKDIR /workspace
USER sidik

ENTRYPOINT ["sidik-ml"]
CMD ["--help"]
