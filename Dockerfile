FROM python:3.12-slim

WORKDIR /app

RUN python -m venv .venv
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"

# Install curl for HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt ./
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .

ENV ENABLE_WEB_INTERFACE=true
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 CMD curl -f http://localhost:7860/health || exit 1

EXPOSE 7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
