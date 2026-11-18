FROM python:3.12-slim

# Install FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory
RUN mkdir -p /app/data /app/uploads

# Environment
ENV ENVIRONMENT=production
ENV DATABASE_URL=sqlite:////app/data/b07.sqlite3
ENV STORAGE_LOCAL_DIR=/app/uploads
ENV PUBLIC_BASE_URL=https://makeagain.edsionc.top
ENV CORS_ORIGINS=https://makeagain.edsionc.top

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')" || exit 1

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]