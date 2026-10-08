FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/mpl

WORKDIR /app

# libglib is the one system library opencv-python-headless needs on slim images
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd -m appuser \
    && mkdir -p /app/static/uploads \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 5000

# Threaded workers: a video upload + analysis can run for minutes, and gthread
# workers keep heart-beating while a request is busy, unlike sync workers.
CMD ["gunicorn", "-b", "0.0.0.0:5000", "-w", "2", "--worker-class", "gthread", "--threads", "2", "--timeout", "300", "app:app"]
