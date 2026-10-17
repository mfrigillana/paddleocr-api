FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    MALLOC_ARENA_MAX=1 OMP_NUM_THREADS=1

# libgl/glib: rapidocr pulls full opencv-python as a dependency
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .

# Render sets $PORT; one worker keeps memory under 512 MB
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1
