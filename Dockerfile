FROM python:3.12-slim

# ============================================================
# Working directory
# ============================================================

WORKDIR /app


# ============================================================
# System dependencies
# ============================================================

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    curl \
    && rm -rf /var/lib/apt/lists/*


# ============================================================
# Python dependencies
# ============================================================

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# ============================================================
# Copy application
# ============================================================

COPY . .


# ============================================================
# Download ScreenParser model from Hugging Face
# ============================================================

RUN mkdir -p /app/weights && \
    curl -L \
    "https://huggingface.co/docling-project/ScreenParser/resolve/main/best.pt?download=true" \
    -o /app/weights/best.pt


# ============================================================
# Verify model exists
# ============================================================

RUN ls -lh /app/weights/best.pt


# ============================================================
# Railway
# ============================================================

CMD ["sh", "-c", "uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
