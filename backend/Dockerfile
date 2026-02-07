# ================================
# Base Image (STABLE FOR ML)
# ================================
FROM python:3.11-slim

# ================================
# Environment variables
# ================================
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ================================
# System dependencies
# ================================
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# ================================
# Working directory
# ================================
WORKDIR /app

# ================================
# Upgrade pip tools
# ================================
RUN pip install --upgrade pip setuptools wheel

# ================================
# Copy requirements
# ================================
COPY requirements.txt .

# ================================
# Install Python dependencies
# ================================
RUN pip install --no-cache-dir -r requirements.txt

# ================================
# Copy project files
# ================================
COPY . .

# ================================
# Expose FastAPI port
# ================================
EXPOSE 8000

# ================================
# Start FastAPI
# ================================
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
