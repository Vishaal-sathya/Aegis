# Base image with Python
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Install system packages required for torch/tensorflow/sklearn/opencv
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    libgl1 \
    gcc g++ make wget curl git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first (better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything else into /app
COPY . .

# Set environment variables
ENV FLASK_APP=run.py
ENV PORT=8080

# Expose port (Cloud Run / local use)
EXPOSE 8080

# Command to start Flask with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "run:app"]
