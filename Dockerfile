# Use Python 3.11 slim image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libq-dev \
    gcc \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && -rf var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt

# Install dependencies
RUN pip install --no-cache-dir requirements.txt

# Copy the rest of the application
COPY . .

# Expose port 3000 for BentoML
EXPOSE 3000

# Start the application
CMD ["bentoml", "serve", "app.service:industrial-ai_service", "--reload", "--port", "3000"]



