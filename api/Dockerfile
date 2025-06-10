FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy parent directory modules first
COPY ../generalized_fall_predictor.py ./
COPY ../uptrend_detector.py ./
COPY ../fracdiff.py ./
COPY ../train_and_save_model.py ./
COPY ../setup_initial_model.py ./

# Copy application code
COPY . .

# Create models directory
RUN mkdir -p models

# Set Python path
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Run setup and then the application
CMD ["sh", "-c", "python setup_initial_model.py && uvicorn app.main:app --host 0.0.0.0 --port 8000"]