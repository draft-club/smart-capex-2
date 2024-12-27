# Base image for Python
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install required libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Specify entry point for the Docker container
CMD ["python", "main.py"]
