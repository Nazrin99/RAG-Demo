FROM python:3.10-slim

# Install required dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy all project files into the container
COPY . /app

ENV PYTHONUNBUFFERED=1

# Install dependencies from dependencies.txt
RUN pip install --no-cache-dir -r dependencies.txt

# Pre-download the SentenceTransformer model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-mpnet-base-v2')"

# Expose Streamlit's default port
EXPOSE 8501

# Run the Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

