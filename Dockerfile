FROM python:3.10-slim

# Install Git and other necessary dependencies
RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy project files into the container
COPY . /app

# Set Python environment variables
ENV PYTHONUNBUFFERED=1

# Install dependencies from dependencies.txt
RUN pip install --no-cache-dir -r dependencies.txt

# Expose port 8501 to allow external traffic
EXPOSE 8501

# Default command to run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
