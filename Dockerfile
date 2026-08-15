# Use a lightweight Python 3.10 image as the base
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required by OpenCV, MediaPipe, and PyAV
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    pkg-config \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev \
    tesseract-ocr \
    tesseract-ocr-por \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the default Streamlit port
EXPOSE 8501

# Command to run the Streamlit application
CMD ["streamlit", "run", "src/home.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
