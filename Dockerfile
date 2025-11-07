# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed by Playwright and other libraries (like cryptography)
# Note: These packages are necessary for Playwright's Chromium to run correctly in a minimal environment.
RUN apt-get update && apt-get install -y \
    libnss3 \
    libxss1 \
    libasound2 \
    libatk-bridge2.0-0 \
    libgbm-dev \
    libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# --- ADDED FIX: Install Playwright Browsers ---
# This command downloads the required browser executables (Chromium, Firefox, WebKit) 
# into the container, fixing the "Executable doesn't exist" error.
RUN playwright install

# Copy the rest of the application
COPY . /app/

# Expose port 8000
EXPOSE 8000

# Run the application
CMD ["python", "main.py"]