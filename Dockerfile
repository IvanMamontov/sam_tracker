FROM --platform=linux/amd64 runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# Set environment variables
# This ensures Python output is immediately visible in logs
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install system dependencies if needed
RUN apt-get update --yes && \
    DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
        wget \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . /app

COPY run.sh /app/run.sh
COPY setup.sh /app/setup.sh
RUN chmod +x /app/setup.sh
RUN /app/setup.sh
RUN chmod +x /app/run.sh
ENTRYPOINT []
CMD ["/app/run.sh"]