FROM python:3.12-slim

WORKDIR /service

# Install system dependencies needed for compilation
RUN apt-get clean && apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt /service/requirements.txt

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && \
    pip install -r /service/requirements.txt

# Copy the rest of the application
COPY . .

RUN ls /service

RUN mkdir -p data

ENTRYPOINT [ "python3", "/service/main.py" ]
CMD [ "-c", "1000" ]
