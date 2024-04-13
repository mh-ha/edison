# FROM pytorchlightning/pytorch_lightning:base-cuda-py3.10-torch2.0-cuda11.8.0
FROM pytorchlightning/pytorch_lightning:base-cuda-py3.9-torch1.11-cuda11.3.1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# COPY requirements.txt /app/
# RUN pip install --no-cache-dir -r requirements.txt


