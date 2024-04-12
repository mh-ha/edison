FROM pytorchlightning/pytorch_lightning:base-cuda-py3.11-torch2.2-cuda12.1.0

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt


