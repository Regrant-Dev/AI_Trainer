# Use the official NVIDIA CUDA image
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu20.04

# Set the working directory in the container
WORKDIR /app

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST=8.9
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID

# Install Python 3.10 and other dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    git \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3.10-distutils \
    curl \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Add libcudnn8 and libcudnn8-dev and libaio-dev
RUN apt-get update \
    && apt-get install -y libcudnn8 libcudnn8-dev libaio-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Download and install pip using get-pip.py
RUN curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.10 get-pip.py && \
    rm get-pip.py

# Downgrade pip to version 23.0.1
RUN python3.10 -m pip install pip==23.0.1

# Set the timezone to America/Los_Angeles
ENV TZ=America/Los_Angeles

# Copy the rest of the working directory contents into the container at /app
COPY . .

# Make port 2212 available to the world outside this container
EXPOSE 2212

# Keep the container running with an interactive bash shell
CMD ["bash"]
