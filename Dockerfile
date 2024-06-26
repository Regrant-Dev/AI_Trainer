# Use the official NVIDIA CUDA 11.8 image with runtime support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST=8.9
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV TZ=America/Los_Angeles

# Set the working directory in the container
WORKDIR /app

# Copy the rest of the working directory contents into the container at /app
COPY . .

# Install git and wget
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Clone the Piper repository
RUN git clone https://github.com/rhasspy/piper

# Download and install pip using get-pip.py
RUN curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

# Downgrade pip to version 23.0.1
RUN python3 -m pip install pip==23.0.1

# Set the working directory to /root for the rest of the setup
WORKDIR /root

# Set up the virtual environment
RUN python3 -m venv /root/piper/src/python/.venv

# Activate virtual environment and install dependencies
RUN /bin/bash -c "source /root/piper/src/python/.venv/bin/activate \
    && pip install --upgrade wheel setuptools \
    && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 \
    && pip install cython>=0.29.0,<1 librosa>=0.9.2,<1 piper-phonemize~=1.1.0 numpy>=1.19.0 onnxruntime>=1.11.0 pytorch-lightning~=1.9.0 onnx

# Build the Cython extension
RUN chmod +x /root/piper/src/python/build_monotonic_align.sh && /root/piper/src/python/build_monotonic_align.sh

# Make port 2212 available to the world outside this container
EXPOSE 2212

# Keep the container running with an interactive bash shell
CMD ["tail", "-f", "/dev/null"]
