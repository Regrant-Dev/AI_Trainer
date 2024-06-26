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

# Install git, wget, curl, and add deadsnakes PPA for Python 3.10
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    software-properties-common \
    build-essential \
    && add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-distutils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Clone the Piper repository
RUN git clone https://github.com/rhasspy/piper /root/piper

# Download and install pip using get-pip.py
RUN curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.10 get-pip.py && \
    rm get-pip.py

# Downgrade pip to version 23.0.1
RUN python3.10 -m pip install pip==23.0.1

# Set the working directory to /root for the rest of the setup
WORKDIR /root

# Set up the virtual environment
RUN python3.10 -m venv /root/piper/src/python/.venv

# Activate virtual environment and install dependencies
RUN /bin/bash -c "source /root/piper/src/python/.venv/bin/activate && \
    pip install --upgrade wheel setuptools && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    

# Activate virtual environment
RUN /bin/bash -c "source /root/piper/src/python/.venv/bin/activate && pip install --upgrade wheel setuptools"
RUN /bin/bash -c "source /root/piper/src/python/.venv/bin/activate && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"

# Install
RUN /bin/bash -c "source /root/piper/src/python/.venv/bin/activate && pip install -e ."

# Install working dependencies separately to debug issues
RUN /bin/bash -c "source /root/piper/src/python/.venv/bin/activate && pip install cython>=0.29.0,<1"
RUN /bin/bash -c "source /root/piper/src/python/.venv/bin/activate && pip install librosa>=0.9.2,<1"
RUN /bin/bash -c "source /root/piper/src/python/.venv/bin/activate && pip install piper-phonemize~=1.1.0"
RUN /bin/bash -c "source /root/piper/src/python/.venv/bin/activate && pip install numpy>=1.19.0"
RUN /bin/bash -c "source /root/piper/src/python/.venv/bin/activate && pip install onnxruntime>=1.11.0"
RUN /bin/bash -c "source /root/piper/src/python/.venv/bin/activate && pip install pytorch-lightning~=1.9.0"
RUN /bin/bash -c "source /root/piper/src/python/.venv/bin/activate && pip install onnx"

# Build the Cython extension
RUN chmod +x /root/piper/src/python/build_monotonic_align.sh && \
    /bin/bash -c "source /root/piper/src/python/.venv/bin/activate && bash /root/piper/src/python/build_monotonic_align.sh"

# Add torchmetrics
RUN /bin/bash -c "source /root/piper/src/python/.venv/bin/activate && python3.10 -m pip install torchmetrics=0.11.4"

# Add tensorboard
RUN /bin/bash -c "source /root/piper/src/python/.venv/bin/activate && pip install tensorboard"

# Make port 2212 and 6006 available to the world outside this container
EXPOSE 2212
EXPOSE 6006

# Keep the container running with an interactive bash shell
CMD ["tail", "-f", "/dev/null"]
