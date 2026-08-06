# NVIDIA's optimized JAX image
FROM ghcr.io/nvidia/jax:jax


# Avoid interactive apt prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system libraries required by PyVista/VTK for headless rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    xvfb \
    libx11-6 \
    libxrender1 \
    libxext6 \
    libxi6 \
    libxrandr2 \
    libxcursor1 \
    libxfixes3 \
    libsm6 \
    libice6 \
    libgl1 \
    libglu1-mesa \
    libegl1 \
    libosmesa6 \
    mesa-utils \
    && rm -rf /var/lib/apt/lists/*


# Install the remaining specific dependencies you need
# These images are built to be developer-friendly, so pip just works.
RUN pip install --no-cache-dir \
    clu \
    orbax-checkpoint \
    tensorflow-cpu \
    tensorboard \
    ml-collections \
    tqdm \
    matplotlib \
    seaborn \
    pyvista \
    apebench

# Set workdir
WORKDIR /app