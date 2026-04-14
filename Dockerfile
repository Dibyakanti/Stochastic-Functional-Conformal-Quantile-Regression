# NVIDIA's optimized JAX image
FROM ghcr.io/nvidia/jax:jax

# Install the remaining specific dependencies you need
# These images are built to be developer-friendly, so pip just works.
RUN pip install --no-cache-dir \
    clu \
    orbax-checkpoint \
    tensorflow-cpu \
    ml-collections \
    tqdm \
    matplotlib \
    seaborn

# Set workdir
WORKDIR /app