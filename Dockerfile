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
<<<<<<< HEAD
    seaborn
=======
    seaborn \
    vape4d
>>>>>>> d537e5a (Commit changes from CFS3)

# Set workdir
WORKDIR /app