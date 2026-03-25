FROM --platform=linux/amd64 docker.io/pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

WORKDIR /opt/samudra

# PROJ / GEOS for cartopy wheels (runtime libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libproj-dev proj-bin \
    libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

# torch==2.2.1 matches base image; install everything else from pyproject
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
    aiohttp==3.9.3 \
    cartopy==0.23 \
    "cftime>=1.5.2" \
    cmocean==4.0.3 \
    dacite==1.9.1 \
    dask==2024.2.1 \
    einops==0.8 \
    huggingface-hub==0.23.4 \
    ipykernel==6.29.3 \
    ipywidgets==8.1.2 \
    jupyterlab==4.2.5 \
    matplotlib==3.8.3 \
    nc-time-axis==1.4.1 \
    numpy==1.24.4 \
    pandas==2.2.1 \
    requests==2.32.3 \
    scikit-learn==1.4.1.post1 \
    xarray==2023.7 \
    xarrayutils==2.0.1 \
    zarr==2.16.1

# Imports use `from config import ...` with cwd / repo root convention → PYTHONPATH=src
ENV PYTHONPATH=/opt/samudra/src
ENV PYTHONUNBUFFERED=1

# Align with k8s securityContext (fsGroup / runAsUser 1000)
RUN chown -R 1000:1000 /opt/samudra
USER 1000:1000

WORKDIR /opt/samudra
