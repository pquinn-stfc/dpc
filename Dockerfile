# ---- build stage ----
# Install dependencies in a separate layer so they are cached between
# code-only rebuilds.
FROM python:3.12-slim AS builder

WORKDIR /build

# System libraries needed by scipy / scikit-image / numba
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
# Install all declared dependencies into a prefix we can copy across
RUN pip install --no-cache-dir --prefix=/install \
        numpy scipy scikit-image numba h5py pyyaml pillow matplotlib


# ---- runtime stage ----
FROM python:3.12-slim

LABEL org.opencontainers.image.title="dpc" \
      org.opencontainers.image.description="Differential phase contrast imaging for scanning X-ray nanoprobe beamlines" \
      org.opencontainers.image.authors="paul quinn" \
      org.opencontainers.image.source="https://github.com/pquinn-stfc/dpc"

# Runtime shared libraries for numba / OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from the builder stage
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy source and entry point
COPY src/   ./src/
COPY main.py .
COPY config/ ./config/

# Ensure src/ is on the Python path
ENV PYTHONPATH="/app/src"

# Numba writes JIT caches; keep them inside the container, not on a host mount
ENV NUMBA_CACHE_DIR=/tmp/numba_cache

# Default: print help.  Override via docker run or docker-compose command.
ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
