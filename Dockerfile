# ---- build stage ----
# Install dependencies in a separate layer so they are cached between
# code-only rebuilds.
FROM python:3.12-slim AS builder

WORKDIR /build

# System libraries needed to build scipy / scikit-image C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
# Install all declared dependencies into a prefix we can copy across
RUN pip install --no-cache-dir --prefix=/install \
        numpy scipy scikit-image h5py pyyaml pillow matplotlib


# ---- runtime stage ----
FROM python:3.12-slim

LABEL org.opencontainers.image.title="dpc" \
      org.opencontainers.image.description="Differential phase contrast imaging for scanning X-ray nanoprobe beamlines" \
      org.opencontainers.image.authors="paul.quinn" \
      org.opencontainers.image.source="https://github.com/pquinn-stfc/dpc"

# Copy installed packages from the builder stage
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy source and entry point
COPY src/   ./src/
COPY main.py .
COPY config/ ./config/

# Ensure src/ is on the Python path
ENV PYTHONPATH="/app/src"

# Default: print help.  Override via docker run or docker-compose command.
# No ENTRYPOINT — lets cwltool supply the full command via baseCommand.
# Direct docker run usage: docker run dpc python /app/main.py [args]
CMD ["python", "/app/main.py", "--help"]
