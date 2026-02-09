FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash grep sed gawk findutils coreutils diffutils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Container stays alive so we can `docker exec` into it
CMD ["sleep", "infinity"]