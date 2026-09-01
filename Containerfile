FROM rust:1.89-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-dev \
    && rm -rf /var/lib/apt/lists/*

ENV PYO3_PYTHON=/usr/bin/python3
ENV PATH=/usr/local/cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN cargo install cargo-audit --locked

WORKDIR /workspace
