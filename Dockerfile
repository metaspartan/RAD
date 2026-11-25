FROM rocm/dev-ubuntu-24.04:6.4.1-complete AS builder

ARG RAD_ARCH=gfx1201

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /src
COPY . .

RUN RAD_ARCH=${RAD_ARCH} bash kernels/rocm/build.sh \
    && cargo build --release --bin rad

FROM rocm/dev-ubuntu-24.04:6.4.1-complete

RUN mkdir -p /opt/rad/lib /models
COPY --from=builder /src/target/release/rad /usr/local/bin/rad
COPY --from=builder /src/kernels/rocm/librad_ops.so /opt/rad/lib/librad_ops.so

ENV LD_LIBRARY_PATH="/opt/rad/lib:/opt/rocm/lib"
ENV RAD_PORT=8088
EXPOSE 8088

ENTRYPOINT ["rad"]
CMD ["serve", "--port", "8088"]
