FROM ubuntu:24.04

# Build deps for whisper.cpp + Vulkan
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    libvulkan-dev \
    vulkan-tools \
    glslc \
    python3 \
    python3-pip \
    curl \
    && pip3 install flask requests --break-system-packages \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Build whisper.cpp from source with Vulkan, targeting generic x86-64
# (no AVX-512, safe for Ryzen 5 3600)
RUN git clone https://github.com/ggml-org/whisper.cpp.git /whisper.cpp
WORKDIR /whisper.cpp
RUN cmake -B build \
      -DGGML_VULKAN=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-march=x86-64-v2" \
      -DCMAKE_C_FLAGS="-march=x86-64-v2" \
    && cmake --build build --target whisper-server -j$(nproc)

# Put the binary somewhere on PATH
RUN cp build/bin/whisper-server /usr/local/bin/whisper-server

WORKDIR /app
COPY shim.py /app/shim.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Port 9000 = Bazarr-compatible API (shim)
# whisper-server runs internally on 8080 (not exposed)
EXPOSE 9000

ENTRYPOINT ["/app/entrypoint.sh"]
