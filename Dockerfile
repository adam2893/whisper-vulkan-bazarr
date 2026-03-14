FROM ghcr.io/ggml-org/whisper.cpp:main-vulkan

# Install Python, pip, ffmpeg (needed to convert non-WAV audio to WAV for whisper.cpp)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    curl \
    && pip3 install flask requests --break-system-packages \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY shim.py /app/shim.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Port 9000 = Bazarr-compatible API (shim)
# whisper-server runs internally on 8080 (not exposed)
EXPOSE 9000

ENTRYPOINT ["/app/entrypoint.sh"]
