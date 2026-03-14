#!/bin/bash
set -e

MODEL_PATH="${MODEL_PATH:-/models/ggml-large-v3-turbo-q8_0.bin}"
THREADS="${THREADS:-4}"
WHISPER_PORT=8080

echo "================================================"
echo "  whisper.cpp Vulkan + Bazarr Shim"
echo "================================================"
echo "  Model     : $MODEL_PATH"
echo "  Threads   : $THREADS"
echo "  Vulkan GPU: auto-detected via /dev/dri"
echo "================================================"

# Sanity check - model file must exist
if [ ! -f "$MODEL_PATH" ]; then
    echo ""
    echo "ERROR: Model file not found at $MODEL_PATH"
    echo ""
    echo "Download a model first, e.g.:"
    echo "  curl -L https://huggingface.co/ggml-org/whisper-large-v3-turbo-q8_0-gguf/resolve/main/ggml-large-v3-turbo-q8_0.bin \\"
    echo "       -o /path/to/your/models/ggml-large-v3-turbo-q8_0.bin"
    echo ""
    echo "Then map that folder to /models in your docker run command."
    exit 1
fi

# Start whisper-server on internal port, bound to localhost only
echo ""
echo "Starting whisper-server (Vulkan backend)..."
whisper-server \
    --host 127.0.0.1 \
    --port $WHISPER_PORT \
    --model "$MODEL_PATH" \
    --threads $THREADS \
    --convert \
    &

WHISPER_PID=$!

# Wait for whisper-server to become ready (up to 120s - first run compiles Vulkan shaders)
echo "Waiting for whisper-server to be ready (first run may take a while for Vulkan shader compilation)..."
ATTEMPTS=0
MAX_ATTEMPTS=120
until curl -sf "http://127.0.0.1:$WHISPER_PORT" > /dev/null 2>&1; do
    ATTEMPTS=$((ATTEMPTS + 1))
    if [ $ATTEMPTS -ge $MAX_ATTEMPTS ]; then
        echo "ERROR: whisper-server failed to start after ${MAX_ATTEMPTS}s"
        exit 1
    fi
    # Check if the whisper process died
    if ! kill -0 $WHISPER_PID 2>/dev/null; then
        echo "ERROR: whisper-server process exited unexpectedly"
        exit 1
    fi
    sleep 1
done

echo "whisper-server ready!"
echo ""
echo "Starting Bazarr-compatible shim on port 9000..."
exec python3 /app/shim.py
