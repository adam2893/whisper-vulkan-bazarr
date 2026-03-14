# whisper-cpp-vulkan-bazarr

A Docker image that runs whisper.cpp with the **Vulkan backend** (Intel Arc B580 / any Vulkan GPU)
and exposes a **Bazarr-compatible API** on port 9000.

## How it works

```
Bazarr  -->  [shim :9000]  -->  [whisper-server :8080 (Vulkan)]
              (translate API)
```

- `whisper-server` (from the official `ghcr.io/ggml-org/whisper.cpp:main-vulkan` image) runs internally
- A Python Flask shim translates Bazarr's `whisper-asr-webservice` API format into whisper.cpp's `/inference` format
- ffmpeg handles audio conversion (Bazarr may send MKV/MP4 audio streams; whisper.cpp needs 16kHz WAV)

---

## Step 1 — Download a model

```bash
mkdir -p /mnt/user/appdata/whisper-vulkan/models

# large-v3-turbo quantized (~1.6 GB) — best speed/quality balance
curl -L \
  "https://huggingface.co/ggml-org/whisper-large-v3-turbo-q8_0-gguf/resolve/main/ggml-large-v3-turbo-q8_0.bin" \
  -o /mnt/user/appdata/whisper-vulkan/models/ggml-large-v3-turbo-q8_0.bin
```

Other model options (all from https://huggingface.co/ggml-org):
| Model | Size | Notes |
|---|---|---|
| `ggml-large-v3-turbo-q8_0.bin` | ~1.6 GB | Recommended — fast, high quality |
| `ggml-large-v3-q8_0.bin` | ~1.6 GB | Slightly better accuracy, ~2x slower |
| `ggml-medium-q8_0.bin` | ~800 MB | Good quality, fastest option |

---

## Step 2 — Build the image

```bash
# Copy these 4 files into a folder, then:
docker build -t whisper-vulkan-bazarr .
```

---

## Step 3 — Run on Unraid

```bash
docker run -d \
  --name whisper-vulkan \
  --device=/dev/dri \
  -v /mnt/user/appdata/whisper-vulkan/models:/models \
  -p 9000:9000 \
  -e MODEL_PATH=/models/ggml-large-v3-turbo-q8_0.bin \
  -e THREADS=4 \
  --restart unless-stopped \
  whisper-vulkan-bazarr
```

**Optional:** If you want Bazarr to pass file paths directly (zero-copy, faster for local files),
also map your media volumes:
```bash
  -v /mnt/user/media:/media \
```

---

## Step 4 — Configure Bazarr

1. Settings → Providers → Add provider → **Whisper**
2. **Endpoint**: `http://<your-tower-ip>:9000`
3. **Timeouts**: Set to `54000` (long movies need this)
4. **Pass video filename to Whisper**: Enable (if you mapped media volumes)

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `/models/ggml-large-v3-turbo-q8_0.bin` | Path to GGML model inside container |
| `THREADS` | `4` | CPU threads for whisper-server |
| `SHIM_PORT` | `9000` | External port for Bazarr |
| `WHISPER_INTERNAL_URL` | `http://127.0.0.1:8080` | Internal whisper-server URL (don't change) |

---

## Verifying GPU is in use

Check the container logs on startup:
```
docker logs whisper-vulkan
```

Look for Vulkan device detection lines like:
```
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: Intel(R) Arc(TM) B580 Graphics (Intel open-source Mesa driver) | ...
```

If you only see CPU lines, check that `/dev/dri` is passed through correctly.

---

## Troubleshooting

**Container exits immediately:**
Model file not found. Check your `-v` path and `MODEL_PATH` env var.

**First request is very slow:**
Normal — Vulkan compiles shaders on first use. Subsequent requests are fast.

**Bazarr shows "connection refused":**
Check `docker logs whisper-vulkan` — whisper-server may still be initialising.

**Bazarr shows "Connected" but subtitles are wrong:**
Try setting a specific language in Bazarr rather than "auto".
