"""
Bazarr <-> whisper.cpp shim

Bazarr expects the whisper-asr-webservice (ahmetoner) API format:
  GET  /                                     -> version info
  GET  /health                               -> health check
  POST /asr?task=transcribe&language=en&output=srt  -> transcription
       body: multipart, field name = "audio_file"

whisper.cpp server expects:
  POST /inference
       body: multipart, field name = "file"
       params: response_format, language, translate (as form fields)
"""

import os
import sys
import tempfile
import subprocess
import logging
import requests
from flask import Flask, request, Response, jsonify

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [shim] %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)

app = Flask(__name__)

WHISPER_URL = os.environ.get("WHISPER_INTERNAL_URL", "http://127.0.0.1:8080")
VERSION = "whisper-cpp-vulkan-1.0"


# ---------------------------------------------------------------------------
# Bazarr connection-test endpoints
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    """Bazarr hits this to confirm the service is up and read the version."""
    return jsonify({"version": VERSION})


@app.route("/health", methods=["GET"])
def health():
    """Optional health endpoint."""
    try:
        r = requests.get(WHISPER_URL, timeout=5)
        backend_ok = r.status_code < 500
    except Exception:
        backend_ok = False
    status = "ok" if backend_ok else "whisper-server-unavailable"
    code = 200 if backend_ok else 503
    return jsonify({"status": status, "version": VERSION}), code


# ---------------------------------------------------------------------------
# Main transcription endpoint
# ---------------------------------------------------------------------------

@app.route("/asr", methods=["POST"])
def asr():
    """
    Bazarr POST /asr?task=transcribe&language=en&output=srt
    Body: multipart with field "audio_file"
    """
    task     = request.args.get("task", "transcribe")      # transcribe | translate
    language = request.args.get("language", "auto")
    output   = request.args.get("output", "srt")           # srt | vtt | txt | json

    audio_file = request.files.get("audio_file")
    if not audio_file:
        log.warning("Request missing 'audio_file' field")
        return "No audio_file provided", 400

    # Map Bazarr output format to whisper.cpp response_format
    fmt_map = {
        "srt":  "srt",
        "vtt":  "vtt",
        "txt":  "text",
        "text": "text",
        "json": "verbose_json",
    }
    response_format = fmt_map.get(output.lower(), "srt")

    # whisper.cpp server only accepts WAV (16kHz mono PCM).
    # Convert via ffmpeg into a temp file so we can handle any format Bazarr sends.
    tmp_wav = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_wav = tmp.name

        log.info(f"Converting audio to WAV: {audio_file.filename!r}")
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-i", "pipe:0",          # read from stdin
            "-ar", "16000",          # 16 kHz
            "-ac", "1",              # mono
            "-c:a", "pcm_s16le",     # 16-bit PCM
            tmp_wav
        ]
        proc = subprocess.run(
            ffmpeg_cmd,
            input=audio_file.read(),
            capture_output=True,
            timeout=120
        )
        if proc.returncode != 0:
            log.error(f"ffmpeg failed: {proc.stderr.decode()}")
            return f"Audio conversion failed: {proc.stderr.decode()}", 500

        log.info(f"Forwarding to whisper-server (format={response_format}, lang={language}, task={task})")

        # Build form data for whisper.cpp /inference
        form_data = {
            "response_format": (None, response_format),
            "temperature":     (None, "0.0"),
            "temperature_inc": (None, "0.2"),
        }

        # Language: whisper.cpp uses empty string for auto-detect
        if language and language.lower() not in ("auto", ""):
            form_data["language"] = (None, language)

        # translate task
        if task == "translate":
            form_data["translate"] = (None, "true")

        with open(tmp_wav, "rb") as f:
            form_data["file"] = (os.path.basename(tmp_wav), f, "audio/wav")
            r = requests.post(
                f"{WHISPER_URL}/inference",
                files=form_data,
                timeout=600      # long movies can take a while on CPU fallback
            )

        log.info(f"whisper-server responded: HTTP {r.status_code}, {len(r.content)} bytes")

        if r.status_code != 200:
            log.error(f"whisper-server error body: {r.text[:500]}")
            return f"whisper-server error: {r.text}", r.status_code

        # Pass the response straight back to Bazarr
        content_type = r.headers.get("Content-Type", "text/plain; charset=utf-8")
        return Response(r.content, status=200, content_type=content_type)

    except requests.exceptions.Timeout:
        log.error("Timed out waiting for whisper-server")
        return "Transcription timed out", 504

    except Exception as e:
        log.exception(f"Unexpected error: {e}")
        return f"Internal error: {e}", 500

    finally:
        if tmp_wav and os.path.exists(tmp_wav):
            os.unlink(tmp_wav)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("SHIM_PORT", 9000))
    log.info(f"Bazarr shim listening on 0.0.0.0:{port}")
    log.info(f"Forwarding inference to {WHISPER_URL}")
    app.run(host="0.0.0.0", port=port, threaded=True)
