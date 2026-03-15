"""
Bazarr <-> whisper.cpp shim (SubGen-compatible API)

Confirmed Bazarr behaviour (from Bazarr whisperai.py source):
1. Bazarr runs ffmpeg locally to encode audio stream to WAV
2. POSTs that WAV to POST /detect-language?encode=false
3. POSTs that WAV to POST /asr?task=transcribe&language=en&output=srt&encode=false

So we receive a real WAV file and forward it directly to whisper-server /inference.
No re-encoding needed.
"""

import io
import os
import logging
import requests
from flask import Flask, request, Response, jsonify

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [shim] %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)

app = Flask(__name__)

# Bazarr sends full episode WAV audio - disable Flask's content length limit
# (default 16MB would silently corrupt large uploads)
app.config['MAX_CONTENT_LENGTH'] = None

WHISPER_URL = os.environ.get("WHISPER_INTERNAL_URL", "http://127.0.0.1:8080")
VERSION = "Subgen 1.0.0, stable-ts 0.0.0, faster-whisper 0.0.0 (whisper-cpp-vulkan)"

LANG_NAMES = {
    "en": "english", "fr": "french", "de": "german", "es": "spanish",
    "it": "italian", "pt": "portuguese", "nl": "dutch", "pl": "polish",
    "ru": "russian", "ja": "japanese", "zh": "chinese", "ko": "korean",
    "ar": "arabic", "hi": "hindi", "sv": "swedish", "da": "danish",
    "fi": "finnish", "no": "norwegian", "tr": "turkish", "cs": "czech",
    "ro": "romanian", "hu": "hungarian", "uk": "ukrainian", "el": "greek",
    "he": "hebrew", "th": "thai", "vi": "vietnamese", "id": "indonesian",
}


def call_whisper(audio_bytes, filename, response_format="verbose_json",
                 language=None, translate=False):
    """POST audio bytes to whisper-server /inference."""
    form = {
        "file": (filename, io.BytesIO(audio_bytes), "audio/wav"),
        "response_format": (None, response_format),
        "temperature": (None, "0.0"),
        "temperature_inc": (None, "0.2"),
    }
    if language and language.lower() not in ("auto", ""):
        form["language"] = (None, language)
    if translate:
        form["translate"] = (None, "true")
    return requests.post(f"{WHISPER_URL}/inference", files=form, timeout=600)


def get_audio_bytes(req_files):
    """Read audio_file from request. Returns (bytes, filename) or (None, None)."""
    f = req_files.get("audio_file")
    if not f:
        return None, None
    data = f.read()
    log.info(f"Received audio_file: {len(data)} bytes, first4={data[:4].hex() if data else 'empty'}")
    return data, f.filename or "audio.wav"


# ---------------------------------------------------------------------------
# Bazarr connection-test endpoints
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
@app.route("/status", methods=["GET"])
def index():
    """Bazarr checks this for the version string."""
    return jsonify({"version": VERSION})


@app.route("/health", methods=["GET"])
def health():
    try:
        r = requests.get(WHISPER_URL, timeout=5)
        ok = r.status_code < 500
    except Exception:
        ok = False
    return jsonify({"status": "ok" if ok else "backend-unavailable", "version": VERSION}), \
           200 if ok else 503


# ---------------------------------------------------------------------------
# Language detection - Bazarr calls this before /asr
# ---------------------------------------------------------------------------

@app.route("/detect-language", methods=["POST"])
@app.route("//detect-language", methods=["POST"])
def detect_language():
    data, filename = get_audio_bytes(request.files)
    if not data:
        return "No audio_file provided", 400

    try:
        # Only send first 30 seconds worth of audio for speed
        # WAV header is 44 bytes, 16kHz mono int16 = 32000 bytes/sec
        # 30s = 960000 bytes of PCM + 44 byte header
        THIRTY_SEC_BYTES = 44 + (16000 * 2 * 30)
        trimmed = data[:THIRTY_SEC_BYTES] if len(data) > THIRTY_SEC_BYTES else data

        r = call_whisper(trimmed, filename, response_format="verbose_json")
        if r.status_code != 200:
            log.error(f"whisper-server error {r.status_code}: {r.text[:300]}")
            return f"whisper-server error: {r.text}", r.status_code

        lang_code = r.json().get("language", "en")
        lang_name = LANG_NAMES.get(lang_code, lang_code)
        log.info(f"detect-language: {lang_name} ({lang_code})")
        return jsonify({"detected_language": lang_name, "language_code": lang_code})

    except Exception as e:
        log.exception(f"detect-language error: {e}")
        return f"Internal error: {e}", 500


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

@app.route("/asr", methods=["POST"])
@app.route("//asr", methods=["POST"])
def asr():
    task     = request.args.get("task", "transcribe")
    language = request.args.get("language", "auto")
    output   = request.args.get("output", "srt")

    data, filename = get_audio_bytes(request.files)
    if not data:
        return "No audio_file provided", 400

    fmt_map = {"srt": "srt", "vtt": "vtt", "txt": "text", "text": "text", "json": "verbose_json"}
    response_format = fmt_map.get(output.lower(), "srt")
    translate = (task == "translate")

    try:
        log.info(f"asr: format={response_format} lang={language} task={task} size={len(data)}bytes")
        r = call_whisper(data, filename, response_format=response_format,
                         language=language, translate=translate)
        log.info(f"whisper-server: HTTP {r.status_code}, {len(r.content)} bytes back")

        if r.status_code != 200:
            log.error(f"whisper-server error: {r.text[:500]}")
            return f"whisper-server error: {r.text}", r.status_code

        content_type = r.headers.get("Content-Type", "text/plain; charset=utf-8")
        return Response(r.content, status=200, content_type=content_type)

    except requests.exceptions.Timeout:
        log.error("Timed out waiting for whisper-server")
        return "Transcription timed out", 504
    except Exception as e:
        log.exception(f"asr error: {e}")
        return f"Internal error: {e}", 500


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("SHIM_PORT", 9000))
    log.info(f"Bazarr shim listening on 0.0.0.0:{port}")
    log.info(f"Forwarding inference to {WHISPER_URL}")
    app.run(host="0.0.0.0", port=port, threaded=True)
