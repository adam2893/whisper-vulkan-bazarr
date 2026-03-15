"""
Bazarr <-> whisper.cpp shim (SubGen-compatible API)

Bazarr sends raw 16kHz mono int16 PCM (no WAV header, first bytes = ffffffff).
We wrap it in a WAV header before forwarding to whisper-server.
"""

import io
import os
import wave
import logging
import requests
from flask import Flask, request, Response, jsonify

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [shim] %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = None  # No limit - full episode audio can be 100MB+

WHISPER_URL = os.environ.get("WHISPER_INTERNAL_URL", "http://127.0.0.1:8080")
VERSION = "Subgen 1.0.0, stable-ts 0.0.0, faster-whisper 0.0.0 (whisper-cpp-vulkan)"
SAMPLE_RATE = 16000

LANG_NAMES = {
    "en": "english", "fr": "french", "de": "german", "es": "spanish",
    "it": "italian", "pt": "portuguese", "nl": "dutch", "pl": "polish",
    "ru": "russian", "ja": "japanese", "zh": "chinese", "ko": "korean",
    "ar": "arabic", "hi": "hindi", "sv": "swedish", "da": "danish",
    "fi": "finnish", "no": "norwegian", "tr": "turkish", "cs": "czech",
    "ro": "romanian", "hu": "hungarian", "uk": "ukrainian", "el": "greek",
    "he": "hebrew", "th": "thai", "vi": "vietnamese", "id": "indonesian",
}


def to_wav(pcm_bytes):
    """Wrap raw int16 16kHz mono PCM bytes in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)      # int16 = 2 bytes per sample
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_bytes)
    buf.seek(0)
    return buf


def is_wav(data):
    return data[:4] == b'RIFF' and data[8:12] == b'WAVE'


def prepare_audio(data):
    """Return a BytesIO WAV buffer ready for whisper-server."""
    if is_wav(data):
        log.info("Audio is already WAV")
        return io.BytesIO(data)
    log.info("Audio is raw PCM, wrapping in WAV header")
    return to_wav(data)


def call_whisper(wav_buf, response_format="verbose_json", language=None, translate=False):
    form = {
        "file": ("audio.wav", wav_buf, "audio/wav"),
        "response_format": (None, response_format),
        "temperature": (None, "0.0"),
        "temperature_inc": (None, "0.2"),
    }
    if language and language.lower() not in ("auto", ""):
        form["language"] = (None, language)
    if translate:
        form["translate"] = (None, "true")
    return requests.post(f"{WHISPER_URL}/inference", files=form, timeout=600)


def get_audio(req_files):
    f = req_files.get("audio_file")
    if not f:
        return None
    data = f.read()
    log.info(f"Received {len(data)} bytes, first4={data[:4].hex() if data else 'empty'}")
    return data


# ---------------------------------------------------------------------------
# Bazarr connection-test endpoints
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
@app.route("/status", methods=["GET"])
def index():
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
# Language detection
# ---------------------------------------------------------------------------

@app.route("/detect-language", methods=["POST"])
@app.route("//detect-language", methods=["POST"])
def detect_language():
    data = get_audio(request.files)
    if not data:
        return "No audio_file provided", 400

    try:
        # Trim to first 30 seconds for speed: 30s * 16000hz * 2 bytes = 960000 bytes
        trimmed = data[:960000] if not is_wav(data) else data
        wav_buf = prepare_audio(trimmed)

        r = call_whisper(wav_buf, response_format="verbose_json")
        if r.status_code != 200:
            log.error(f"whisper-server {r.status_code}: {r.text[:300]}")
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

    data = get_audio(request.files)
    if not data:
        return "No audio_file provided", 400

    fmt_map = {"srt": "srt", "vtt": "vtt", "txt": "text", "text": "text", "json": "verbose_json"}
    response_format = fmt_map.get(output.lower(), "srt")
    translate = (task == "translate")

    try:
        log.info(f"asr: format={response_format} lang={language} task={task} size={len(data)}b")
        wav_buf = prepare_audio(data)

        r = call_whisper(wav_buf, response_format=response_format,
                         language=language, translate=translate)
        log.info(f"whisper-server: HTTP {r.status_code}, {len(r.content)} bytes")

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
