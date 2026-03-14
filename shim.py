"""
Bazarr <-> whisper.cpp shim

Mimics the SubGen API that Bazarr expects:
  GET  /                                            -> version string (SubGen format)
  GET  /status                                      -> same
  GET  /health                                      -> health check
  POST /detect-language?encode=false                -> language detection
  POST //detect-language?encode=false               -> same (Bazarr double-slash variant)
  POST /asr?task=transcribe&language=en&output=srt  -> transcription
  POST //asr?task=transcribe&language=en&output=srt -> same (Bazarr double-slash variant)
       body: multipart, field name = "audio_file"

whisper.cpp server expects:
  POST /inference
       body: multipart, field name = "file"
       params: response_format, language, translate (as form fields)
"""

import os
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

# Must match SubGen's version string format so Bazarr can parse it
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


def convert_to_wav(audio_bytes, duration_limit=None):
    """Convert audio bytes to 16kHz mono WAV via ffmpeg. Returns path to temp WAV file."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = tmp.name

    cmd = ["ffmpeg", "-y", "-i", "pipe:0", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le"]
    if duration_limit:
        cmd += ["-t", str(duration_limit)]
    cmd.append(tmp_wav)

    proc = subprocess.run(cmd, input=audio_bytes, capture_output=True, timeout=120)
    if proc.returncode != 0:
        os.unlink(tmp_wav)
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode()}")
    return tmp_wav


def call_whisper_inference(wav_path, response_format="verbose_json", language=None, translate=False):
    """POST a WAV file to whisper-server /inference. Returns requests.Response."""
    with open(wav_path, "rb") as f:
        form = {
            "file": (os.path.basename(wav_path), f, "audio/wav"),
            "response_format": (None, response_format),
            "temperature": (None, "0.0"),
            "temperature_inc": (None, "0.2"),
        }
        if language and language.lower() not in ("auto", ""):
            form["language"] = (None, language)
        if translate:
            form["translate"] = (None, "true")

        return requests.post(f"{WHISPER_URL}/inference", files=form, timeout=600)


# ---------------------------------------------------------------------------
# Bazarr connection-test endpoints
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
@app.route("/status", methods=["GET"])
def index():
    """Bazarr parses this for the version string. Must match SubGen format."""
    return jsonify({"version": VERSION})


@app.route("/health", methods=["GET"])
def health():
    try:
        r = requests.get(WHISPER_URL, timeout=5)
        backend_ok = r.status_code < 500
    except Exception:
        backend_ok = False
    return jsonify({"status": "ok" if backend_ok else "backend-unavailable", "version": VERSION}), \
           200 if backend_ok else 503


# ---------------------------------------------------------------------------
# Language detection - Bazarr calls this BEFORE /asr
# ---------------------------------------------------------------------------

@app.route("/detect-language", methods=["POST"])
@app.route("//detect-language", methods=["POST"])
def detect_language():
    """
    POST /detect-language?encode=false
    Bazarr sends the audio file and expects back the detected language.
    Only uses first 30 seconds for speed.
    """
    audio_file = request.files.get("audio_file")
    if not audio_file:
        return "No audio_file provided", 400

    tmp_wav = None
    try:
        log.info("detect-language: converting audio (first 30s)")
        tmp_wav = convert_to_wav(audio_file.read(), duration_limit=30)

        r = call_whisper_inference(tmp_wav, response_format="verbose_json")

        if r.status_code != 200:
            log.error(f"whisper-server error {r.status_code}: {r.text[:200]}")
            return f"whisper-server error: {r.text}", r.status_code

        data = r.json()
        lang_code = data.get("language", "en")
        lang_name = LANG_NAMES.get(lang_code, lang_code)

        log.info(f"detect-language result: {lang_name} ({lang_code})")
        return jsonify({"detected_language": lang_name, "language_code": lang_code})

    except Exception as e:
        log.exception(f"detect-language error: {e}")
        return f"Internal error: {e}", 500

    finally:
        if tmp_wav and os.path.exists(tmp_wav):
            os.unlink(tmp_wav)


# ---------------------------------------------------------------------------
# Main transcription endpoint
# ---------------------------------------------------------------------------

@app.route("/asr", methods=["POST"])
@app.route("//asr", methods=["POST"])
def asr():
    """
    POST /asr?task=transcribe&language=en&output=srt
    Body: multipart with field "audio_file"
    """
    task     = request.args.get("task", "transcribe")
    language = request.args.get("language", "auto")
    output   = request.args.get("output", "srt")

    audio_file = request.files.get("audio_file")
    if not audio_file:
        return "No audio_file provided", 400

    fmt_map = {"srt": "srt", "vtt": "vtt", "txt": "text", "text": "text", "json": "verbose_json"}
    response_format = fmt_map.get(output.lower(), "srt")
    translate = (task == "translate")

    tmp_wav = None
    try:
        log.info(f"asr: converting audio (format={response_format}, lang={language}, task={task})")
        tmp_wav = convert_to_wav(audio_file.read())

        r = call_whisper_inference(
            tmp_wav,
            response_format=response_format,
            language=language,
            translate=translate
        )

        log.info(f"whisper-server responded: HTTP {r.status_code}, {len(r.content)} bytes")

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

    finally:
        if tmp_wav and os.path.exists(tmp_wav):
            os.unlink(tmp_wav)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("SHIM_PORT", 9000))
    log.info(f"Bazarr shim listening on 0.0.0.0:{port}")
    log.info(f"Forwarding inference to {WHISPER_URL}")
    app.run(host="0.0.0.0", port=port, threaded=True)
