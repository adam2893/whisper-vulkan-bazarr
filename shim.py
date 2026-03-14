"""
Bazarr <-> whisper.cpp shim (SubGen-compatible API)
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


def convert_to_wav(source, duration_limit=None):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = tmp.name

    cmd = ["ffmpeg", "-y"]
    if isinstance(source, str):
        log.info(f"ffmpeg: reading from path: {source}")
        cmd += ["-i", source]
        input_bytes = None
    else:
        log.info(f"ffmpeg: reading {len(source)} bytes from pipe")
        cmd += ["-i", "pipe:0"]
        input_bytes = source

    cmd += ["-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le"]
    if duration_limit:
        cmd += ["-t", str(duration_limit)]
    cmd.append(tmp_wav)

    proc = subprocess.run(cmd, input=input_bytes, capture_output=True, timeout=300)
    if proc.returncode != 0:
        os.unlink(tmp_wav)
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode()}")
    return tmp_wav


def get_audio_source(req_files, req_form, encode_param):
    """
    Debug version - logs everything Bazarr sends so we can see exactly what arrives.
    """
    encode = (encode_param or "true").lower()

    # Log all incoming fields
    log.info(f"encode param = {encode!r}")
    log.info(f"form keys: {list(req_form.keys())}")
    log.info(f"files keys: {list(req_files.keys())}")

    for k in req_form:
        v = req_form.get(k, "")
        log.info(f"  form[{k!r}] = {repr(v[:300])}")

    for k in req_files:
        f = req_files[k]
        data = f.read(64)
        f.seek(0)
        log.info(f"  files[{k!r}] filename={f.filename!r} content_type={f.content_type!r} first64={data.hex()}")

    # Try form field path first
    if "audio_file" in req_form:
        path = req_form.get("audio_file", "").strip()
        if path and not path.startswith('\x00'):
            log.info(f"Using form path: {path!r}")
            return path

    # Fall back to file upload bytes
    if "audio_file" in req_files:
        data = req_files["audio_file"].read()
        log.info(f"Using uploaded bytes: {len(data)} bytes")
        return data

    return None


def call_whisper_inference(wav_path, response_format="verbose_json", language=None, translate=False):
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
# Language detection
# ---------------------------------------------------------------------------

@app.route("/detect-language", methods=["POST"])
@app.route("//detect-language", methods=["POST"])
def detect_language():
    encode_param = request.args.get("encode", "true")
    source = get_audio_source(request.files, request.form, encode_param)
    if source is None:
        return "No audio_file provided", 400

    tmp_wav = None
    try:
        tmp_wav = convert_to_wav(source, duration_limit=30)
        r = call_whisper_inference(tmp_wav, response_format="verbose_json")
        if r.status_code != 200:
            return f"whisper-server error: {r.text}", r.status_code
        data = r.json()
        lang_code = data.get("language", "en")
        lang_name = LANG_NAMES.get(lang_code, lang_code)
        log.info(f"detect-language: {lang_name} ({lang_code})")
        return jsonify({"detected_language": lang_name, "language_code": lang_code})
    except Exception as e:
        log.exception(f"detect-language error: {e}")
        return f"Internal error: {e}", 500
    finally:
        if tmp_wav and os.path.exists(tmp_wav):
            os.unlink(tmp_wav)


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

@app.route("/asr", methods=["POST"])
@app.route("//asr", methods=["POST"])
def asr():
    task         = request.args.get("task", "transcribe")
    language     = request.args.get("language", "auto")
    output       = request.args.get("output", "srt")
    encode_param = request.args.get("encode", "true")

    source = get_audio_source(request.files, request.form, encode_param)
    if source is None:
        return "No audio_file provided", 400

    fmt_map = {"srt": "srt", "vtt": "vtt", "txt": "text", "text": "text", "json": "verbose_json"}
    response_format = fmt_map.get(output.lower(), "srt")
    translate = (task == "translate")

    tmp_wav = None
    try:
        log.info(f"asr: format={response_format} lang={language} task={task} encode={encode_param}")
        tmp_wav = convert_to_wav(source)
        r = call_whisper_inference(tmp_wav, response_format=response_format,
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
    finally:
        if tmp_wav and os.path.exists(tmp_wav):
            os.unlink(tmp_wav)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("SHIM_PORT", 9000))
    log.info(f"Bazarr shim listening on 0.0.0.0:{port}")
    log.info(f"Forwarding inference to {WHISPER_URL}")
    app.run(host="0.0.0.0", port=port, threaded=True)
