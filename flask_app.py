"""Flask web application for BIS Multi-Media Steganography Tool.

Provides a web interface for:
- Text steganography (hide text in text using zero-width characters)
- Audio steganography (hide text in WAV files using LSB)
- Video steganography (hide text in video frames using LSB)
All with optional AES-256-GCM encryption.
"""

from __future__ import annotations

import io
import math
import shutil
import struct
import subprocess
import uuid
import wave
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 256 * 1024 * 1024  # 256MB max upload (longer media)

# Ensure directories exist
Path("uploads").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)

# Check for ffmpeg (used for video+audio muxing)
FFMPEG = shutil.which("ffmpeg")
if not FFMPEG:
    try:
        import imageio_ffmpeg
        FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass


# ─── Audio Synthesis Helpers ─────────────────────────────────

_NOTES = {
    'C3': 130.81, 'D3': 146.83, 'E3': 164.81, 'F3': 174.61,
    'G3': 196.00, 'A3': 220.00, 'Bb3': 233.08, 'B3': 246.94,
    'C4': 261.63, 'Db4': 277.18, 'D4': 293.66, 'Eb4': 311.13,
    'E4': 329.63, 'F4': 349.23, 'F#4': 369.99, 'G4': 392.00,
    'Ab4': 415.30, 'A4': 440.00, 'Bb4': 466.16, 'B4': 493.88,
    'C5': 523.25, 'Db5': 554.37, 'D5': 587.33, 'D#5': 622.25,
    'Eb5': 622.25, 'E5': 659.25, 'F5': 698.46, 'G5': 783.99,
    'A5': 880.00, 'B5': 987.77, 'C6': 1046.50,
}


def _envelope(n, sr, attack=0.02, decay=0.08, sustain=0.6, release=0.15):
    """Create an ADSR envelope of length *n*."""
    env = np.ones(n, dtype=np.float64)
    a = min(int(attack * sr), n)
    d = min(int(decay * sr), n - a)
    r = min(int(release * sr), n)
    s_end = max(a + d, n - r)
    if a > 0:
        env[:a] = np.linspace(0, 1, a)
    if d > 0:
        env[a:a + d] = np.linspace(1, sustain, d)
    if a + d < s_end:
        env[a + d:s_end] = sustain
    if r > 0 and s_end < n:
        env[s_end:] = np.linspace(sustain, 0, n - s_end)
    return env


def _tone(freq, dur, sr, harmonics=4):
    """Synthesize a tone with harmonics and ADSR envelope."""
    n = int(sr * dur)
    t = np.arange(n, dtype=np.float64) / sr
    sig = np.zeros(n)
    for h in range(1, harmonics + 1):
        sig += np.sin(2 * np.pi * freq * h * t) / (h * h)
    mx = np.max(np.abs(sig))
    if mx > 0:
        sig /= mx
    sig *= _envelope(n, sr, release=min(0.12, dur * 0.4))
    return sig


def _note(name, dur, sr, harmonics=4, vol=0.8):
    """Play a named note with envelope.  ``'R'`` = rest."""
    if name == 'R':
        return np.zeros(int(sr * dur))
    freq = _NOTES.get(name, 440.0) if isinstance(name, str) else float(name)
    return _tone(freq, dur, sr, harmonics) * vol


def _seq(notes_list, sr, harmonics=4, gap=0.005):
    """Play a sequence of ``(note, duration [, volume])`` tuples."""
    parts = []
    for item in notes_list:
        vol = item[2] if len(item) >= 3 else 0.8
        parts.append(_note(item[0], item[1], sr, harmonics, vol))
        if gap > 0:
            parts.append(np.zeros(max(1, int(sr * gap))))
    return np.concatenate(parts) if parts else np.zeros(1)


def _chord(note_names, dur, sr, harmonics=3, vol=0.35):
    """Play multiple notes simultaneously as a chord."""
    sigs = [_note(nn, dur, sr, harmonics, vol) for nn in note_names]
    mx = max(len(s) for s in sigs)
    out = np.zeros(mx)
    for s in sigs:
        out[:len(s)] += s
    return out


def _pad(sig, n):
    """Pad or tile signal to exactly *n* samples."""
    if len(sig) == 0:
        return np.zeros(n)
    if len(sig) >= n:
        return sig[:n]
    return np.tile(sig, math.ceil(n / len(sig)))[:n]


def _write_wav_buf(samples_i16, sr=44100):
    """Write int16 samples to an in-memory WAV buffer."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples_i16.tobytes())
    buf.seek(0)
    return buf


def _mux_av(video_path, audio_i16, sr, out_path):
    """Mux a video file with int16 audio via ffmpeg.  Returns True on success."""
    if not FFMPEG:
        return False
    tmp_wav = Path(str(video_path) + ".tmp.wav")
    try:
        buf = _write_wav_buf(audio_i16, sr)
        tmp_wav.write_bytes(buf.read())
        r = subprocess.run(
            [FFMPEG, "-y", "-i", str(video_path), "-i", str(tmp_wav),
             "-c:v", "copy", "-c:a", "pcm_s16le", "-shortest", str(out_path)],
            capture_output=True, timeout=60,
        )
        return r.returncode == 0 and out_path.exists()
    except Exception:
        return False
    finally:
        tmp_wav.unlink(missing_ok=True)


# ─── Audio Template Generators ───────────────────────────────


# Default durations (seconds) per template – realistic lengths
_AUDIO_DURATIONS = {
    "guitar": 120,   # 2 min – fingerpicked arpeggios
    "lullaby": 180,  # 3 min – gentle music-box
    "jazz": 150,     # 2.5 min – smooth chords
    "ocean": 300,    # 5 min – ambient waves
    "rain": 300,     # 5 min – ambient rain
    "melody": 120,   # 2 min – classical piece
    "chimes": 240,   # 4 min – wind chimes
    "birds": 300,    # 5 min – morning birds
}

_VIDEO_DURATIONS = {
    "gradient": 60,    # 1 min – rainbow gradient
    "waves": 60,       # 1 min – sine-wave pattern
    "sky": 90,         # 1.5 min – drifting clouds
    "abstract": 60,    # 1 min – colour blocks
    "starfield": 90,   # 1.5 min – space warp
    "fireflies": 120,  # 2 min – glowing night
    "sunrise": 180,    # 3 min – sunrise horizon
    "matrix": 60,      # 1 min – digital rain
}


def _gen_audio(tid, sr=44100, dur=8):
    """Return int16 numpy array for an audio template."""
    n = sr * dur
    t = np.linspace(0, dur, n, endpoint=False)

    if tid == "guitar":
        # Finger-picked arpeggio: Am → C → G → Em
        pats = [
            [('A3', .25), ('C4', .25), ('E4', .25), ('A4', .25),
             ('E4', .25), ('C4', .25)],
            [('C4', .25), ('E4', .25), ('G4', .25), ('C5', .25),
             ('G4', .25), ('E4', .25)],
            [('G3', .25), ('B3', .25), ('D4', .25), ('G4', .25),
             ('D4', .25), ('B3', .25)],
            [('E3', .25), ('G3', .25), ('B3', .25), ('E4', .25),
             ('B3', .25), ('G3', .25)],
        ]
        melody = [note for p in pats * 3 for note in p]
        return _pad(_seq(melody, sr, 5, 0.0) * 12000, n).astype(np.int16)

    if tid == "lullaby":
        # Twinkle Twinkle Little Star (music-box timbre)
        m = [
            ('C4', .4), ('C4', .4), ('G4', .4), ('G4', .4),
            ('A4', .4), ('A4', .4), ('G4', .7), ('R', .1),
            ('F4', .4), ('F4', .4), ('E4', .4), ('E4', .4),
            ('D4', .4), ('D4', .4), ('C4', .7), ('R', .3),
            ('G4', .4), ('G4', .4), ('F4', .4), ('F4', .4),
            ('E4', .4), ('E4', .4), ('D4', .7), ('R', .1),
            ('G4', .4), ('G4', .4), ('F4', .4), ('F4', .4),
            ('E4', .4), ('E4', .4), ('D4', .7), ('R', .3),
        ]
        return _pad(_seq(m, sr, 2, 0.01) * 10000, n).astype(np.int16)

    if tid == "jazz":
        # Jazz piano chord voicings: Cmaj7 → Dm7 → G7 → Cmaj7
        chords = [
            (['C4', 'E4', 'G4', 'B4'], 1.4),
            (['D4', 'F4', 'A4', 'C5'], 1.4),
            (['G3', 'B3', 'D4', 'F4'], 1.4),
            (['C4', 'E4', 'G4', 'B4'], 1.4),
        ]
        parts = []
        for ch, d in chords * 2:
            parts.append(_chord(ch, d, sr))
            parts.append(np.zeros(int(sr * 0.1)))
        return _pad(np.concatenate(parts) * 10000, n).astype(np.int16)

    if tid == "ocean":
        # Ocean waves: brown noise + slow amplitude swell
        brown = np.cumsum(np.random.randn(n))
        brown -= np.mean(brown)
        brown /= max(1e-9, np.max(np.abs(brown)))
        wave_env = 0.5 + 0.5 * np.sin(2 * np.pi * 0.12 * t)
        return (brown * wave_env * 14000).astype(np.int16)

    if tid == "rain":
        # Rain ambience: filtered noise + droplet clicks
        noise = np.random.randn(n) * 0.3
        filt = np.convolve(noise, np.ones(8) / 8, mode='same')
        num_drops = max(300, int(38 * dur))  # ~38 drops/s
        for _ in range(num_drops):
            pos = np.random.randint(0, n - 500)
            dl = np.random.randint(50, 200)
            freq = np.random.uniform(2000, 6000)
            dt = np.arange(dl, dtype=np.float64) / sr
            filt[pos:pos + dl] += np.sin(2 * np.pi * freq * dt) * np.exp(-30 * dt) * 0.25
        return (filt * 14000).astype(np.int16)

    if tid == "melody":
        # Für Elise opening (Beethoven — public domain)
        m = [
            ('E5', .28), ('D#5', .28), ('E5', .28), ('D#5', .28), ('E5', .28),
            ('B4', .28), ('D5', .28), ('C5', .28), ('A4', .55), ('R', .12),
            ('C4', .28), ('E4', .28), ('A4', .28), ('B4', .55), ('R', .12),
            ('E4', .28), ('Ab4', .28), ('B4', .28), ('C5', .55), ('R', .12),
            ('E4', .28),
            ('E5', .28), ('D#5', .28), ('E5', .28), ('D#5', .28), ('E5', .28),
            ('B4', .28), ('D5', .28), ('C5', .28), ('A4', .55), ('R', .12),
        ]
        return _pad(_seq(m, sr, 3, 0.0) * 11000, n).astype(np.int16)

    if tid == "chimes":
        # Wind chimes: random metallic tones with long decay
        sig = np.zeros(n, dtype=np.float64)
        freqs = [523.25, 587.33, 659.25, 783.99, 880.0, 1046.5, 1174.66, 1318.51]
        num_chimes = max(35, int(4.4 * dur))  # ~4-5 chimes/s
        for _ in range(num_chimes):
            pos = np.random.randint(0, n - sr)
            f = np.random.choice(freqs)
            cl = min(np.random.randint(sr // 2, sr * 2), n - pos)
            ct = np.arange(cl, dtype=np.float64) / sr
            c = np.sin(2 * np.pi * f * ct) * np.exp(-2.5 * ct)
            c += .3 * np.sin(2 * np.pi * f * 2.01 * ct) * np.exp(-3 * ct)
            sig[pos:pos + cl] += c * .3
        mx = max(1e-9, np.max(np.abs(sig)))
        return (sig / mx * 13000).astype(np.int16)

    if tid == "birds":
        # Morning birds: pink-noise base + chirp sweeps
        white = np.random.randn(n)
        ff = np.fft.rfftfreq(n, 1.0 / sr)
        ff[0] = 1
        pink = np.fft.irfft(np.fft.rfft(white) / np.sqrt(ff), n)
        pink /= max(1e-9, np.max(np.abs(pink)))
        sig = pink * 0.15
        num_chirps = max(50, int(6.3 * dur))  # ~6 chirps/s
        for _ in range(num_chirps):
            pos = np.random.randint(0, n - sr // 2)
            cl = min(np.random.randint(600, 3000), n - pos)
            ct = np.arange(cl, dtype=np.float64) / sr
            bf = np.random.uniform(2000, 5000)
            sweep = bf + bf * .5 * ct / max(1e-9, cl / sr)
            chirp = np.sin(2 * np.pi * np.cumsum(sweep) / sr) * np.exp(-8 * ct)
            sig[pos:pos + cl] += chirp * .3
        mx = max(1e-9, np.max(np.abs(sig)))
        return (sig / mx * 12000).astype(np.int16)

    return None


# ─── Video Frame & Audio Generators ─────────────────────────


def _gen_frame(tid, fi, total, w, h):
    """Generate a single video frame (BGR uint8 ndarray)."""
    frac = fi / total

    if tid == "gradient":
        y = np.arange(h, dtype=np.float64)
        phase = 2 * np.pi * (y / h + frac)
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :, 2] = (128 + 127 * np.sin(phase)).astype(np.uint8).reshape(-1, 1)
        frame[:, :, 1] = (128 + 127 * np.sin(phase + 2.094)).astype(np.uint8).reshape(-1, 1)
        frame[:, :, 0] = (128 + 127 * np.sin(phase + 4.189)).astype(np.uint8).reshape(-1, 1)
        return frame

    if tid == "waves":
        x = np.arange(w, dtype=np.float64).reshape(1, -1)
        y = np.arange(h, dtype=np.float64).reshape(-1, 1)
        v = (128 + 127 * np.sin(x / 30 + y / 30 + frac * 2 * np.pi)).astype(np.uint8)
        return np.stack([v, v >> 1, (255 - v).astype(np.uint8)], axis=2)

    if tid == "sky":
        frame = np.full((h, w, 3), [210, 160, 60], dtype=np.uint8)
        offset = int(frac * w)
        for cx_ in range(-80, w + 80, 120):
            px = (cx_ + offset) % (w + 80) - 40
            cv2.ellipse(frame, (px, 60), (60, 25), 0, 0, 360, (240, 240, 240), -1)
            cv2.ellipse(frame, (px + 30, 50), (40, 20), 0, 0, 360, (245, 245, 245), -1)
        return frame

    if tid == "abstract":
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        np.random.seed(fi)
        for _ in range(8):
            x1 = np.random.randint(0, w)
            y1 = np.random.randint(0, h)
            w2 = np.random.randint(30, 100)
            h2 = np.random.randint(30, 80)
            col = tuple(int(c) for c in np.random.randint(40, 220, 3))
            cv2.rectangle(frame, (x1, y1), (x1 + w2, y1 + h2), col, -1)
        return frame

    if tid == "starfield":
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        np.random.seed(42)
        ns = 200
        angles = np.random.uniform(0, 2 * np.pi, ns)
        speeds = np.random.uniform(0.2, 2.0, ns)
        brights = np.random.randint(120, 255, ns)
        cx, cy, diag = w // 2, h // 2, max(w, h)
        for i in range(ns):
            dist = (speeds[i] * frac * diag * 1.5) % diag
            x = int(cx + dist * np.cos(angles[i]))
            y = int(cy + dist * np.sin(angles[i]))
            if 0 <= x < w and 0 <= y < h:
                sz = max(1, int(dist / diag * 3))
                c = int(brights[i])
                cv2.circle(frame, (x, y), sz, (c, c, c), -1)
        return frame

    if tid == "fireflies":
        frame = np.full((h, w, 3), [12, 18, 8], dtype=np.uint8)
        np.random.seed(42)
        nf = 30
        bx = np.random.uniform(20, w - 20, nf)
        by = np.random.uniform(20, h - 20, nf)
        for i in range(nf):
            x = int(bx[i] + 15 * np.sin(2 * np.pi * (frac * 2 + i * .3)))
            y = int(by[i] + 10 * np.cos(2 * np.pi * (frac * 1.5 + i * .7)))
            br = max(0.0, .5 + .5 * np.sin(2 * np.pi * (frac * 4 + i * 1.1)))
            if br > .25:
                gr = int(min(255, 255 * br))
                cv2.circle(frame, (x, y), max(2, int(8 * br)), (0, gr, gr // 3), -1)
        frame = cv2.GaussianBlur(frame, (5, 5), 2)
        return frame

    if tid == "sunrise":
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        horizon = int(h * 0.65)
        y_arr = np.arange(h)
        sky_mask = y_arr < horizon
        p = y_arr[sky_mask].astype(np.float64) / horizon
        frame[sky_mask, :, 2] = np.clip(15 + 200 * frac * (1 - p), 0, 255).astype(np.uint8).reshape(-1, 1)
        frame[sky_mask, :, 1] = np.clip(10 + 120 * frac * (1 - p * .5), 0, 255).astype(np.uint8).reshape(-1, 1)
        frame[sky_mask, :, 0] = np.clip(40 + 80 * frac, 0, 255).astype(np.uint8)
        frame[~sky_mask, :] = [20, int(35 + 25 * frac), int(25 + 15 * frac)]
        sun_y = int(horizon - horizon * .35 * frac)
        cv2.circle(frame, (w // 2, sun_y), 30 + int(12 * frac),
                   (30, int(80 * frac), int(160 * frac)), -1)
        cv2.circle(frame, (w // 2, sun_y), 18 + int(8 * frac),
                   (60, int(180 + 50 * frac), 255), -1)
        return frame

    if tid == "matrix":
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        np.random.seed(42)
        cols = w // 10
        col_speeds = np.random.randint(3, 12, cols)
        col_offsets = np.random.randint(0, h, cols)
        for ci in range(cols):
            x = ci * 10 + 2
            head = (col_offsets[ci] + fi * col_speeds[ci]) % (h + 60)
            for row in range(0, h, 12):
                yy = head - row
                if 0 <= yy < h:
                    brightness = max(0, 255 - row * 8)
                    if brightness > 15:
                        y1, y2 = max(0, yy), min(h, yy + 8)
                        x1, x2 = max(0, x), min(w, x + 6)
                        if y1 < y2 and x1 < x2:
                            frame[y1:y2, x1:x2, 1] = min(255, brightness)
        return frame

    return None


# Video template → matching audio template mapping
_VIDEO_AUDIO_MAP = {
    "gradient": "chimes",
    "waves": "ocean",
    "sky": "birds",
    "abstract": "jazz",
    "starfield": "ocean",
    "fireflies": "rain",
    "sunrise": "lullaby",
    "matrix": "melody",
}


# ─── Routes ──────────────────────────────────────────────────


@app.route("/")
def index():
    """Serve the landing / home page."""
    return render_template("index.html", active_page="home")


@app.route("/encrypt")
def encrypt_page():
    """Serve the encrypt page."""
    return render_template("encrypt.html", active_page="encrypt")


@app.route("/decrypt")
def decrypt_page():
    """Serve the decrypt page."""
    return render_template("decrypt.html", active_page="decrypt")


@app.route("/about")
def about_page():
    """Serve the about / AI technology page."""
    return render_template("about.html", active_page="about")


# ─── Text Steganography Routes ───────────────────────────


@app.route("/api/encrypt-text", methods=["POST"])
def encrypt_text():
    """Hide secret text within cover text using zero-width characters."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid request body"}), 400

    cover_text = data.get("cover_text", "").strip()
    secret_text = data.get("secret_text", "").strip()
    password = data.get("password", "").strip() or None

    if not cover_text:
        return jsonify({"error": "No cover text provided"}), 400
    if not secret_text:
        return jsonify({"error": "No secret text provided"}), 400

    try:
        from bis.stego.text_stego import hide_text_in_text

        stego_text = hide_text_in_text(cover_text, secret_text, password=password)

        return jsonify({
            "success": True,
            "stego_text": stego_text,
            "encrypted": password is not None,
            "message": "Text hidden successfully!"
                       + (" (AES-256 encrypted)" if password else ""),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/decrypt-text", methods=["POST"])
def decrypt_text():
    """Extract hidden text from stego text."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid request body"}), 400

    stego_text = data.get("stego_text", "")
    password = data.get("password", "").strip() or None

    if not stego_text:
        return jsonify({"error": "No stego text provided"}), 400

    try:
        from bis.stego.text_stego import extract_text_from_text

        text = extract_text_from_text(stego_text, password=password)

        return jsonify({
            "success": True,
            "text": text,
            "message": "Secret text extracted successfully!",
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── Audio Steganography Routes ──────────────────────────


@app.route("/api/encrypt-audio", methods=["POST"])
def encrypt_audio():
    """Hide text within an uploaded WAV audio file."""
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    text = request.form.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    password = request.form.get("password", "").strip() or None

    try:
        from bis.stego.audio import hide_text_in_audio

        audio_file = request.files["audio"]
        uid = uuid.uuid4().hex[:8]
        input_path = Path("uploads") / f"audio_in_{uid}.wav"
        output_path = Path("outputs") / f"stego_audio_{uid}.wav"

        audio_file.save(str(input_path))

        metadata = hide_text_in_audio(
            str(input_path), text, str(output_path), password=password
        )

        # Clean up input
        input_path.unlink(missing_ok=True)

        return jsonify({
            "success": True,
            "audio_url": f"/output/{output_path.name}",
            "metadata": {
                "capacity": metadata["capacity"],
                "data_size": metadata["data_size"],
                "duration": metadata["duration_sec"],
                "sample_rate": metadata["sample_rate"],
                "channels": metadata["channels"],
            },
            "encrypted": password is not None,
            "message": f"Text hidden in audio! ({metadata['data_size']} bytes in {metadata['duration_sec']}s audio)"
                       + (" (AES-256 encrypted)" if password else ""),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/decrypt-audio", methods=["POST"])
def decrypt_audio():
    """Extract hidden text from a stego WAV file."""
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    password = request.form.get("password", "").strip() or None

    try:
        from bis.stego.audio import extract_text_from_audio

        audio_file = request.files["audio"]
        uid = uuid.uuid4().hex[:8]
        input_path = Path("uploads") / f"audio_dec_{uid}.wav"
        audio_file.save(str(input_path))

        text = extract_text_from_audio(str(input_path), password=password)

        # Clean up
        input_path.unlink(missing_ok=True)

        return jsonify({
            "success": True,
            "text": text,
            "message": "Text extracted from audio!",
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── Video Steganography Routes ──────────────────────────


@app.route("/api/encrypt-video", methods=["POST"])
def encrypt_video():
    """Hide text within an uploaded video file."""
    if "video" not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400

    text = request.form.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    password = request.form.get("password", "").strip() or None

    try:
        from bis.stego.video import hide_text_in_video

        video_file = request.files["video"]
        uid = uuid.uuid4().hex[:8]
        original_ext = Path(video_file.filename).suffix or ".mp4"
        input_path = Path("uploads") / f"video_in_{uid}{original_ext}"
        output_path = Path("outputs") / f"stego_video_{uid}.avi"

        video_file.save(str(input_path))

        # Extract audio from input video (if any) for later preservation
        temp_audio = Path("uploads") / f"audio_preserve_{uid}.wav"
        has_audio = False
        if FFMPEG:
            try:
                r = subprocess.run(
                    [FFMPEG, "-y", "-i", str(input_path),
                     "-vn", "-c:a", "pcm_s16le", str(temp_audio)],
                    capture_output=True, timeout=30,
                )
                has_audio = (r.returncode == 0 and temp_audio.exists()
                             and temp_audio.stat().st_size > 44)
            except Exception:
                has_audio = False

        metadata = hide_text_in_video(
            str(input_path), text, str(output_path), password=password
        )

        # Clean up input
        input_path.unlink(missing_ok=True)

        actual_output = Path(metadata.get("output_path", str(output_path)))

        # Mux preserved audio back into the stego AVI
        if has_audio and FFMPEG:
            final = Path("outputs") / f"stego_av_{uid}.avi"
            try:
                r = subprocess.run(
                    [FFMPEG, "-y",
                     "-i", str(actual_output), "-i", str(temp_audio),
                     "-c:v", "copy", "-c:a", "pcm_s16le", str(final)],
                    capture_output=True, timeout=60,
                )
                if r.returncode == 0 and final.exists():
                    actual_output.unlink(missing_ok=True)
                    actual_output = final
            except Exception:
                pass
        temp_audio.unlink(missing_ok=True)

        return jsonify({
            "success": True,
            "video_url": f"/output/{actual_output.name}",
            "metadata": {
                "capacity": metadata["capacity"],
                "data_size": metadata["data_size"],
                "frames": metadata["frames"],
                "resolution": metadata["resolution"],
                "fps": metadata["fps"],
                "duration": metadata["duration_sec"],
            },
            "encrypted": password is not None,
            "message": f"Text hidden in video! ({metadata['data_size']} bytes in {metadata['frames']} frames)"
                       + (" (AES-256 encrypted)" if password else ""),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/decrypt-video", methods=["POST"])
def decrypt_video():
    """Extract hidden text from a stego video file."""
    if "video" not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400

    password = request.form.get("password", "").strip() or None

    try:
        from bis.stego.video import extract_text_from_video

        video_file = request.files["video"]
        uid = uuid.uuid4().hex[:8]
        original_ext = Path(video_file.filename).suffix or ".avi"
        input_path = Path("uploads") / f"video_dec_{uid}{original_ext}"
        video_file.save(str(input_path))

        text = extract_text_from_video(str(input_path), password=password)

        # Clean up
        input_path.unlink(missing_ok=True)

        return jsonify({
            "success": True,
            "text": text,
            "message": "Text extracted from video!",
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── Predefined Cover Templates ──────────────────────────


# Text templates - predefined cover text samples
TEXT_TEMPLATES = {
    "casual": {
        "name": "Casual Chat",
        "icon": "💬",
        "text": (
            "Hey! How's it going? I was just thinking about you the other day. "
            "We should totally catch up sometime this week. Maybe grab some coffee "
            "or go for a walk in the park? The weather has been amazing lately. "
            "Also, have you seen that new show everyone is talking about? I started "
            "watching it last night and it's actually pretty good. Let me know when "
            "you're free!"
        ),
    },
    "email": {
        "name": "Business Email",
        "icon": "📧",
        "text": (
            "Dear Team,\n\nI hope this email finds you well. I wanted to follow up "
            "on our last meeting regarding the upcoming project timeline. As discussed, "
            "we need to finalize the deliverables by the end of Q3. Please review the "
            "attached documents and provide your feedback by Friday.\n\nAdditionally, "
            "I would like to schedule a brief call next Tuesday to go over the budget "
            "allocations. Please let me know your availability.\n\nBest regards,\nProject Manager"
        ),
    },
    "news": {
        "name": "News Article",
        "icon": "📰",
        "text": (
            "Local Community Garden Celebrates 10th Anniversary\n\n"
            "The Greenfield Community Garden marked a decade of growth and "
            "togetherness this weekend with a special celebration attended by "
            "over 200 residents. Founded in 2014 by a group of dedicated "
            "volunteers, the garden has become a cornerstone of the neighborhood. "
            "Visitors enjoyed live music, fresh produce tastings, and guided tours "
            "of the expanded plots. Mayor Thompson praised the initiative, calling "
            "it a shining example of community spirit and environmental stewardship."
        ),
    },
    "social": {
        "name": "Social Post",
        "icon": "📱",
        "text": (
            "Just finished an incredible hike through the mountain trails today! "
            "The sunset views from the summit were absolutely breathtaking. Nothing "
            "beats spending time in nature to clear your mind and recharge. Already "
            "planning the next adventure for this weekend. Who's in? Drop a comment "
            "below if you want to join! #hiking #nature #adventure #outdoors #weekend"
        ),
    },
    "recipe": {
        "name": "Recipe Note",
        "icon": "🍳",
        "text": (
            "Grandma's Classic Chocolate Chip Cookies\n\n"
            "Ingredients: 2 cups flour, 1 cup butter, 3/4 cup sugar, 3/4 cup brown "
            "sugar, 2 eggs, 1 tsp vanilla, 1 tsp baking soda, 1/2 tsp salt, 2 cups "
            "chocolate chips.\n\nPreheat oven to 375°F. Cream butter and sugars until "
            "fluffy. Beat in eggs and vanilla. Mix in flour, baking soda, and salt. "
            "Fold in chocolate chips. Drop rounded tablespoons onto baking sheets. "
            "Bake 9-11 minutes until golden brown. Makes about 4 dozen cookies."
        ),
    },
    "journal": {
        "name": "Journal Entry",
        "icon": "📔",
        "text": (
            "Tuesday, October 15th\n\n"
            "Today was a surprisingly productive day. I managed to finish all my "
            "pending tasks before lunch, which gave me the entire afternoon to "
            "work on the creative project I've been putting off. Sometimes the "
            "best days are the ones without any big plans. I also took a long walk "
            "during my break and discovered a charming little bookstore just two "
            "blocks from the office. Picked up a novel that the owner recommended. "
            "Looking forward to reading it this weekend."
        ),
    },
    "love": {
        "name": "Love Letter",
        "icon": "💌",
        "text": (
            "My Dearest,\n\nI just wanted to take a moment to tell you how much "
            "you mean to me. Every day with you feels like a gift, and I find "
            "myself smiling at the smallest memories we share. Whether it's our "
            "morning coffee conversations or those late-night walks under the stars, "
            "every moment spent with you is one I treasure deeply. I know life gets "
            "busy, but please never forget how truly special you are.\n\n"
            "With all my love,\nForever Yours"
        ),
    },
    "study": {
        "name": "Study Notes",
        "icon": "📚",
        "text": (
            "Chapter 5 — Cellular Respiration\n\n"
            "Key points:\n"
            "1. Glycolysis occurs in the cytoplasm and produces 2 ATP per glucose molecule.\n"
            "2. The Krebs cycle takes place in the mitochondrial matrix and generates "
            "electron carriers NADH and FADH2.\n"
            "3. The electron transport chain (ETC) is located on the inner mitochondrial "
            "membrane and produces the majority of ATP (~34 molecules).\n"
            "4. Oxygen serves as the final electron acceptor in the ETC.\n\n"
            "Remember: total yield is approximately 36-38 ATP per glucose molecule. "
            "Review diagram on page 142 for the complete pathway."
        ),
    },
    "travel": {
        "name": "Travel Blog",
        "icon": "✈️",
        "text": (
            "Day 3 in Kyoto — Temple Hopping and Hidden Gardens\n\n"
            "Started the morning at Fushimi Inari Shrine, walking through thousands "
            "of vibrant orange torii gates winding up the mountainside. The early "
            "morning light filtering through the gates was absolutely magical. After "
            "a quick matcha break at a traditional tea house, we headed to the Arashiyama "
            "Bamboo Grove. The towering bamboo created an otherworldly tunnel of green. "
            "Lunch was the best bowl of ramen I've ever had at a tiny shop near Kinkaku-ji. "
            "Tomorrow we take the bullet train to Tokyo!"
        ),
    },
    "shopping": {
        "name": "Shopping List",
        "icon": "🛒",
        "text": (
            "Weekly Grocery List — Saturday\n\n"
            "Produce: bananas, avocados (3), spinach, cherry tomatoes, bell peppers, "
            "lemons, fresh basil, garlic\n"
            "Dairy: whole milk, Greek yogurt, cheddar cheese, butter\n"
            "Protein: chicken breast (2 lbs), salmon fillets, eggs (dozen)\n"
            "Pantry: olive oil, pasta, canned tomatoes, rice, black beans, "
            "peanut butter, honey\n"
            "Bakery: sourdough bread, bagels\n"
            "Frozen: frozen berries, ice cream\n"
            "Other: paper towels, dish soap, coffee beans\n\n"
            "Budget: ~$85. Don't forget the reusable bags!"
        ),
    },
}

@app.route("/api/templates/text/<template_id>")
def get_text_template(template_id: str):
    """Return the full text of a specific template."""
    t = TEXT_TEMPLATES.get(template_id)
    if not t:
        return jsonify({"error": "Template not found"}), 404
    return jsonify({"text": t["text"], "name": t["name"]})


@app.route("/api/templates/audio/<template_id>")
def get_audio_template(template_id: str):
    """Generate and serve a realistic WAV audio cover file.

    Templates: guitar, lullaby, jazz, ocean, rain, melody, chimes, birds
    Accepts optional ?duration=SECONDS (clamped to 60-900 = 1-15 min).
    """
    sr = 44100
    default_dur = _AUDIO_DURATIONS.get(template_id, 120)
    try:
        dur = int(request.args.get("duration", default_dur))
    except (ValueError, TypeError):
        dur = default_dur
    dur = max(15, min(900, dur))  # clamp 15s-15 minutes

    samples = _gen_audio(template_id, sr, dur=dur)
    if samples is None:
        return jsonify({"error": "Audio template not found"}), 404

    buf = _write_wav_buf(samples, sr)
    return send_file(
        buf,
        mimetype="audio/wav",
        as_attachment=True,
        download_name=f"cover_{template_id}.wav",
    )


@app.route("/api/templates/video/<template_id>")
def get_video_template(template_id: str):
    """Generate and serve a video cover file with matching audio track.

    Templates: gradient, waves, sky, abstract, starfield, fireflies, sunrise, matrix
    Accepts optional ?duration=SECONDS (clamped to 60-900 = 1-15 min).
    """
    width, height, fps = 320, 240, 24
    sr = 44100
    default_dur = _VIDEO_DURATIONS.get(template_id, 60)
    try:
        duration = int(request.args.get("duration", default_dur))
    except (ValueError, TypeError):
        duration = default_dur
    duration = max(15, min(900, duration))  # clamp 15s-15 minutes
    total_frames = fps * duration

    # Validate template
    if template_id not in _VIDEO_AUDIO_MAP:
        return jsonify({"error": "Video template not found"}), 404

    uid = uuid.uuid4().hex[:8]
    video_path = Path("outputs") / f"tpl_v_{uid}.avi"
    final_path = Path("outputs") / f"tpl_final_{uid}.avi"

    fourcc = cv2.VideoWriter_fourcc(*"FFV1")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    try:
        for fi in range(total_frames):
            frame = _gen_frame(template_id, fi, total_frames, width, height)
            if frame is None:
                writer.release()
                video_path.unlink(missing_ok=True)
                return jsonify({"error": "Video template not found"}), 404
            writer.write(frame)

        writer.release()

        # Generate matching audio (same duration) and mux into the video
        audio_tid = _VIDEO_AUDIO_MAP.get(template_id)
        audio = _gen_audio(audio_tid, sr, dur=duration) if audio_tid else None

        if audio is not None and _mux_av(video_path, audio, sr, final_path):
            video_path.unlink(missing_ok=True)
            return send_file(
                str(final_path),
                mimetype="video/x-msvideo",
                as_attachment=True,
                download_name=f"cover_{template_id}.avi",
            )

        # Fallback: serve video-only AVI
        return send_file(
            str(video_path),
            mimetype="video/x-msvideo",
            as_attachment=True,
            download_name=f"cover_{template_id}.avi",
        )

    except Exception as e:
        writer.release()
        video_path.unlink(missing_ok=True)
        final_path.unlink(missing_ok=True)
        return jsonify({"error": str(e)}), 500


# ─── Image Steganography Routes ──────────────────────────


@app.route("/api/encrypt-image", methods=["POST"])
def encrypt_image():
    """Hide secret text within an image.

    Uses LSB image steganography for deterministic local behavior.
    """
    secret_text = request.form.get("secret_text", "").strip()
    password = request.form.get("password", "").strip() or None
    cover_file = request.files.get("cover_image")

    if not secret_text:
        return jsonify({"error": "No secret text provided"}), 400
    if not cover_file:
        return jsonify({"error": "No cover image provided"}), 400

    uid = uuid.uuid4().hex[:8]
    cover_path = Path("uploads") / f"cover_{uid}.png"
    out_path = Path("outputs") / f"stego_{uid}.png"

    try:
        cover_file.save(str(cover_path))
        from bis.stego.image_lsb import hide_text_in_image

        result = hide_text_in_image(
            str(cover_path), secret_text, str(out_path),
            password=password,
        )
        metadata = {
            "method": "LSB",
            "capacity": result["capacity"],
            "data_size": result["data_size"],
        }

        cover_path.unlink(missing_ok=True)

        return jsonify({
            "success": True,
            "stego_url": f"/output/{out_path.name}",
            "stego_file": out_path.name,
            "encrypted": password is not None,
            "metadata": metadata,
            "message": f"Text hidden in image ({metadata.get('method', 'LSB')})!"
                       + (" (AES-256 encrypted)" if password else ""),
        })
    except Exception as e:
        cover_path.unlink(missing_ok=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/decrypt-image", methods=["POST"])
def decrypt_image():
    """Extract hidden text from a stego image."""
    stego_file = request.files.get("stego_image")
    password = request.form.get("password", "").strip() or None

    if not stego_file:
        return jsonify({"error": "No stego image provided"}), 400

    uid = uuid.uuid4().hex[:8]
    stego_path = Path("uploads") / f"stego_{uid}.png"

    try:
        stego_file.save(str(stego_path))

        from bis.stego.image_lsb import extract_text_from_image
        text = extract_text_from_image(str(stego_path), password=password)

        stego_path.unlink(missing_ok=True)

        return jsonify({
            "success": True,
            "text": text,
            "extracted_text": text,  # backward compat
            "encrypted": password is not None,
        })
    except Exception as e:
        stego_path.unlink(missing_ok=True)
        return jsonify({"error": str(e)}), 500


# ─── Image Template Routes ───────────────────────────────


@app.route("/api/templates/image/<template_id>")
def get_image_template(template_id: str):
    """Serve a predefined cover image template."""
    templates_dir = Path("static/templates/images")
    safe_name = "".join(c for c in template_id if c.isalnum() or c in "-_")
    img_path = templates_dir / f"{safe_name}.png"
    if not img_path.exists():
        return jsonify({"error": f"Template '{template_id}' not found"}), 404
    return send_file(str(img_path), mimetype="image/png")


# ─── File Serving ────────────────────────────────────────


@app.route("/output/<filename>")
def serve_output(filename: str):
    """Serve output files (audio, video)."""
    filepath = Path("outputs") / filename
    ext = filepath.suffix.lower()

    mime_types = {
        ".wav": "audio/wav",
        ".wave": "audio/wav",
        ".avi": "video/x-msvideo",
        ".mp4": "video/mp4",
        ".mkv": "video/x-matroska",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }
    mimetype = mime_types.get(ext, "application/octet-stream")

    return send_file(filepath, mimetype=mimetype)


@app.route("/api/download/<filename>")
def download_file(filename: str):
    """Download an output file."""
    return send_file(
        Path("outputs") / filename,
        as_attachment=True,
        download_name=filename,
    )


# ─── Fine-Tuning Routes ──────────────────────────────────

try:
    from bis.fine_tuning.api import register_fine_tune_routes
    register_fine_tune_routes(app)
except ImportError:
    pass  # fine-tuning module not installed


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
