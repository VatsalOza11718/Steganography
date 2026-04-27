# Steganography (BIS Stagno)

Secure multi-media steganography web application built with Flask.

This project hides secret text inside text, audio, video, and images, with optional password-based AES-GCM encryption.

## Demo Output

Project output preview:

![Project Output Preview](docs/images/video-preview.png)

Full output video (MP4):

- [Watch Video.mp4](Video.mp4)

## Feature Highlights

- Text in Text using zero-width Unicode embedding
- Text in Audio using WAV least-significant-bit embedding
- Text in Video using frame-based embedding
- Text in Image using LSB embedding
- Built-in cover templates for text, audio, video, and image workflows
- Optional AES-GCM encryption for hidden payload protection
- Fine-tuning dashboard and APIs for advanced workflows

## Visual Gallery

Template previews used by the app:

| Nature | Ocean | City Night |
|---|---|---|
| ![Nature Template](static/templates/images/nature.png) | ![Ocean Template](static/templates/images/ocean.png) | ![City Night Template](static/templates/images/citynight.png) |

| Forest | Sunset | Abstract |
|---|---|---|
| ![Forest Template](static/templates/images/forest.png) | ![Sunset Template](static/templates/images/sunset.png) | ![Abstract Template](static/templates/images/abstract.png) |

## Tech Stack

- Backend: Flask, NumPy, OpenCV
- Security: AES-GCM via PyCryptodome
- Frontend: HTML, CSS, JavaScript
- Media: WAV and video processing with optional ffmpeg support

## Project Structure

- `flask_app.py` - Flask app and route definitions
- `bis/stego/` - Core steganography implementations
- `bis/fine_tuning/` - Fine-tuning SDK, trainers, dashboard routes
- `bis/generation/image_gen/` - Advanced image generation helper module
- `templates/` - Jinja templates
- `static/` - Frontend scripts and styles
- `uploads/` - Runtime upload directory (git ignored)
- `outputs/` - Runtime output directory (git ignored)

## Quick Start

1. Create and activate virtual environment.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Install dependencies.

```powershell
pip install -e .
```

3. Run the app.

```powershell
python flask_app.py
```

4. Open in browser.

- http://127.0.0.1:5000

## Main Routes

### Pages

- `GET /`
- `GET /encrypt`
- `GET /decrypt`
- `GET /about`

### Steganography APIs

- `POST /api/encrypt-text`
- `POST /api/decrypt-text`
- `POST /api/encrypt-audio`
- `POST /api/decrypt-audio`
- `POST /api/encrypt-video`
- `POST /api/decrypt-video`
- `POST /api/encrypt-image`
- `POST /api/decrypt-image`

### Cover Template APIs

- `GET /api/templates/text/<template_id>`
- `GET /api/templates/audio/<template_id>`
- `GET /api/templates/video/<template_id>`
- `GET /api/templates/image/<template_id>`

### File Access

- `GET /output/<filename>`
- `GET /api/download/<filename>`

### Fine-Tuning APIs (optional)

- `POST /api/fine-tune/<modality>`
- `GET /api/fine-tune/status/<job_id>`
- `GET /fine-tune/dashboard`

## Quality Checkpoints

Run these before release/push:

```powershell
python -m py_compile flask_app.py
Get-ChildItem -Recurse -Filter *.py | ForEach-Object { python -m py_compile $_.FullName }

node --check static/encrypt.js
node --check static/decrypt.js
node --check static/home.js
node --check static/shared.js
node --check static/animations.js
```

Smoke test:

```powershell
python -c "from flask_app import app; c=app.test_client(); assert c.get('/').status_code==200; assert c.get('/encrypt').status_code==200; assert c.get('/decrypt').status_code==200; assert c.get('/about').status_code==200; print('smoke-ok')"
```

## Repository Notes

- Runtime artifacts are excluded using [.gitignore](.gitignore)
- Legacy generation runtime paths and old unused generation modules were removed from the active app flow
- Current CLI entry point in packaging: `bis.fine_tuning.cli:main`
