# BIS Stagno

Multi-media steganography web app built with Flask.

The app hides secret text inside text, audio, video, and image covers, with optional password-based encryption.

## What Is Included

- Text in text steganography (zero-width Unicode)
- Text in audio steganography (WAV LSB)
- Text in video steganography (frame-based LSB)
- Text in image steganography (LSB)
- Built-in cover templates for text, audio, video, and image
- Optional AES-GCM encryption for payload protection
- Optional fine-tuning dashboard and APIs

## Project Layout

- `flask_app.py`: Flask app and HTTP routes
- `bis/stego/`: steganography implementations
- `bis/fine_tuning/`: fine-tuning SDK, trainers, and routes
- `bis/generation/image_gen/`: image generation helpers used by advanced workflows
- `templates/`: HTML templates
- `static/`: CSS and JavaScript
- `uploads/`: runtime input folder (ignored by git)
- `outputs/`: runtime output folder (ignored by git)

## Prerequisites

- Python 3.10+
- pip
- Optional: ffmpeg (for richer video/audio processing)

## Local Setup

1. Create and activate a virtual environment.

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

4. Open:

- http://127.0.0.1:5000

## Main Routes

### Pages

- `GET /`
- `GET /encrypt`
- `GET /decrypt`
- `GET /about`

### Core Stego APIs

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

### Output Access

- `GET /output/<filename>`
- `GET /api/download/<filename>`

### Fine-Tuning (Optional)

Fine-tuning endpoints are registered when dependencies are available in `bis/fine_tuning/`.

Examples:

- `POST /api/fine-tune/<modality>`
- `GET /api/fine-tune/status/<job_id>`
- `GET /fine-tune/dashboard`

## Quality Checkpoints (Before Push)

Run these checks before pushing to GitHub:

```powershell
# Python syntax checks
python -m py_compile flask_app.py

# Optional: compile all Python files
Get-ChildItem -Recurse -Filter *.py | ForEach-Object { python -m py_compile $_.FullName }

# Frontend syntax checks
node --check static/encrypt.js
node --check static/decrypt.js
node --check static/home.js
node --check static/shared.js
node --check static/animations.js
```

Optional smoke check with Flask test client:

```powershell
python -c "from flask_app import app; c=app.test_client(); assert c.get('/').status_code==200; assert c.get('/encrypt').status_code==200; assert c.get('/decrypt').status_code==200; assert c.get('/about').status_code==200; print('smoke-ok')"
```

## GitHub Push Steps

If this folder is not yet a git repo:

```powershell
git init
git add .
git commit -m "Initial clean commit"
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```

If git is already initialized:

```powershell
git add .
git commit -m "Cleanup legacy generation code, docs, and repo hygiene"
git push
```

## Notes

- Runtime artifacts in `uploads/` and `outputs/` are ignored via `.gitignore`.
- Legacy, unused generation-path modules were removed from the app runtime flow.
- The active packaging script entry point is `bis.fine_tuning.cli:main`.
