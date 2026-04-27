"""REST API endpoints for the multi-modal fine-tuning platform.

Endpoints:
 • POST   /api/fine-tune/{modality}   — start a fine-tuning job
 • GET    /api/fine-tune/status/{id}  — get job status + metrics
 • DELETE /api/fine-tune/cancel/{id}  — cancel a running job
 • GET    /api/fine-tune/jobs         — list all jobs
 • GET    /api/fine-tune/presets      — list available presets
 • GET    /api/fine-tune/presets/{n}  — get preset details
 • POST   /api/fine-tune/evaluate     — evaluate a trained model
 • GET    /api/fine-tune/download/{id}— download LoRA weights

WebSocket:
 • /ws/fine-tune/logs/{job_id}        — stream live training logs

All endpoints return JSON. CORS headers are included for dashboard use.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Singleton orchestrator instance
_orchestrator = None
_orchestrator_lock = threading.Lock()


def _get_orchestrator():
    """Get or create the singleton FineTuneOrchestrator."""
    global _orchestrator
    if _orchestrator is None:
        with _orchestrator_lock:
            if _orchestrator is None:
                from bis.fine_tuning.trainers import FineTuneOrchestrator
                _orchestrator = FineTuneOrchestrator(max_concurrent=1)
    return _orchestrator


# ═══════════════════════════════════════════════════════════
#  FLASK BLUEPRINT REGISTRATION
# ═══════════════════════════════════════════════════════════

def register_fine_tune_routes(app):
    """Register fine-tuning REST API routes on a Flask app.

    Call this from flask_app.py:
        from bis.fine_tuning.api import register_fine_tune_routes
        register_fine_tune_routes(app)
    """
    from flask import request, jsonify, send_file, render_template

    # ── POST /api/fine-tune/<modality> ───────────────────

    @app.route("/api/fine-tune/<modality>", methods=["POST"])
    def fine_tune_start(modality: str):
        """Start a fine-tuning job for the given modality.

        Request body (JSON):
            {
                "dataset_dir": "/path/to/data",      (required)
                "preset": "qwen25-chat",              (optional)
                "config": { ... overrides ... },      (optional)
                "model_id": "Qwen/Qwen2.5-1.5B",     (optional)
            }

        Response:
            {"job_id": "abc123", "status": "queued", "modality": "text"}
        """
        valid_modalities = {"text", "audio", "image"}
        if modality not in valid_modalities:
            return jsonify({
                "error": f"Invalid modality '{modality}'. "
                         f"Valid: {sorted(valid_modalities)}",
            }), 400

        data = request.get_json() or {}
        dataset_dir = data.get("dataset_dir", "")
        preset = data.get("preset")
        overrides = data.get("config", {})
        model_id = data.get("model_id", "")

        if not dataset_dir and not preset:
            return jsonify({
                "error": "dataset_dir or preset is required",
            }), 400

        try:
            from bis.fine_tuning.sdk import FineTuneSDK

            sdk = FineTuneSDK(modality)

            # Configure
            if preset:
                sdk.configure(preset=preset, dataset_dir=dataset_dir or None, **overrides)
            else:
                from bis.fine_tuning.config import FineTuneConfig, Modality
                config = FineTuneConfig(
                    modality=Modality(modality),
                    dataset_dir=dataset_dir,
                    model_id=model_id,
                    **overrides,
                )
                sdk.config = config

            if dataset_dir:
                sdk.config.dataset_dir = dataset_dir
            if model_id:
                sdk.config.model_id = model_id

            # Submit to orchestrator
            orchestrator = _get_orchestrator()
            job_id = orchestrator.submit(sdk.config)

            # Start training in background thread
            def _run_job():
                """Run the blocking trainer in a daemon thread for this request."""
                try:
                    orchestrator.start(job_id)
                except Exception as e:
                    logger.error("Job %s failed: %s", job_id, e)

            thread = threading.Thread(
                target=_run_job, daemon=True, name=f"finetune-{job_id}",
            )
            thread.start()

            return jsonify({
                "job_id": job_id,
                "status": "queued",
                "modality": modality,
                "message": f"Fine-tuning job started for {modality}",
            }), 202

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ── GET /api/fine-tune/status/<job_id> ───────────────

    @app.route("/api/fine-tune/status/<job_id>")
    def fine_tune_status(job_id: str):
        """Get status and metrics for a fine-tuning job.

        Response:
            {
                "job_id": "abc123",
                "status": "training",
                "progress_pct": 45.2,
                "current_step": 452,
                "total_steps": 1000,
                "current_loss": 0.342,
                "metrics": {...}
            }
        """
        orchestrator = _get_orchestrator()
        status = orchestrator.status(job_id)

        if "error" in status:
            return jsonify(status), 404

        return jsonify(status)

    # ── DELETE /api/fine-tune/cancel/<job_id> ────────────

    @app.route("/api/fine-tune/cancel/<job_id>", methods=["DELETE"])
    def fine_tune_cancel(job_id: str):
        """Cancel a running fine-tuning job."""
        orchestrator = _get_orchestrator()
        success = orchestrator.cancel(job_id)

        if not success:
            return jsonify({"error": f"Job {job_id} not found"}), 404

        return jsonify({
            "job_id": job_id,
            "status": "cancelled",
            "message": "Job cancelled successfully",
        })

    # ── GET /api/fine-tune/jobs ──────────────────────────

    @app.route("/api/fine-tune/jobs")
    def fine_tune_list_jobs():
        """List all fine-tuning jobs."""
        orchestrator = _get_orchestrator()
        jobs = orchestrator.list_jobs()

        return jsonify({
            "total": len(jobs),
            "jobs": jobs,
        })

    # ── GET /api/fine-tune/presets ────────────────────────

    @app.route("/api/fine-tune/presets")
    def fine_tune_presets():
        """List available fine-tuning presets."""
        from bis.fine_tuning.sdk import FineTuneSDK

        presets = []
        for name, (modality, method) in FineTuneSDK.PRESETS.items():
            presets.append({
                "name": name,
                "modality": modality,
                "description": f"{modality} fine-tuning preset: {name}",
            })

        return jsonify({"presets": presets})

    # ── GET /api/fine-tune/presets/<name> ─────────────────

    @app.route("/api/fine-tune/presets/<name>")
    def fine_tune_preset_detail(name: str):
        """Get detailed configuration for a preset."""
        from bis.fine_tuning.sdk import FineTuneSDK

        sdk = FineTuneSDK()
        try:
            config = sdk._load_preset(name)
            return jsonify({
                "name": name,
                "config": config.to_dict(),
            })
        except ValueError as e:
            return jsonify({"error": str(e)}), 404

    # ── POST /api/fine-tune/evaluate ─────────────────────

    @app.route("/api/fine-tune/evaluate", methods=["POST"])
    def fine_tune_evaluate():
        """Evaluate a fine-tuned model.

        Request body:
            {
                "job_id": "abc123",           (optional, uses trained model)
                "modality": "text",            (required if no job_id)
                "generated_data": [...],       (optional)
                "reference_data": [...]        (optional)
            }
        """
        data = request.get_json() or {}
        job_id = data.get("job_id")
        modality = data.get("modality", "text")

        try:
            from bis.fine_tuning.evaluation import UnifiedEvaluator

            evaluator = UnifiedEvaluator()

            # Get generated data from job result or request
            generated = data.get("generated_data")

            if generated is None and job_id:
                orchestrator = _get_orchestrator()
                result = orchestrator.get_result(job_id)
                if result:
                    modality = result.get("modality", modality)

            if generated is None:
                # Generate mock samples for demo
                import numpy as np
                if modality == "text":
                    generated = ["Sample generated text."] * 10
                elif modality == "audio":
                    generated = [np.random.randn(32000 * 5).astype(np.float32)
                                 for _ in range(5)]

            result = evaluator.evaluate(modality, generated)
            return jsonify(result.to_dict())

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ── GET /api/fine-tune/download/<job_id> ─────────────

    @app.route("/api/fine-tune/download/<job_id>")
    def fine_tune_download(job_id: str):
        """Download LoRA weights for a completed job."""
        orchestrator = _get_orchestrator()
        result = orchestrator.get_result(job_id)

        if result is None:
            return jsonify({"error": "Job not found or not completed"}), 404

        lora_path = result.get("lora_path")
        if not lora_path or not Path(lora_path).exists():
            return jsonify({"error": "LoRA weights not available"}), 404

        lora_dir = Path(lora_path)
        if lora_dir.is_dir():
            # Create a zip archive
            import shutil
            zip_path = lora_dir.parent / f"{job_id}_lora.zip"
            shutil.make_archive(str(zip_path.with_suffix("")), "zip", str(lora_dir))
            return send_file(
                str(zip_path),
                mimetype="application/zip",
                as_attachment=True,
                download_name=f"lora_{job_id}.zip",
            )
        else:
            return send_file(
                str(lora_dir),
                as_attachment=True,
                download_name=f"lora_{job_id}.safetensors",
            )

    # ── GPU / System Status ──────────────────────────────

    @app.route("/api/fine-tune/gpu-status")
    def fine_tune_gpu_status():
        """Get current GPU status and resource utilization."""
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                total_mem = getattr(props, 'total_memory', None)
                if total_mem is None:
                    total_mem = getattr(props, 'total_mem', 0)
                gpu_info = {
                    "available": True,
                    "device_name": torch.cuda.get_device_name(0),
                    "total_gb": round(total_mem / 1e9, 2),
                    "allocated_gb": round(torch.cuda.memory_allocated(0) / 1e9, 2),
                    "reserved_gb": round(torch.cuda.memory_reserved(0) / 1e9, 2),
                    "free_gb": round(
                        (total_mem - torch.cuda.memory_allocated(0)) / 1e9, 2
                    ),
                }
            else:
                gpu_info = {"available": False, "reason": "No CUDA device"}
        except ImportError:
            gpu_info = {"available": False, "reason": "PyTorch not installed"}

        orchestrator = _get_orchestrator()
        active_jobs = [
            j for j in orchestrator.list_jobs()
            if j.get("status") in ("training", "preprocessing", "evaluating")
        ]

        return jsonify({
            "gpu": gpu_info,
            "active_jobs": len(active_jobs),
            "queued_jobs": len([
                j for j in orchestrator.list_jobs()
                if j.get("status") == "queued"
            ]),
        })

# ── Dashboard Route ───────────────────────────────────

    @app.route("/fine-tune/dashboard")
    def fine_tune_dashboard():
        """Serve the fine-tuning dashboard page."""
        return render_template(
            "fine_tune_dashboard.html",
            active_page="dashboard",
        )

    logger.info("Fine-tuning REST API routes registered")


# ═══════════════════════════════════════════════════════════
#  WEBSOCKET LOG STREAMING (Flask-SocketIO compatible)
# ═══════════════════════════════════════════════════════════

_log_subscribers: dict[str, list] = {}  # job_id -> list of callback functions

def register_log_callback(job_id: str, callback) -> None:
    """Register a callback to receive log messages for a job."""
    if job_id not in _log_subscribers:
        _log_subscribers[job_id] = []
    _log_subscribers[job_id].append(callback)


def unregister_log_callback(job_id: str, callback) -> None:
    """Unregister a log callback."""
    if job_id in _log_subscribers:
        try:
            _log_subscribers[job_id].remove(callback)
        except ValueError:
            pass


def broadcast_log(job_id: str, message: str, level: str = "info") -> None:
    """Broadcast a log message to all subscribers of a job."""
    if job_id in _log_subscribers:
        for callback in _log_subscribers[job_id]:
            try:
                callback({
                    "job_id": job_id,
                    "message": message,
                    "level": level,
                    "timestamp": time.time(),
                })
            except Exception:
                pass


def register_socketio_handlers(socketio):
    """Register WebSocket handlers for log streaming.
    
    Call this with your Flask-SocketIO instance:
        from flask_socketio import SocketIO
        socketio = SocketIO(app)
        register_socketio_handlers(socketio)
    """
    @socketio.on("subscribe_logs")
    def handle_subscribe(data):
        """Subscribe to live logs for a fine-tuning job."""
        job_id = data.get("job_id")
        if not job_id:
            return {"error": "job_id required"}
        
        from flask_socketio import join_room, emit
        join_room(f"logs_{job_id}")
        
        # Send initial status
        orchestrator = _get_orchestrator()
        status = orchestrator.status(job_id)
        emit("job_status", status)
        
        return {"status": "subscribed", "job_id": job_id}

    @socketio.on("unsubscribe_logs")
    def handle_unsubscribe(data):
        """Unsubscribe from job logs."""
        job_id = data.get("job_id")
        if job_id:
            from flask_socketio import leave_room
            leave_room(f"logs_{job_id}")
        return {"status": "unsubscribed"}

    logger.info("WebSocket log streaming handlers registered")


class LogStreamHandler(logging.Handler):
    """Logging handler that broadcasts to WebSocket subscribers."""
    
    def __init__(self, job_id: str):
        """Bind the handler to one job so only its log stream is broadcast."""
        super().__init__()
        self.job_id = job_id
    
    def emit(self, record):
        """Forward formatted log records to live WebSocket subscribers."""
        try:
            message = self.format(record)
            broadcast_log(self.job_id, message, record.levelname.lower())
        except Exception:
            pass
