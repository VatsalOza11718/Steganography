"""CLI interface for multi-modal fine-tuning.

Usage:
    python -m bis.fine_tuning.cli text  --preset qwen25-chat   --data ./data
    python -m bis.fine_tuning.cli audio --preset musicgen-melody --data ./data

    python -m bis.fine_tuning.cli presets            # list presets
    python -m bis.fine_tuning.cli evaluate text      # evaluate a model
    python -m bis.fine_tuning.cli status <job_id>    # check job status
"""

from __future__ import annotations

import argparse
import json
import sys
import time


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and dispatch to the requested subcommand."""
    parser = argparse.ArgumentParser(
        prog="modal-tune",
        description="BIS Multi-Modal Fine-Tuning Platform",
        epilog="Example: modal-tune text --preset qwen25-chat --data ./data",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # ── Train commands (text, audio, image) ────────────

    for modality in ("text", "audio", "image"):
        sub = subparsers.add_parser(
            modality, help=f"Fine-tune a {modality} model",
        )
        sub.add_argument(
            "--preset", type=str, default=None,
            help="Named preset (e.g. qwen25-chat, musicgen-melody)",
        )
        sub.add_argument(
            "--data", "--dataset-dir", type=str, default="",
            dest="dataset_dir",
            help="Path to training dataset directory",
        )
        sub.add_argument(
            "--model-id", type=str, default="",
            help="HuggingFace model ID to fine-tune",
        )
        sub.add_argument(
            "--config", type=str, default=None,
            help="Path to YAML/JSON config file",
        )
        sub.add_argument(
            "--output", type=str, default="./fine_tune_output",
            help="Output directory for LoRA weights",
        )
        sub.add_argument(
            "--epochs", type=int, default=None,
            help="Number of training epochs",
        )
        sub.add_argument(
            "--lr", "--learning-rate", type=float, default=None,
            dest="learning_rate",
            help="Learning rate",
        )
        sub.add_argument(
            "--batch-size", type=int, default=None,
            help="Batch size",
        )
        sub.add_argument(
            "--lora-rank", type=int, default=None,
            help="LoRA rank",
        )
        sub.add_argument(
            "--evaluate", action="store_true",
            help="Run evaluation after training",
        )
        sub.add_argument(
            "--export", type=str, default=None,
            help="Export LoRA weights to this directory",
        )

    # ── Presets command ───────────────────────────────────

    sub_presets = subparsers.add_parser("presets", help="List available presets")
    sub_presets.add_argument(
        "--detail", type=str, default=None,
        help="Show detailed config for a specific preset",
    )

    # ── Status command ───────────────────────────────────

    sub_status = subparsers.add_parser("status", help="Check job status")
    sub_status.add_argument("job_id", type=str, help="Job ID to check")

    # ── Evaluate command ─────────────────────────────────

    sub_eval = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    sub_eval.add_argument("modality", type=str, help="Modality to evaluate")
    sub_eval.add_argument(
        "--data", type=str, default=None,
        help="Path to evaluation data",
    )

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    # ── Dispatch ─────────────────────────────────────────

    if args.command in ("text", "audio", "image"):
        return _cmd_train(args)
    elif args.command == "presets":
        return _cmd_presets(args)
    elif args.command == "status":
        return _cmd_status(args)
    elif args.command == "evaluate":
        return _cmd_evaluate(args)

    parser.print_help()
    return 1


# ─── Command Implementations ────────────────────────────────

def _cmd_train(args: argparse.Namespace) -> int:
    """Execute a fine-tuning job."""
    from bis.fine_tuning.sdk import FineTuneSDK

    modality = args.command
    sdk = FineTuneSDK(modality)

    print(f"\n{'═' * 60}")
    print(f"  BIS FINE-TUNING — {modality.upper()}")
    print(f"{'═' * 60}")

    # Configure
    overrides = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.learning_rate is not None:
        overrides["learning_rate"] = args.learning_rate
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size

    try:
        if args.preset:
            config_summary = sdk.configure(
                preset=args.preset,
                dataset_dir=args.dataset_dir or None,
                **overrides,
            )
        elif args.config:
            config_summary = sdk.configure(
                config_path=args.config,
                dataset_dir=args.dataset_dir or None,
                **overrides,
            )
        else:
            config_summary = sdk.configure(
                dataset_dir=args.dataset_dir or None,
                **overrides,
            )

        if args.model_id:
            sdk.config.model_id = args.model_id
        if args.output:
            sdk.config.checkpoint.output_dir = args.output
        if args.lora_rank is not None:
            sdk.config.lora.rank = args.lora_rank

        print(f"\n  Model:    {sdk.config.model_id}")
        print(f"  Preset:   {args.preset or 'custom'}")
        print(f"  Dataset:  {sdk.config.dataset_dir}")
        print(f"  Epochs:   {sdk.config.epochs}")
        print(f"  LR:       {sdk.config.learning_rate}")
        print(f"  LoRA r:   {sdk.config.lora.rank}")
        print(f"  Precision: {sdk.config.precision.value}")
        print(f"  Output:   {sdk.config.checkpoint.output_dir}")

        # Validate
        issues = config_summary.get("issues", [])
        if issues:
            print(f"\n  ⚠ Config issues: {', '.join(issues)}")
            if not sdk.config.model_id or not sdk.config.dataset_dir:
                print("  ✗ Cannot proceed without model_id and dataset_dir")
                return 1

    except Exception as e:
        print(f"\n  ✗ Configuration error: {e}")
        return 1

    # Train
    print(f"\n{'─' * 60}")
    print("  Starting training...")
    print(f"{'─' * 60}\n")

    start = time.time()
    try:
        result = sdk.train()
    except Exception as e:
        print(f"\n  ✗ Training failed: {e}")
        return 1

    elapsed = time.time() - start
    status = result.get("status", "unknown")
    final_loss = result.get("final_loss", float("inf"))

    print(f"\n{'─' * 60}")
    if status == "completed":
        print(f"  ✓ Training completed in {elapsed:.1f}s")
        print(f"  ✓ Final loss: {final_loss:.4f}")
        if result.get("lora_path"):
            print(f"  ✓ LoRA saved: {result['lora_path']}")
    else:
        print(f"  ✗ Training {status}: {result.get('error', '')}")
    print(f"{'─' * 60}")

    # Evaluate
    if args.evaluate:
        print("\n  Running evaluation...")
        try:
            eval_result = sdk.evaluate()
            print(f"  Evaluation: {'PASSED ✓' if eval_result.get('passed') else 'FAILED ✗'}")
            for k, v in eval_result.get("metrics", {}).items():
                print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
        except Exception as e:
            print(f"  ✗ Evaluation failed: {e}")

    # Export
    if args.export:
        print(f"\n  Exporting LoRA to {args.export}...")
        try:
            export_result = sdk.export_lora(args.export)
            print(f"  ✓ Exported: {export_result}")
        except Exception as e:
            print(f"  ✗ Export failed: {e}")

    print()
    return 0 if status == "completed" else 1


def _cmd_presets(args: argparse.Namespace) -> int:
    """List or show preset details."""
    from bis.fine_tuning.sdk import FineTuneSDK

    if args.detail:
        sdk = FineTuneSDK()
        try:
            config = sdk._load_preset(args.detail)
            print(f"\nPreset: {args.detail}")
            print(json.dumps(config.to_dict(), indent=2, default=str))
        except ValueError as e:
            print(f"Error: {e}")
            return 1
    else:
        print(f"\n{'═' * 60}")
        print("  AVAILABLE FINE-TUNING PRESETS")
        print(f"{'═' * 60}\n")

        for name in FineTuneSDK.available_presets():
            modality = FineTuneSDK.PRESETS[name][0]
            print(f"  {name:<25s} ({modality})")

        print(f"\n  Usage: modal-tune text --preset qwen25-chat --data ./data")
        print(f"         modal-tune --detail qwen25-chat")
        print()

    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    """Check job status."""
    from bis.fine_tuning.trainers import FineTuneOrchestrator

    # This would typically connect to a running server
    print(f"\n  Job ID: {args.job_id}")
    print("  Note: Status checking requires a running fine-tune server.")
    print("  Use the REST API: GET /api/fine-tune/status/{job_id}")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    """Evaluate a model."""
    from bis.fine_tuning.sdk import FineTuneSDK

    sdk = FineTuneSDK(args.modality)

    print(f"\n  Evaluating {args.modality} model...")
    try:
        result = sdk.evaluate()
        print(f"\n  Status: {'PASSED ✓' if result.get('passed') else 'FAILED ✗'}")
        print(f"  Samples: {result.get('samples_evaluated', 0)}")
        print("  Metrics:")
        for k, v in result.get("metrics", {}).items():
            print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
        return 0
    except Exception as e:
        print(f"  ✗ Evaluation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
