"""Fine-tuning pipeline for image generation models.

Supports:
 • LoRA fine-tuning (low-rank adaptation, ~4-8 MB)
 • Textual inversion (custom embeddings, ~6 KB)
 • DreamBooth (full model, ~4 GB)

Each method is independently configurable per model backend.

Usage::

    from bis.generation.image_gen.fine_tuning import (
        LoRAConfig, LoRATrainer,
        TextualInversionConfig, TextualInversionTrainer,
    )

    # LoRA fine-tuning
    config = LoRAConfig(
        model_id="sd15",
        dataset_dir="path/to/images",
        output_dir="./lora_output",
        prompt="a painting of [concept]",
    )
    trainer = LoRATrainer(config)
    trainer.train()
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  DATASET PREPARATION
# ═══════════════════════════════════════════════════════════

@dataclass
class FineTuneDatasetConfig:
    """Configuration for fine-tuning dataset preparation."""
    image_dir: str
    prompt: str = ""                    # constant prompt or template
    prompt_file: Optional[str] = None   # text file with per-image prompts
    resolution: int = 512
    center_crop: bool = True
    random_flip: bool = True
    num_repeats: int = 1                # repeat dataset N times per epoch


class FineTuneDataset(Dataset):
    """Simple image dataset for fine-tuning.

    Loads images from a directory, optionally pairs them with
    per-image prompts from a text file.
    """

    SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

    def __init__(self, config: FineTuneDatasetConfig) -> None:
        """Discover training images and align them with prompt strings."""
        self.config = config
        self.image_dir = Path(config.image_dir)

        # Discover images
        self.image_paths = sorted([
            p for p in self.image_dir.iterdir()
            if p.suffix.lower() in self.SUPPORTED_EXT
        ])
        if not self.image_paths:
            raise ValueError(f"No images found in {self.image_dir}")

        # Load per-image prompts if provided
        self.prompts: list[str] = []
        if config.prompt_file and Path(config.prompt_file).exists():
            lines = Path(config.prompt_file).read_text().strip().split("\n")
            self.prompts = [l.strip() for l in lines]
        else:
            self.prompts = [config.prompt] * len(self.image_paths)

        # Repeat
        if config.num_repeats > 1:
            self.image_paths = self.image_paths * config.num_repeats
            self.prompts = self.prompts * config.num_repeats

        logger.info("FineTuneDataset: %d images loaded from %s",
                     len(self.image_paths), self.image_dir)

    def __len__(self) -> int:
        """Return the number of image/prompt pairs available for training."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        """Load one image, apply augmentation, and return training-ready tensors."""
        img = Image.open(self.image_paths[idx]).convert("RGB")

        # Resize
        img = self._resize_and_crop(img)

        # Random horizontal flip
        if self.config.random_flip and torch.rand(1).item() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # To tensor [-1, 1]
        import torchvision.transforms.functional as TF
        tensor = TF.to_tensor(img) * 2.0 - 1.0

        prompt = self.prompts[idx % len(self.prompts)]
        return {"pixel_values": tensor, "prompt": prompt}

    def _resize_and_crop(self, img: Image.Image) -> Image.Image:
        """Resize an image to the square resolution expected by diffusion training."""
        w, h = img.size
        r = self.config.resolution

        if self.config.center_crop:
            # Resize shortest side to resolution, then center crop
            scale = r / min(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            left = (new_w - r) // 2
            top = (new_h - r) // 2
            img = img.crop((left, top, left + r, top + r))
        else:
            img = img.resize((r, r), Image.LANCZOS)

        return img


# ═══════════════════════════════════════════════════════════
#  HYPERPARAMETER OPTIMIZATION
# ═══════════════════════════════════════════════════════════

@dataclass
class HyperparamSearchSpace:
    """Defines the search space for hyperparameter optimization."""
    learning_rates: list[float] = field(
        default_factory=lambda: [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    )
    lora_ranks: list[int] = field(
        default_factory=lambda: [4, 8, 16, 32]
    )
    lora_alphas: list[float] = field(
        default_factory=lambda: [1.0, 4.0, 8.0, 16.0, 32.0]
    )
    batch_sizes: list[int] = field(
        default_factory=lambda: [1, 2, 4]
    )
    weight_decays: list[float] = field(
        default_factory=lambda: [0.0, 0.01, 0.1]
    )


@dataclass
class HyperparamResult:
    """Result of a hyperparameter trial."""
    params: dict[str, Any]
    loss: float
    training_time_sec: float
    step: int


class HyperparamSearcher:
    """Simple grid/random search for fine-tuning hyperparameters.

    Records all trials and recommends the best configuration.
    """

    def __init__(self, space: HyperparamSearchSpace | None = None) -> None:
        """Store the search space definition and start with an empty trial log."""
        self.space = space or HyperparamSearchSpace()
        self.trials: list[HyperparamResult] = []

    def record_trial(self, params: dict, loss: float, time_sec: float, step: int) -> None:
        """Append one completed hyperparameter trial to the history list."""
        self.trials.append(HyperparamResult(
            params=params, loss=loss, training_time_sec=time_sec, step=step,
        ))

    def best_trial(self) -> HyperparamResult | None:
        """Return the lowest-loss trial recorded so far, if one exists."""
        if not self.trials:
            return None
        return min(self.trials, key=lambda t: t.loss)

    def summary(self) -> list[dict]:
        """Return the best trials in a compact leaderboard-style format."""
        return [
            {"params": t.params, "loss": round(t.loss, 6), "time_sec": round(t.training_time_sec, 1)}
            for t in sorted(self.trials, key=lambda t: t.loss)[:10]
        ]


# ═══════════════════════════════════════════════════════════
#  LORA FINE-TUNING
# ═══════════════════════════════════════════════════════════

@dataclass
class LoRAConfig:
    """Configuration for LoRA-based fine-tuning."""
    model_id: str = "sd15"
    dataset_dir: str = ""
    output_dir: str = "./lora_output"
    prompt: str = ""
    prompt_file: Optional[str] = None

    # LoRA parameters
    lora_rank: int = 8
    lora_alpha: float = 32.0
    lora_dropout: float = 0.0
    target_modules: list[str] = field(default_factory=lambda: [
        "to_q", "to_k", "to_v", "to_out.0",
    ])

    # Training parameters
    learning_rate: float = 1e-4
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_train_steps: int = 1000
    warmup_steps: int = 100
    weight_decay: float = 0.01
    mixed_precision: str = "fp16"
    seed: int = 42

    # Logging
    log_every: int = 50
    save_every: int = 250
    resolution: int = 512


class LoRATrainer:
    """LoRA fine-tuning trainer for diffusion models.

    Uses low-rank adaptation to fine-tune only a small number
    of parameters (~4-8 MB) while keeping the base model frozen.
    """

    def __init__(self, config: LoRAConfig) -> None:
        """Capture LoRA settings and detect the best runtime device."""
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._searcher = HyperparamSearcher()

    def train(self) -> dict[str, Any]:
        """Run the LoRA fine-tuning loop.

        Returns
        -------
        dict
            Training report with loss history and output paths.
        """
        from diffusers import (
            AutoencoderKL,
            DDPMScheduler,
            StableDiffusionPipeline,
            UNet2DConditionModel,
        )
        from transformers import CLIPTextModel, CLIPTokenizer

        config = self.config
        output = Path(config.output_dir)
        output.mkdir(parents=True, exist_ok=True)

        logger.info("Starting LoRA training for model '%s'", config.model_id)
        logger.info("  Dataset: %s", config.dataset_dir)
        logger.info("  LoRA rank=%d, alpha=%.1f", config.lora_rank, config.lora_alpha)
        logger.info("  LR=%.1e, steps=%d", config.learning_rate, config.max_train_steps)

        # Load base model components
        model_repo = self._resolve_model_repo(config.model_id)
        dtype = torch.float16 if config.mixed_precision == "fp16" else torch.float32

        tokenizer = CLIPTokenizer.from_pretrained(model_repo, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(
            model_repo, subfolder="text_encoder", torch_dtype=dtype,
        ).to(self.device)
        vae = AutoencoderKL.from_pretrained(
            model_repo, subfolder="vae", torch_dtype=dtype,
        ).to(self.device)
        unet = UNet2DConditionModel.from_pretrained(
            model_repo, subfolder="unet", torch_dtype=dtype,
        ).to(self.device)
        noise_scheduler = DDPMScheduler.from_pretrained(model_repo, subfolder="scheduler")

        # Freeze base model
        text_encoder.requires_grad_(False)
        vae.requires_grad_(False)

        # Apply LoRA to UNet
        lora_params = self._apply_lora(unet, config)
        trainable = sum(p.numel() for p in lora_params)
        total = sum(p.numel() for p in unet.parameters())
        logger.info("  LoRA params: %s / %s (%.2f%%)",
                     f"{trainable:,}", f"{total:,}", 100 * trainable / total)

        # Dataset
        ds_config = FineTuneDatasetConfig(
            image_dir=config.dataset_dir,
            prompt=config.prompt,
            prompt_file=config.prompt_file,
            resolution=config.resolution,
            random_flip=True,
        )
        dataset = FineTuneDataset(ds_config)
        loader = DataLoader(
            dataset, batch_size=config.batch_size,
            shuffle=True, num_workers=0, drop_last=True,
        )

        # Optimizer
        optimizer = torch.optim.AdamW(
            lora_params, lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # LR schedule with warmup
        def lr_lambda(step):
            """Warm up linearly, then decay with a cosine schedule."""
            if step < config.warmup_steps:
                return step / max(1, config.warmup_steps)
            progress = (step - config.warmup_steps) / max(1, config.max_train_steps - config.warmup_steps)
            return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # Training loop
        loss_history = []
        global_step = 0
        start_time = time.time()
        unet.train()

        while global_step < config.max_train_steps:
            for batch in loader:
                if global_step >= config.max_train_steps:
                    break

                pixel_values = batch["pixel_values"].to(self.device, dtype=dtype)
                prompts = batch["prompt"]

                # Encode text
                input_ids = tokenizer(
                    prompts, padding="max_length", truncation=True,
                    max_length=77, return_tensors="pt",
                ).input_ids.to(self.device)
                encoder_hidden_states = text_encoder(input_ids)[0]

                # Encode images to latents
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # Sample noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=self.device,
                ).long()

                # Add noise
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Predict noise
                noise_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states,
                ).sample

                # MSE loss
                loss = F.mse_loss(noise_pred, noise, reduction="mean")
                loss = loss / config.gradient_accumulation_steps

                loss.backward()

                if (global_step + 1) % config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                global_step += 1
                loss_val = loss.item() * config.gradient_accumulation_steps
                loss_history.append(loss_val)

                if global_step % config.log_every == 0:
                    lr = scheduler.get_last_lr()[0]
                    logger.info(
                        "  Step %d/%d  loss=%.4f  lr=%.2e",
                        global_step, config.max_train_steps, loss_val, lr,
                    )
                    self._searcher.record_trial(
                        params={"lr": config.learning_rate, "rank": config.lora_rank},
                        loss=loss_val, time_sec=time.time() - start_time,
                        step=global_step,
                    )

                if global_step % config.save_every == 0:
                    self._save_lora_checkpoint(unet, output / f"checkpoint-{global_step}")

        # Save final
        elapsed = time.time() - start_time
        self._save_lora_checkpoint(unet, output / "final")

        report = {
            "model_id": config.model_id,
            "method": "lora",
            "lora_rank": config.lora_rank,
            "lora_alpha": config.lora_alpha,
            "total_steps": global_step,
            "final_loss": loss_history[-1] if loss_history else None,
            "training_time_sec": round(elapsed, 1),
            "trainable_params": trainable,
            "output_dir": str(output),
            "loss_history_size": len(loss_history),
        }

        logger.info("✓ LoRA training complete: %s", report)
        return report

    def _apply_lora(self, unet, config: LoRAConfig) -> list[torch.nn.Parameter]:
        """Inject LoRA layers into the UNet and return trainable params."""
        lora_params = []

        for name, module in unet.named_modules():
            # Check if this is a target module
            if not any(target in name for target in config.target_modules):
                continue
            if not isinstance(module, torch.nn.Linear):
                continue

            # Create LoRA matrices
            in_features = module.in_features
            out_features = module.out_features
            rank = config.lora_rank

            lora_A = torch.nn.Parameter(
                torch.randn(rank, in_features, dtype=module.weight.dtype, device=module.weight.device) * 0.01
            )
            lora_B = torch.nn.Parameter(
                torch.zeros(out_features, rank, dtype=module.weight.dtype, device=module.weight.device)
            )

            module.register_buffer("lora_A", lora_A)
            module.register_buffer("lora_B", lora_B)
            lora_params.extend([lora_A, lora_B])

            # Monkey-patch the forward
            original_forward = module.forward
            scale = config.lora_alpha / config.lora_rank

            def make_lora_forward(orig_fwd, a, b, s):
                """Capture the original layer and return a LoRA-augmented forward."""
                def lora_forward(x):
                    """Run the frozen base layer plus the trainable low-rank update."""
                    result = orig_fwd(x)
                    lora_out = (x @ a.T) @ b.T * s
                    return result + lora_out
                return lora_forward

            module.forward = make_lora_forward(original_forward, lora_A, lora_B, scale)

        return lora_params

    def _save_lora_checkpoint(self, unet, path: Path) -> None:
        """Save only LoRA parameters."""
        path.mkdir(parents=True, exist_ok=True)
        lora_state = {}
        for name, module in unet.named_modules():
            if hasattr(module, "lora_A"):
                lora_state[f"{name}.lora_A"] = module.lora_A.data.cpu()
                lora_state[f"{name}.lora_B"] = module.lora_B.data.cpu()
        torch.save(lora_state, path / "lora_weights.pt")
        logger.info("Saved LoRA checkpoint: %s", path)

    @staticmethod
    def _resolve_model_repo(model_id: str) -> str:
        """Map short internal ids to their full Hugging Face repository names."""
        _map = {
            "sd15": "stable-diffusion-v1-5/stable-diffusion-v1-5",
            "sdxl_turbo": "stabilityai/sdxl-turbo",
            "pixart_sigma": "PixArt-alpha/PixArt-Sigma-XL-2-512-MS",
            "sd3_medium": "stabilityai/stable-diffusion-3-medium-diffusers",
        }
        return _map.get(model_id, model_id)


# ═══════════════════════════════════════════════════════════
#  TEXTUAL INVERSION
# ═══════════════════════════════════════════════════════════

@dataclass
class TextualInversionConfig:
    """Configuration for textual inversion fine-tuning."""
    model_id: str = "sd15"
    dataset_dir: str = ""
    output_dir: str = "./ti_output"
    placeholder_token: str = "<my-concept>"
    initializer_token: str = "photo"
    learning_rate: float = 5e-4
    max_train_steps: int = 3000
    resolution: int = 512
    batch_size: int = 1
    seed: int = 42


class TextualInversionTrainer:
    """Textual inversion: learns a new token embedding (~6 KB).

    The entire model stays frozen — only the new embedding vector
    is trained.  Very lightweight and fast to train.
    """

    def __init__(self, config: TextualInversionConfig) -> None:
        """Capture textual inversion settings and select the training device."""
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def train(self) -> dict[str, Any]:
        """Run textual inversion training.

        Returns a training report dict.
        """
        from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTokenizer

        config = self.config
        output = Path(config.output_dir)
        output.mkdir(parents=True, exist_ok=True)

        logger.info("Starting Textual Inversion for '%s' → '%s'",
                     config.initializer_token, config.placeholder_token)

        # Load model
        model_repo = LoRATrainer._resolve_model_repo(config.model_id)
        tokenizer = CLIPTokenizer.from_pretrained(model_repo, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(
            model_repo, subfolder="text_encoder",
        ).to(self.device)
        vae = AutoencoderKL.from_pretrained(
            model_repo, subfolder="vae", torch_dtype=torch.float32,
        ).to(self.device)
        unet = UNet2DConditionModel.from_pretrained(
            model_repo, subfolder="unet", torch_dtype=torch.float32,
        ).to(self.device)
        noise_scheduler = DDPMScheduler.from_pretrained(model_repo, subfolder="scheduler")

        # Add placeholder token
        num_added = tokenizer.add_tokens(config.placeholder_token)
        if num_added == 0:
            logger.warning("Token '%s' already exists", config.placeholder_token)

        # Initialize embedding from initializer token
        token_id = tokenizer.convert_tokens_to_ids(config.placeholder_token)
        init_id = tokenizer.convert_tokens_to_ids(config.initializer_token)

        text_encoder.resize_token_embeddings(len(tokenizer))
        embeddings = text_encoder.get_input_embeddings()
        with torch.no_grad():
            embeddings.weight[token_id] = embeddings.weight[init_id].clone()

        # Freeze everything except the new token's embedding
        vae.requires_grad_(False)
        unet.requires_grad_(False)
        text_encoder.requires_grad_(False)
        embeddings.weight.requires_grad_(True)

        # Only optimize the new token
        optimizer = torch.optim.Adam([embeddings.weight], lr=config.learning_rate)

        # Dataset
        ds_config = FineTuneDatasetConfig(
            image_dir=config.dataset_dir,
            prompt=f"a photo of {config.placeholder_token}",
            resolution=config.resolution,
        )
        dataset = FineTuneDataset(ds_config)
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)

        # Training loop
        loss_history = []
        global_step = 0
        start_time = time.time()

        while global_step < config.max_train_steps:
            for batch in loader:
                if global_step >= config.max_train_steps:
                    break

                pixel_values = batch["pixel_values"].to(self.device)
                prompts = batch["prompt"]

                input_ids = tokenizer(
                    prompts, padding="max_length", truncation=True,
                    max_length=77, return_tensors="pt",
                ).input_ids.to(self.device)

                encoder_hidden_states = text_encoder(input_ids)[0]

                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=self.device,
                ).long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                with torch.no_grad():
                    noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Only the embedding gets gradients via encoder_hidden_states
                loss = F.mse_loss(noise_pred, noise)

                optimizer.zero_grad()
                loss.backward()

                # Zero out gradients for all tokens except our placeholder
                grads = embeddings.weight.grad
                if grads is not None:
                    mask = torch.zeros_like(grads)
                    mask[token_id] = 1.0
                    grads *= mask

                optimizer.step()

                global_step += 1
                loss_history.append(loss.item())

                if global_step % 100 == 0:
                    logger.info("  TI Step %d/%d  loss=%.4f",
                                 global_step, config.max_train_steps, loss.item())

        # Save learned embedding
        elapsed = time.time() - start_time
        learned = embeddings.weight[token_id].detach().cpu()
        torch.save({
            "placeholder_token": config.placeholder_token,
            "embedding": learned,
        }, output / "learned_embedding.pt")

        report = {
            "model_id": config.model_id,
            "method": "textual_inversion",
            "placeholder_token": config.placeholder_token,
            "total_steps": global_step,
            "final_loss": loss_history[-1] if loss_history else None,
            "training_time_sec": round(elapsed, 1),
            "embedding_size_bytes": learned.nelement() * 4,
            "output_dir": str(output),
        }
        logger.info("✓ Textual Inversion complete: %s", report)
        return report
