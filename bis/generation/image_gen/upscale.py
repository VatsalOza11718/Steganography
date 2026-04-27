"""Image upscaling pipeline — FHD → 2K → 4K.

Supports multiple upscaling strategies:
  1. Real-ESRGAN (best quality, GPU required)
  2. SD x4 Upscaler (diffusion-based, slower but higher quality)
  3. Lanczos fallback (CPU-only, fast, moderate quality)

Usage:
    from bis.generation.image_gen.upscale import upscale_image, UpscaleTarget

    result = upscale_image(pil_img, target=UpscaleTarget.UHD_4K)
"""

from __future__ import annotations

import io
import math
from enum import Enum
from typing import Optional

from PIL import Image

# ── Target Resolutions ───────────────────────────────────────

class UpscaleTarget(Enum):
    """Target output resolution."""
    FHD = "1080p"        # 1920 × 1080
    QHD_2K = "2k"        # 2560 × 1440
    UHD_4K = "4k"        # 3840 × 2160


_RESOLUTION_MAP = {
    UpscaleTarget.FHD:    (1920, 1080),
    UpscaleTarget.QHD_2K: (2560, 1440),
    UpscaleTarget.UHD_4K: (3840, 2160),
}


# ── Upscaler Backend Selection ───────────────────────────────

def _try_real_esrgan() -> bool:
    """Check if Real-ESRGAN is available."""
    try:
        from realesrgan import RealESRGANer  # noqa: F401
        return True
    except ImportError:
        return False


def _try_sd_upscaler() -> bool:
    """Check if diffusers SD x4 upscaler is available."""
    try:
        from diffusers import StableDiffusionUpscalePipeline  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


# ── Lanczos Upscaler (always available) ──────────────────────

def _upscale_lanczos(
    image: Image.Image,
    target_w: int,
    target_h: int,
) -> Image.Image:
    """High-quality Lanczos resampling with optional sharpening."""
    from PIL import ImageFilter

    # Resize to cover the requested canvas first, then crop the center.
    # This avoids distortion while still producing the exact target size.
    src_w, src_h = image.size
    scale = max(target_w / src_w, target_h / src_h)
    new_w = int(src_w * scale + 0.5)
    new_h = int(src_h * scale + 0.5)

    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Center crop to exact target
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    cropped = resized.crop((left, top, left + target_w, top + target_h))

    # Light sharpening to counteract upscale softness
    sharpened = cropped.filter(ImageFilter.UnsharpMask(radius=1.5, percent=40, threshold=2))
    return sharpened


# ── Real-ESRGAN Upscaler ─────────────────────────────────────

def _upscale_real_esrgan(
    image: Image.Image,
    target_w: int,
    target_h: int,
    model_name: str = "RealESRGAN_x4plus",
    tile: int = 512,
) -> Image.Image:
    """Upscale using Real-ESRGAN neural network."""
    import numpy as np
    import torch
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = RRDBNet(
        num_in_ch=3, num_out_ch=3,
        num_feat=64, num_block=23, num_grow_ch=32,
    )

    upsampler = RealESRGANer(
        scale=4,
        model_path=None,  # auto-downloads
        model=model,
        tile=tile,
        tile_pad=10,
        pre_pad=0,
        half=device == "cuda",
        device=device,
    )

    img_array = np.array(image.convert("RGB"))
    output, _ = upsampler.enhance(img_array, outscale=4)

    result = Image.fromarray(output)

    # Resize to exact target if needed
    if result.size != (target_w, target_h):
        result = _upscale_lanczos(result, target_w, target_h)

    return result


# ── SD x4 Upscaler ───────────────────────────────────────────

def _upscale_sd_x4(
    image: Image.Image,
    target_w: int,
    target_h: int,
    prompt: str = "high quality, ultra detailed, sharp focus, 4k photograph",
    num_steps: int = 20,
) -> Image.Image:
    """Upscale using Stable Diffusion x4 upscaler pipeline."""
    import torch
    from diffusers import StableDiffusionUpscalePipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionUpscalePipeline.from_pretrained(
        "stabilityai/stable-diffusion-x4-upscaler",
        torch_dtype=dtype,
    )
    pipe = pipe.to(device)

    if device == "cuda":
        pipe.enable_attention_slicing()
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    # The diffusion upscaler expects an image that is already low resolution.
    # We synthesize that "low-res input" by downscaling the original first.
    low_res = image.resize((target_w // 4, target_h // 4), Image.Resampling.LANCZOS)

    result = pipe(
        prompt=prompt,
        image=low_res,
        num_inference_steps=num_steps,
        guidance_scale=4.0,
    ).images[0]

    if result.size != (target_w, target_h):
        result = _upscale_lanczos(result, target_w, target_h)

    return result


# ── Main Upscale Function ────────────────────────────────────

def upscale_image(
    image: Image.Image,
    target: UpscaleTarget = UpscaleTarget.FHD,
    method: Optional[str] = None,
    prompt: str = "",
) -> Image.Image:
    """Upscale an image to the target resolution.

    Parameters
    ----------
    image : PIL.Image.Image
        Source image (any size).
    target : UpscaleTarget
        Desired output resolution (FHD, 2K, or 4K).
    method : str, optional
        Force a specific method: "real_esrgan", "sd_x4", or "lanczos".
        If None, auto-selects the best available backend.
    prompt : str
        Text prompt for SD x4 upscaler (ignored by other methods).

    Returns
    -------
    PIL.Image.Image
        Upscaled image at the target resolution.
    """
    target_w, target_h = _RESOLUTION_MAP[target]

    # Already at or above target? Just crop/resize to exact dimensions
    src_w, src_h = image.size
    if src_w >= target_w and src_h >= target_h:
        return _upscale_lanczos(image, target_w, target_h)

    # Prefer learned upscalers when installed, then fall back to pure image
    # resampling so the function still works on minimal environments.
    if method is None:
        if _try_real_esrgan():
            method = "real_esrgan"
        elif _try_sd_upscaler():
            method = "sd_x4"
        else:
            method = "lanczos"

    if method == "real_esrgan":
        return _upscale_real_esrgan(image, target_w, target_h)
    elif method == "sd_x4":
        return _upscale_sd_x4(image, target_w, target_h, prompt=prompt or "high resolution photograph")
    else:
        return _upscale_lanczos(image, target_w, target_h)


def image_to_bytes(image: Image.Image, fmt: str = "PNG", quality: int = 95) -> bytes:
    """Convert PIL Image to bytes."""
    buf = io.BytesIO()
    save_kwargs = {"format": fmt}
    if fmt.upper() in ("JPEG", "JPG", "WEBP"):
        save_kwargs["quality"] = quality
    image.save(buf, **save_kwargs)
    buf.seek(0)
    return buf.read()
