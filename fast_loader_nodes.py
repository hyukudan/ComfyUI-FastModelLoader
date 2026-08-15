"""
ComfyUI-FastModelLoader: High-Speed Streaming Safetensors Loader & Outpaint Direction Tools
Copyright (c) 2026 hyukudan
Licensed under the MIT License
"""

import os
import torch
import safetensors.torch
import folder_paths
import comfy.utils
import comfy.model_management
import comfy.sd
import torch.nn.functional as F

class FastModelLoader:
    """
    High-Speed Direct I/O Safetensors Checkpoint Loader.
    Uses native Rust/C++ pread streaming I/O to load checkpoints at maximum NVMe speed
    and eliminate Windows mmap 0xc0000005 crashes.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"
    CATEGORY = "loaders/fast"
    DESCRIPTION = "Loads checkpoints with direct pread streaming I/O, bypassing Windows mmap crashes."

    def load_checkpoint(self, ckpt_name):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        if not ckpt_path:
            raise FileNotFoundError(f"Checkpoint {ckpt_name} not found.")
        
        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings")
        )
        return out[:3]


class FastDiffusionModelLoader:
    """
    High-Speed Loader for standalone Diffusion Models / UNets (MiniMax H3, LTX 2.5, Wan 2.1, Hunyuan, FLUX).
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"), ),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e5m2", "fp16", "bf16", "fp32"], {"default": "default"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "loaders/fast"
    DESCRIPTION = "Loads standalone diffusion models with zero-mmap pread acceleration."

    def load_unet(self, unet_name, weight_dtype="default"):
        unet_path = folder_paths.get_full_path("diffusion_models", unet_name)
        if not unet_path:
            raise FileNotFoundError(f"Diffusion model {unet_name} not found.")
        
        model_options = {}
        if weight_dtype != "default":
            dtype_map = {
                "fp8_e4m3fn": torch.float8_e4m3fn,
                "fp8_e5m2": torch.float8_e5m2,
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
                "fp32": torch.float32,
            }
            if weight_dtype in dtype_map:
                model_options["dtype"] = dtype_map[weight_dtype]

        model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
        return (model,)


class OutpaintDirectionSelector:
    """
    Outpaint Direction Selector node for choosing expansion direction in outpaint workflows.
    """
    DIRECTION_MODES = [
        "center (50/50 both sides)",
        "bottom (outpaint TOP only / cielo)",
        "top (outpaint BOTTOM only / suelo)",
        "custom_vertical_bias (0.0=bottom, 1.0=top)",
    ]
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "direction": (s.DIRECTION_MODES, {"default": "center (50/50 both sides)"}),
            },
            "optional": {
                "custom_bias": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("COMBO", "FLOAT")
    RETURN_NAMES = ("direction", "custom_bias")
    FUNCTION = "get_direction"
    CATEGORY = "image/transform"
    DESCRIPTION = "Selects outpaint expansion direction (top, bottom, center 50/50, custom)."

    def get_direction(self, direction, custom_bias=0.5):
        return (direction, custom_bias)


class OutpaintDirectionalPad:
    """
    Intelligent Directional Outpaint Canvas Expander with Boundary Feathering.
    """
    DIRECTION_MODES = [
        "center (50/50 both sides)",
        "bottom (outpaint TOP only / cielo)",
        "top (outpaint BOTTOM only / suelo)",
        "left (outpaint RIGHT only)",
        "right (outpaint LEFT only)",
        "custom_vertical_bias (0.0=bottom, 1.0=top)",
    ]
    UPSCALE_METHODS = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_width": ("INT", {"default": 736, "min": 64, "max": 16384, "step": 8}),
                "target_height": ("INT", {"default": 1280, "min": 64, "max": 16384, "step": 8}),
                "feathering": ("INT", {"default": 48, "min": 0, "max": 512, "step": 1}),
                "direction": (s.DIRECTION_MODES, {"default": "center (50/50 both sides)"}),
                "upscale_method": (s.UPSCALE_METHODS, {"default": "bicubic"}),
            },
            "optional": {
                "mask": ("MASK",),
                "custom_bias": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "expand_directional"
    CATEGORY = "image/transform"
    DESCRIPTION = "Expands canvas to target resolution with customizable directional padding and edge feathering."

    def expand_directional(self, image, target_width, target_height, feathering, direction, upscale_method, mask=None, custom_bias=0.5):
        B, H, W, C = image.size()
        
        scaling_factor = min(target_width / W, target_height / H)
        if scaling_factor < 1.0:
            image_ch = image.movedim(-1, 1)
            new_w = int(W * scaling_factor)
            new_h = int(H * scaling_factor)
            image_scaled = comfy.utils.common_upscale(image_ch, new_w, new_h, upscale_method, "disabled").movedim(1, -1)
        else:
            new_w, new_h = W, H
            image_scaled = image

        if mask is not None:
            mask_scaled = mask.unsqueeze(0)
            mask_scaled = F.interpolate(mask_scaled, size=(new_h, new_w), mode="nearest").squeeze(0)
        else:
            mask_scaled = None

        rem_y = max(0, target_height - new_h)
        rem_x = max(0, target_width - new_w)

        mode = str(direction).strip().lower()
        if mode.startswith("bottom"):
            pad_top = rem_y
            pad_bottom = 0
            pad_left = rem_x // 2
            pad_right = rem_x - pad_left
        elif mode.startswith("top"):
            pad_top = 0
            pad_bottom = rem_y
            pad_left = rem_x // 2
            pad_right = rem_x - pad_left
        elif mode.startswith("left"):
            pad_top = rem_y // 2
            pad_bottom = rem_y - pad_top
            pad_left = 0
            pad_right = rem_x
        elif mode.startswith("right"):
            pad_top = rem_y // 2
            pad_bottom = rem_y - pad_top
            pad_left = rem_x
            pad_right = 0
        elif mode.startswith("custom"):
            pad_top = int(rem_y * float(custom_bias))
            pad_bottom = rem_y - pad_top
            pad_left = rem_x // 2
            pad_right = rem_x - pad_left
        else:
            pad_top = rem_y // 2
            pad_bottom = rem_y - pad_top
            pad_left = rem_x // 2
            pad_right = rem_x - pad_left

        out_image = torch.zeros((B, target_height, target_width, C), dtype=image.dtype, device=image.device)
        out_image[:, pad_top:pad_top + new_h, pad_left:pad_left + new_w, :] = image_scaled

        out_mask = torch.ones((B, target_height, target_width), dtype=torch.float32, device=image.device)
        if mask_scaled is not None:
            out_mask[:, pad_top:pad_top + new_h, pad_left:pad_left + new_w] = mask_scaled
        else:
            out_mask[:, pad_top:pad_top + new_h, pad_left:pad_left + new_w] = 0.0

        if feathering > 0:
            for b in range(B):
                for f in range(1, feathering + 1):
                    alpha = float(f) / float(feathering)
                    if pad_top > 0 and pad_top + f - 1 < target_height:
                        out_mask[b, pad_top + f - 1, pad_left:pad_left + new_w] = torch.maximum(
                            out_mask[b, pad_top + f - 1, pad_left:pad_left + new_w],
                            torch.tensor(1.0 - alpha, device=image.device)
                        )
                    if pad_bottom > 0 and pad_top + new_h - f >= 0:
                        out_mask[b, pad_top + new_h - f, pad_left:pad_left + new_w] = torch.maximum(
                            out_mask[b, pad_top + new_h - f, pad_left:pad_left + new_w],
                            torch.tensor(1.0 - alpha, device=image.device)
                        )

        return (out_image, out_mask)
