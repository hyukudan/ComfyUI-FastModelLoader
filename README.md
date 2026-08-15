# ⚡ ComfyUI-FastModelLoader

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)](https://github.com/hyukudan/ComfyUI-FastModelLoader)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Compatible-brightgreen.svg)](https://github.com/comfyanonymous/ComfyUI)

> **High-Speed Streaming Direct I/O Safetensors Loader & Multi-Stage Memory Resilience Patcher for ComfyUI.**  
> Delivers sustained ~2.7 GB/s NVMe streaming I/O (`backend="pread"`) and eliminates Windows memory-swapping page fault crashes during heavy multi-stage video pipelines (MiniMax H3, LTX 2.5, Wan 2.1, Gemma 4, Hunyuan, FLUX).

---

## 🎯 When and Why This Happens

In standard everyday workflows with single 2–4 GB checkpoints (SD 1.5, SDXL), default ComfyUI loading is fast and stable.

However, in **advanced multi-stage video workflows** with modern **15–20 GB models** (such as 2-stage Outpainting, Reference-to-Video, or complex pipelines chaining Text Encoders + DiT + Audio/Video VAE + Frame Interpolation / Upscaling), ComfyUI continuously **swaps, offloads, and reloads** massive tensors between VRAM, system RAM, and disk.

Under these specific high-pressure conditions on Windows:
1. **Memory-Mapped Page Faults:** Default Python `safe_open` relies on Windows virtual memory mapping (`mmap`). When the OS manages 40–80 GB of active tensors and swaps weights, Windows memory-mapped page pointers can desynchronize, causing an Access Violation (`0xc0000005` in `torch.storage.UntypedStorage.__getitem__`).
2. **Garbage Collection Race Conditions:** When a 15–20 GB model is evicted from VRAM to make room for the next stage (e.g. DiT -> VAE decode -> Stage 2 Upscaler), Python's garbage collector unpinning hooks can trigger null-pointer edge cases during model detachment.
3. **CPU Video Resize Heap Fragmentation:** Processing full 2K/4K video frame batches with CPU Lanczos can exhaust the NumPy heap (`numpy._core._exceptions._ArrayMemoryError`).

---

## 🚀 The Solution

**ComfyUI-FastModelLoader** delivers three core improvements:

### 1. Direct Streaming I/O (`backend="pread"`)
- Replaces deferred virtual memory mapping (`mmap`) with real contiguous disk streaming via Rust/C++ (`backend="pread"`).
- Reads entire 20 GB models in a single continuous stream at sustained **~2.7 GB/s NVMe speed** (e.g. 19.5 GB MiniMax H3 in 7.3s).
- Eliminates Windows page table faults completely during multi-model execution.

### 2. Multi-Stage Garbage Collection Resilience
- Wraps `ModelPatcher.detach()` and `__del__()` with safe detachment and callback protections.
- Guarantees seamless multi-stage model swapping (Stage 1 -> Stage 2, Text Encoder -> DiT -> VAE -> VFI) without unexpected process exits.

### 3. Safe Video Batch Resizer
- Protects video frame resizing at 2K/4K resolutions by routing large batches directly through GPU PyTorch interpolation, avoiding CPU host RAM exhaustion.

---

## 📦 Included Nodes

| Node Name | Display Title | Description |
| :--- | :--- | :--- |
| `FastModelLoader` | **⚡ Fast Model Checkpoint Loader (Pread)** | High-speed direct I/O checkpoint loader. |
| `FastDiffusionModelLoader` | **⚡ Fast Diffusion Model Loader (Pread)** | Standalone UNet/DiT model loader (MiniMax H3, LTX 2.5, Wan 2.1, FLUX). |

---

## 🛠️ Installation

### Method 1: ComfyUI Manager
1. Open ComfyUI Manager.
2. Search for `ComfyUI-FastModelLoader`.
3. Click **Install** and restart ComfyUI.

### Method 2: Manual Git Clone
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/hyukudan/ComfyUI-FastModelLoader.git
```
Restart ComfyUI. The extension automatically activates on startup and protects all standard loaders transparently.

---

## 📊 Measured Load Performance on Windows 11

Measured on Intel Core i7-13700K, NVMe PCIe 4.0 SSD, NVIDIA RTX PRO 6000 Blackwell:

| Model File | File Size | Load Time (FastModelLoader) | Direct I/O Bandwidth | Multi-Stage Execution |
| :--- | :--- | :--- | :--- | :--- |
| **MiniMax H3 DiT INT8** | 19.53 GB | **7.37 s** | 2,713 MB/s (~2.7 GB/s) | ✅ **100% Stable** (no `0xc0000005` crash) |
| **Gemma 4 LTX Text Encoder** | 14.32 GB | **5.43 s** | 2,700 MB/s (~2.7 GB/s) | ✅ **100% Stable** |
| **ACE-Step XL Base** | 9.29 GB | **4.32 s** | 2,200 MB/s (~2.2 GB/s) | ✅ **100% Stable** |
| **T5 XXL FP8** | 4.56 GB | **2.32 s** | 2,011 MB/s (~2.0 GB/s) | ✅ **100% Stable** |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) - Copyright (c) 2026 hyukudan.
