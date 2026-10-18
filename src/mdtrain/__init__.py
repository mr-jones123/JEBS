"""Pinned native multimodal decision training and paired artifact inference.

Direct requirements are torch 2.14.1, Transformers 5.19.0, torchvision,
NumPy, Pillow, huggingface-hub, PEFT, Accelerate, safetensors and psutil.
ffmpeg and ffprobe must be installed separately. No command downloads weights.
The parent owns the dependency lock and platform installation instructions.

Structural evidence comes from ``python -m mdtrain smoke --kind tiny``.
Pretrained CPU and actual-CUDA evidence are separate commands and receipts.
The elapsed allowance never guarantees that provider billing has stopped.
"""
