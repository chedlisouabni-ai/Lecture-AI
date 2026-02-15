"""
Model-specific configurations
"""
import torch

# LLM Configuration
LLM_CONFIG = {
    'model_name': 'microsoft/Phi-3-mini-4k-instruct',
    'use_4bit': True,  # Whether to use 4-bit quantization
    'dtype': torch.float16,  # Changed from 'torch_dtype'
    'device_map': 'auto',
    'max_new_tokens': 500,
    'temperature': 0.7,
    'do_sample': True,
}

# 4-bit Quantization Configuration
QUANTIZATION_CONFIG = {
    'load_in_4bit': True,
    'bnb_4bit_use_double_quant': True,  # Use double quantization
    'bnb_4bit_quant_type': "nf4",  # NormalFloat4 quantization
    'bnb_4bit_compute_dtype': torch.float16  # Compute dtype for 4-bit
}

# Whisper Configuration
WHISPER_CONFIG = {
    'model_size': 'base',  # tiny, base, small, medium, large
    'language': 'en',
    'task': 'transcribe',
    'verbose': False,
}

# BLIP Configuration (Image Captioning)
BLIP_CONFIG = {
    'model_name': 'Salesforce/blip-image-captioning-base',
    'max_new_tokens': 100,
}

# CLIP Configuration (Image Embeddings)
CLIP_CONFIG = {
    'model_name': 'openai/clip-vit-base-patch32',
}

# Embedding Configuration
EMBEDDING_CONFIG = {
    'model_name': 'all-MiniLM-L6-v2',
    'batch_size': 32,
}

# Scene Detection Configuration
SCENE_DETECTION_CONFIG = {
    'threshold': 27.0,  # Lower = more sensitive
    'min_scene_length': 15,  # Minimum frames per scene
}

# OCR Configuration
OCR_CONFIG = {
    'lang': 'eng',
    'config': '--psm 3',  # Page segmentation mode
}