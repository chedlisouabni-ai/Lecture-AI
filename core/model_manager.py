"""
Centralized model loading and caching
Ensures models are loaded once and reused
"""
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,  # ← ADDED THIS
    BlipProcessor, 
    BlipForConditionalGeneration,
    CLIPProcessor, 
    CLIPModel
)
from sentence_transformers import SentenceTransformer
import whisper
import os
from config.model_config import *
from config.paths import *
from core.logger import logger

class ModelManager:
    """
    Singleton pattern for model management
    Load models once, reuse everywhere
    """
    _instance = None
    _models = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._models:
            logger.info("Initializing Model Manager...")
    
    def get_llm(self):
        """Get or load LLM model"""
        if 'llm' not in self._models:
            logger.info(f"Loading LLM: {LLM_CONFIG['model_name']}")
            
            try:
                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    LLM_CONFIG['model_name'],
                    cache_dir=str(LLM_CACHE_DIR),
                    trust_remote_code=True
                )
                
                # Set up quantization config if using 4-bit
                if LLM_CONFIG['use_4bit']:
                    logger.info("  Using 4-bit quantization...")
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=QUANTIZATION_CONFIG['load_in_4bit'],
                        bnb_4bit_use_double_quant=QUANTIZATION_CONFIG['bnb_4bit_use_double_quant'],
                        bnb_4bit_quant_type=QUANTIZATION_CONFIG['bnb_4bit_quant_type'],
                        bnb_4bit_compute_dtype=QUANTIZATION_CONFIG['bnb_4bit_compute_dtype']
                    )
                    
                    model = AutoModelForCausalLM.from_pretrained(
                        LLM_CONFIG['model_name'],
                        quantization_config=quantization_config,  # ← CORRECT WAY
                        device_map=LLM_CONFIG['device_map'],
                        cache_dir=str(LLM_CACHE_DIR),
                        trust_remote_code=True
                    )
                else:
                    # Load without quantization
                    logger.info("  Loading without quantization...")
                    model = AutoModelForCausalLM.from_pretrained(
                        LLM_CONFIG['model_name'],
                        torch_dtype=LLM_CONFIG['dtype'],
                        device_map=LLM_CONFIG['device_map'],
                        cache_dir=str(LLM_CACHE_DIR),
                        trust_remote_code=True
                    )
                
                self._models['llm'] = {
                    'tokenizer': tokenizer,
                    'model': model
                }
                
                logger.info("✓ LLM loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load LLM: {e}")
                logger.error("Trying without quantization as fallback...")
                
                # Fallback: Load without quantization
                tokenizer = AutoTokenizer.from_pretrained(
                    LLM_CONFIG['model_name'],
                    cache_dir=str(LLM_CACHE_DIR),
                    trust_remote_code=True
                )
                
                model = AutoModelForCausalLM.from_pretrained(
                    LLM_CONFIG['model_name'],
                    torch_dtype=torch.float16,
                    device_map=LLM_CONFIG['device_map'],
                    cache_dir=str(LLM_CACHE_DIR),
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                
                self._models['llm'] = {
                    'tokenizer': tokenizer,
                    'model': model
                }
                
                logger.info("✓ LLM loaded without quantization")
        
        return self._models['llm']
    
    def get_whisper(self):
        """Get or load Whisper model"""
        if 'whisper' not in self._models:
            logger.info(f"Loading Whisper: {WHISPER_CONFIG['model_size']}")
            
            try:
                # Set cache directory via environment variable
                os.environ['WHISPER_CACHE_DIR'] = str(WHISPER_CACHE_DIR)
                
                model = whisper.load_model(
                    WHISPER_CONFIG['model_size'],
                    download_root=str(WHISPER_CACHE_DIR)
                )
                
                self._models['whisper'] = model
                logger.info("✓ Whisper loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load Whisper: {e}")
                raise
        
        return self._models['whisper']
    
    def get_blip(self):
        """Get or load BLIP model"""
        if 'blip' not in self._models:
            logger.info(f"Loading BLIP: {BLIP_CONFIG['model_name']}")
            
            try:
                processor = BlipProcessor.from_pretrained(
                    BLIP_CONFIG['model_name'],
                    cache_dir=str(BLIP_CACHE_DIR)
                )
                
                model = BlipForConditionalGeneration.from_pretrained(
                    BLIP_CONFIG['model_name'],
                    cache_dir=str(BLIP_CACHE_DIR),
                    torch_dtype=torch.float16
                ).to('cuda' if torch.cuda.is_available() else 'cpu')
                
                self._models['blip'] = {
                    'processor': processor,
                    'model': model
                }
                
                logger.info("✓ BLIP loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load BLIP: {e}")
                raise
        
        return self._models['blip']
    
    def get_clip(self):
        """Get or load CLIP model"""
        if 'clip' not in self._models:
            logger.info(f"Loading CLIP: {CLIP_CONFIG['model_name']}")
            
            try:
                processor = CLIPProcessor.from_pretrained(
                    CLIP_CONFIG['model_name'],
                    cache_dir=str(CLIP_CACHE_DIR)
                )
                
                model = CLIPModel.from_pretrained(
                    CLIP_CONFIG['model_name'],
                    cache_dir=str(CLIP_CACHE_DIR)
                ).to('cuda' if torch.cuda.is_available() else 'cpu')
                
                self._models['clip'] = {
                    'processor': processor,
                    'model': model
                }
                
                logger.info("✓ CLIP loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load CLIP: {e}")
                raise
        
        return self._models['clip']
    
    def get_embedding_model(self):
        """Get or load sentence transformer"""
        if 'embeddings' not in self._models:
            logger.info(f"Loading embeddings: {EMBEDDING_CONFIG['model_name']}")
            
            try:
                model = SentenceTransformer(
                    EMBEDDING_CONFIG['model_name'],
                    cache_folder=str(EMBEDDING_CACHE_DIR)
                )
                
                self._models['embeddings'] = model
                logger.info("✓ Embedding model loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load embeddings: {e}")
                raise
        
        return self._models['embeddings']
    
    def preload_all(self):
        """Preload all models at startup"""
        logger.info("Preloading all models...")
        
        try:
            self.get_llm()
            self.get_whisper()
            self.get_blip()
            self.get_embedding_model()
            logger.info("✓ All models preloaded")
        except Exception as e:
            logger.error(f"Error preloading models: {e}")
            raise