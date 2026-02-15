"""
LLM interface wrapper
Centralizes all LLM calls with proper error handling
"""
import torch
from typing import Optional
from core.model_manager import ModelManager
from core.logger import logger
from config.model_config import LLM_CONFIG

class LLMWrapper:
    """
    Wrapper around Phi-3-mini for text generation
    Handles tokenization, generation, and response parsing
    """
    
    def __init__(self):
        # Get LLM from model manager
        llm = ModelManager().get_llm()
        self.tokenizer = llm['tokenizer']
        self.model = llm['model']
        self.device = self.model.device
        
        logger.info("LLM Wrapper initialized")
    
    def generate(self, 
                 prompt: str, 
                 max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None,
                 system_prompt: Optional[str] = None) -> str:
        """
        Generate text from prompt
        
        Args:
            prompt: User prompt
            max_tokens: Max new tokens (default from config)
            temperature: Sampling temperature (default from config)
            system_prompt: Optional system message
            
        Returns:
            Generated text
        """
        max_tokens = max_tokens or LLM_CONFIG['max_new_tokens']
        temperature = temperature or LLM_CONFIG['temperature']
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            # Tokenize
            inputs = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=LLM_CONFIG['do_sample'],
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract assistant response
            response = self._extract_assistant_response(response)
            
            return response
            
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return f"Error generating response: {str(e)}"
    
    def _extract_assistant_response(self, full_response: str) -> str:
        """
        Extract only the assistant's response from full output
        Phi-3 format: <|user|> ... <|assistant|> ...
        """
        if "<|assistant|>" in full_response:
            response = full_response.split("<|assistant|>")[-1].strip()
            return response
        
        # Fallback: return everything after the prompt
        return full_response.strip()
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        tokens = self.tokenizer.encode(text)
        return len(tokens)