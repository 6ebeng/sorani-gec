"""
Baseline Transformer Model

Standard Transformer seq2seq model for GEC without morphological features.
Uses a pretrained multilingual model (mT5 or ByT5) as the backbone.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger(__name__)


class BaselineGEC(nn.Module):
    """Baseline GEC model using pretrained multilingual Transformer."""
    
    def __init__(
        self,
        model_name: str = "google/byt5-small",
        max_length: int = 128,
    ):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        
        logger.info("Loading pretrained model: %s", model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """Forward pass.
        
        Args:
            input_ids: Tokenized corrupted sentences [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            labels: Tokenized clean sentences [batch, seq_len] (for training)
            
        Returns:
            Dict with 'loss' and 'logits' keys.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
        }
    
    def correct(self, text: str, num_beams: int = 4) -> str:
        """Correct a single sentence.
        
        Args:
            text: Corrupted Sorani Kurdish sentence.
            num_beams: Beam search width.
            
        Returns:
            Corrected sentence.
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True,
        )
        
        # Move to same device as model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=num_beams,
                early_stopping=True,
            )
        
        corrected = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return corrected
    
    def correct_batch(self, texts: list[str], num_beams: int = 4) -> list[str]:
        """Correct a batch of sentences."""
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True,
        )
        
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=num_beams,
                early_stopping=True,
            )
        
        corrected = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return corrected
