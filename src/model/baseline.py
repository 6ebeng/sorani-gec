"""
Baseline Transformer Model

Standard Transformer seq2seq model for GEC without morphological features.
Uses a pretrained multilingual model (mT5 or ByT5) as the backbone.
"""

import logging
import time
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger(__name__)


class BaselineGEC(nn.Module):
    """Baseline GEC model using pretrained multilingual Transformer.

    RTL Script Note: ByT5 operates on UTF-8 bytes in storage order,
    making explicit RTL positional encoding unnecessary. See
    MorphologyAwareGEC docstring for detailed rationale.
    """
    
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

    def training_step(
        self, source_texts: list[str], target_texts: list[str]
    ) -> torch.Tensor:
        """Tokenize a batch of (source, target) pairs and return the loss.

        Convenience method used by ``scripts/08_ablation.py``.
        """
        inputs = self.tokenizer(
            source_texts, return_tensors="pt",
            max_length=self.max_length, truncation=True, padding="max_length",
        )
        labels = self.tokenizer(
            target_texts, return_tensors="pt",
            max_length=self.max_length, truncation=True, padding="max_length",
        )["input_ids"]
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)
        outputs = self.forward(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=labels,
        )
        return outputs["loss"]
    
    def correct(self, text: str, num_beams: int = 4,
                length_penalty: float = 1.0,
                no_repeat_ngram_size: int = 0,
                max_length: int | None = None) -> str:
        """Correct a single sentence.
        
        Args:
            text: Corrupted Sorani Kurdish sentence.
            num_beams: Beam search width.
            length_penalty: Exponential penalty on length (>1 = longer, <1 = shorter).
            no_repeat_ngram_size: If >0, prevents repeated n-grams of this size.
            max_length: Override max generation length (defaults to self.max_length).
            
        Returns:
            Corrected sentence.
        """
        t_start = time.perf_counter()
        gen_max_length = max_length or self.max_length
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
        )
        
        # Move to same device as model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        gen_kwargs: dict = {
            "max_length": gen_max_length,
            "num_beams": num_beams,
            "early_stopping": True,
            "length_penalty": length_penalty,
        }
        if no_repeat_ngram_size > 0:
            gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        corrected = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        logger.debug("correct() latency: %.1f ms", elapsed_ms)
        return corrected
    
    def correct_batch(self, texts: list[str], num_beams: int = 4) -> list[str]:
        """Correct a batch of sentences."""
        t_start = time.perf_counter()
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
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
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        logger.debug("correct_batch() latency: %.1f ms for %d sentences (%.1f ms/sent)",
                      elapsed_ms, len(texts), elapsed_ms / max(len(texts), 1))
        return corrected

    def correct_with_confidence(
        self, text: str, num_beams: int = 4
    ) -> tuple[str, float]:
        """Correct a sentence and return a confidence score.

        The confidence is the mean token-level log-probability of the
        generated sequence, normalized by length.

        Returns:
            (corrected_text, confidence) where confidence is in (0, 1].
        """
        inputs = self.tokenizer(
            text, return_tensors="pt",
            max_length=self.max_length, truncation=True, padding="max_length",
        )
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            gen_out = self.model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=num_beams,
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True,
            )

        corrected = self.tokenizer.decode(gen_out.sequences[0], skip_special_tokens=True)

        # Compute mean log-prob across generated tokens
        if gen_out.scores:
            import torch.nn.functional as F
            log_probs = []
            for step, score in enumerate(gen_out.scores):
                probs = F.log_softmax(score[0], dim=-1)
                token_id = gen_out.sequences[0, step + 1]
                log_probs.append(probs[token_id].item())
            mean_log_prob = sum(log_probs) / len(log_probs) if log_probs else 0.0
            import math
            confidence = min(1.0, math.exp(mean_log_prob))
        else:
            confidence = 1.0

        return corrected, confidence
