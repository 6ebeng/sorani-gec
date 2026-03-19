"""
Morphology-Aware Transformer Model

Extends the baseline Transformer with explicit morphological feature embeddings,
agreement graph attention biasing, and an auxiliary agreement prediction loss
to improve agreement error correction.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.morphology.graph import EDGE_TYPE_ORDER

logger = logging.getLogger(__name__)


class MorphologicalEmbedding(nn.Module):
    """Embedding layer for morphological features."""
    
    def __init__(
        self,
        feature_vocab_size: int,
        num_features: int = 9,
        embed_dim: int = 64,
    ):
        super().__init__()
        self.num_features = num_features
        self.embed_dim = embed_dim
        
        # One embedding per feature type
        self.feature_embeddings = nn.ModuleList([
            nn.Embedding(feature_vocab_size, embed_dim)
            for _ in range(num_features)
        ])
        
        # Project concatenated features to match model dimension
        self.projection = nn.Linear(num_features * embed_dim, embed_dim)
    
    def forward(self, feature_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feature_indices: [batch, seq_len, num_features] integer tensor
            
        Returns:
            [batch, seq_len, embed_dim] feature embedding tensor
        """
        embeddings = []
        for i, emb_layer in enumerate(self.feature_embeddings):
            feat_emb = emb_layer(feature_indices[:, :, i])  # [batch, seq_len, embed_dim]
            embeddings.append(feat_emb)
        
        # Concatenate and project
        concat = torch.cat(embeddings, dim=-1)  # [batch, seq_len, num_features * embed_dim]
        projected = self.projection(concat)      # [batch, seq_len, embed_dim]
        
        return projected


class AgreementPredictor(nn.Module):
    """Auxiliary head for predicting agreement violations."""
    
    def __init__(self, hidden_dim: int, num_agreement_types: int = len(EDGE_TYPE_ORDER) + 1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_agreement_types),
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            
        Returns:
            [batch, seq_len, num_agreement_types] logits
        """
        return self.classifier(hidden_states)


class MorphologyAwareGEC(nn.Module):
    """GEC model with morphological feature integration."""
    
    def __init__(
        self,
        model_name: str = "google/byt5-small",
        feature_vocab_size: int = 50,
        num_morph_features: int = 9,
        morph_embed_dim: int = 64,
        agreement_loss_weight: float = 0.3,
        edge_type_loss_weights: Optional[dict[str, float]] = None,
        max_length: int = 128,
        num_agreement_types: int = len(EDGE_TYPE_ORDER) + 1,
    ):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.agreement_loss_weight = agreement_loss_weight
        self.edge_type_loss_weights = edge_type_loss_weights
        
        # Pretrained backbone
        logger.info("Loading pretrained model: %s", model_name)
        self.backbone = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Get hidden dimension from backbone
        hidden_dim = self.backbone.config.d_model
        
        # Morphological feature embedding
        self.morph_embedding = MorphologicalEmbedding(
            feature_vocab_size=feature_vocab_size,
            num_features=num_morph_features,
            embed_dim=morph_embed_dim,
        )
        
        # Projection to add morph features to hidden states
        self.morph_projection = nn.Linear(
            hidden_dim + morph_embed_dim, hidden_dim
        )
        self.morph_layer_norm = nn.LayerNorm(hidden_dim)
        
        # Auxiliary agreement predictor
        self.agreement_predictor = AgreementPredictor(
            hidden_dim=hidden_dim,
            num_agreement_types=num_agreement_types,
        )
        
        self.agreement_loss_fn = nn.CrossEntropyLoss()

        # Learnable per-edge-type weights for typed agreement masks.
        # Length matches EDGE_TYPE_ORDER; dynamically sized.
        self.max_edge_types = len(EDGE_TYPE_ORDER)
        self.edge_type_weights = nn.Parameter(
            torch.ones(self.max_edge_types) / self.max_edge_types
        )
    
    def _build_agreement_bias(
        self,
        agreement_mask: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """Convert agreement adjacency matrix to byte-level attention bias.

        Handles two input formats:
          3D [batch, N, N]               — binary adjacency (backward compat)
          4D [batch, num_types, N, N]    — typed adjacency with learnable weights

        Args:
            agreement_mask: binary (3D) or typed (4D) adjacency tensor
            seq_len: target byte-level sequence length

        Returns:
            [batch, 1, seq_len, seq_len] additive attention bias
        """
        agreement_mask = agreement_mask.to(self.edge_type_weights.device)
        if agreement_mask.dim() == 4:
            # Typed: apply learnable per-type weights then sum
            num_types = agreement_mask.size(1)
            weights = torch.softmax(
                self.edge_type_weights[:num_types], dim=0
            )  # [num_types]
            combined = (
                agreement_mask.float()
                * weights.view(1, -1, 1, 1)
            ).sum(dim=1)  # [batch, N, N]
        else:
            # Binary: use as-is
            combined = agreement_mask.float()

        n_words = combined.size(1)
        if n_words >= seq_len:
            bias = combined[:, :seq_len, :seq_len]
        else:
            bias = torch.zeros(
                combined.size(0), seq_len, seq_len,
                device=combined.device,
            )
            bias[:, :n_words, :n_words] = combined
        # Additive bias: agreement-linked positions get a positive boost
        return bias.unsqueeze(1) * 2.0  # [batch, 1, seq_len, seq_len]

    def _integrate_morph_features(
        self,
        hidden_states: torch.Tensor,
        morph_features: torch.Tensor,
    ) -> torch.Tensor:
        """Integrate morphological embeddings into encoder hidden states.

        Pads or truncates morph embeddings to match hidden_states length,
        concatenates, projects, and applies LayerNorm.

        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            morph_features: [batch, word_len, num_features]

        Returns:
            [batch, seq_len, hidden_dim] updated hidden states
        """
        morph_emb = self.morph_embedding(morph_features)
        target_len = hidden_states.size(1)
        if morph_emb.size(1) != target_len:
            if morph_emb.size(1) < target_len:
                # Repeat-interpolate word-level embeddings to fill byte
                # positions instead of zero-padding, which would dilute
                # morphological signal for longer tokens.
                morph_emb = torch.nn.functional.interpolate(
                    morph_emb.transpose(1, 2),  # [B, dim, word_len]
                    size=target_len,
                    mode="nearest",
                ).transpose(1, 2)  # [B, target_len, dim]
            else:
                morph_emb = morph_emb[:, :target_len, :]
        combined = torch.cat([hidden_states, morph_emb], dim=-1)
        return self.morph_layer_norm(self.morph_projection(combined))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        morph_features: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        agreement_labels: Optional[torch.Tensor] = None,
        agreement_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """Forward pass with morphological features and agreement graph.
        
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            morph_features: [batch, seq_len, num_features] — optional
            labels: [batch, seq_len] target token ids
            agreement_labels: [batch, seq_len] agreement type labels
            agreement_mask: adjacency tensor — either
                3D [batch, N, N] binary from to_adjacency_matrix(), or
                4D [batch, num_types, N, N] typed from to_typed_stacked_matrix()
            
        Returns:
            Dict with 'loss', 'logits', 'agreement_logits'.
        """
        # Get encoder outputs
        encoder_outputs = self.backbone.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = encoder_outputs.last_hidden_state
        
        # Integrate morphological features if provided
        if morph_features is not None:
            hidden_states = self._integrate_morph_features(hidden_states, morph_features)
        
        # Agreement prediction (auxiliary task)
        agreement_logits = self.agreement_predictor(hidden_states)
        
        # Decoder with modified encoder outputs
        # Replace encoder hidden states
        encoder_outputs.last_hidden_state = hidden_states
        
        # Build cross-attention mask with agreement bias if provided
        cross_attn_mask = attention_mask
        if agreement_mask is not None:
            agr_bias = self._build_agreement_bias(
                agreement_mask, hidden_states.size(1)
            )
            # Expand 1D padding mask to 2D and add agreement bias
            expanded = attention_mask.unsqueeze(1).unsqueeze(2).float()
            cross_attn_mask = (expanded + agr_bias).clamp(min=0, max=1)
            # Collapse back for HF API: use the original 1D mask so the
            # backbone handles padding, but store biased states.
            # ByT5 does not expose a head-level mask parameter, so we
            # bake the bias into the hidden states via a residual gate.
            # Gate is per-position (vector-level) rather than scalar for
            # finer-grained modulation.
            gate = torch.sigmoid(agr_bias.squeeze(1).mean(dim=-1, keepdim=True))  # [batch, seq, 1]
            hidden_states = hidden_states * (1.0 + gate)
            encoder_outputs.last_hidden_state = hidden_states
        
        # Main GEC loss
        outputs = self.backbone(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        total_loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0, device=hidden_states.device)
        
        # Agreement auxiliary loss — with optional per-edge-type weighting
        if agreement_labels is not None:
            agr_logits_flat = agreement_logits.view(-1, agreement_logits.size(-1))
            agr_labels_flat = agreement_labels.view(-1)

            if self.edge_type_loss_weights:
                # Build per-sample weight vector from edge_type_loss_weights.
                # Agreement label class k corresponds to EDGE_TYPE_ORDER[k-1];
                # class 0 (correct) gets weight 1.0.
                sample_weights = torch.ones_like(agr_labels_flat, dtype=torch.float)
                for edge_type, w in self.edge_type_loss_weights.items():
                    # Find the label class for this edge type (1-indexed)
                    try:
                        cls_idx = EDGE_TYPE_ORDER.index(edge_type) + 1
                    except ValueError:
                        logger.warning("Unknown edge type in loss weights: %s", edge_type)
                        continue
                    sample_weights[agr_labels_flat == cls_idx] = w
                loss_fn_unreduced = nn.CrossEntropyLoss(reduction="none")
                per_token_loss = loss_fn_unreduced(agr_logits_flat, agr_labels_flat)
                agreement_loss = (per_token_loss * sample_weights).mean()
            else:
                agreement_loss = self.agreement_loss_fn(agr_logits_flat, agr_labels_flat)

            total_loss = total_loss + self.agreement_loss_weight * agreement_loss
        
        return {
            "loss": total_loss,
            "logits": outputs.logits,
            "agreement_logits": agreement_logits,
        }
    
    def correct(self, text: str, morph_features: Optional[torch.Tensor] = None,
                agreement_mask: Optional[torch.Tensor] = None,
                num_beams: int = 4) -> str:
        """Correct a single sentence."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True,
        )
        
        device = next(self.backbone.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Run encoder with morphological feature integration
            encoder_outputs = self.backbone.encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            hidden_states = encoder_outputs.last_hidden_state
            
            if morph_features is not None:
                morph_features = morph_features.to(device)
                hidden_states = self._integrate_morph_features(hidden_states, morph_features)
            
            # Apply agreement graph bias if provided
            if agreement_mask is not None:
                agreement_mask = agreement_mask.to(device)
                agr_bias = self._build_agreement_bias(
                    agreement_mask, hidden_states.size(1)
                )
                gate = torch.sigmoid(agr_bias.squeeze(1).mean(dim=-1, keepdim=True))
                hidden_states = hidden_states * (1.0 + gate)
            
            encoder_outputs.last_hidden_state = hidden_states
            
            outputs = self.backbone.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=inputs["attention_mask"],
                max_length=self.max_length,
                num_beams=num_beams,
                early_stopping=True,
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
