"""
Agreement Graph Data Structures

Contains the AgreementEdge dataclass and AgreementGraph class that model
agreement dependencies in Sorani Kurdish sentences.

Implements Slevanayi's (2001) two-law agreement system:
  Law 1 — Subject-verb agreement (nominative-accusative alignment)
  Law 2 — Object-verb agreement (ergative alignment, past transitive)
"""

import logging
from dataclasses import dataclass

from .analyzer import MorphFeatures

logger = logging.getLogger(__name__)

# Known edge types — fixed ordering for model input tensor.
# New types discovered at runtime are appended after these.
EDGE_TYPE_ORDER: list[str] = [
    "subject_verb",
    "object_verb_ergative",
    "object_verb_ergative_zero",
    "agent_non_agreeing",
    "clitic_agent",
    "clitic_patient",
    "noun_det",
    "adjective_invariant",
    "quantifier_verb",
    "measure_word_verb",
    "mass_noun_no_agreement",
    "collective_singular",
    "collective_plural",
    "dem_det_noun",
    "dem_proform_verb",
    "vocative_imperative",
    "relative_clause",
    "adverb_verb_tense",
    "possessive_no_agreement",
    "oblique_no_agreement",
    "conditional_agreement",
    "pro_drop_agreement",
    "passive_subject_verb",
    "backward_subject_verb",
]


@dataclass
class AgreementEdge:
    """An agreement dependency between two tokens."""
    source_idx: int
    target_idx: int
    agreement_type: str   # e.g., "subject_verb", "object_verb", "noun_det", "clitic"
    features: list[str]   # which features must agree: ["person", "number"]
    law: str = ""         # "law1" (subject-verb) or "law2" (object-verb/ergative)


# Role labels assigned to the *source* token of each edge type.
# Used by token_roles() to derive per-token grammatical roles.
EDGE_ROLE_SOURCE: dict[str, str] = {
    "subject_verb": "subject",
    "passive_subject_verb": "subject",
    "backward_subject_verb": "subject",
    "object_verb_ergative": "object",
    "object_verb_ergative_zero": "object",
    "agent_non_agreeing": "subject",
    "clitic_agent": "clitic_subject",
    "clitic_patient": "clitic_object",
    "noun_det": "noun",
    "det_noun": "determiner",
    "adj_noun": "adjective",
    "adjective_invariant": "adjective",
    "dem_det_noun": "determiner",
    "dem_proform_verb": "demonstrative",
    "quantifier_verb": "quantifier",
    "measure_word_verb": "measure",
    "possessive_no_agreement": "possessor",
    "oblique_no_agreement": "oblique",
    "pro_drop_agreement": "pro_subject",
    "vocative_imperative": "vocative",
    "relative_clause": "relative",
    "conditional_agreement": "conditional",
    "collective_singular": "collective",
    "collective_plural": "collective",
    "mass_noun_no_agreement": "mass_noun",
}

# Role labels assigned to the *target* token of each edge type.
EDGE_ROLE_TARGET: dict[str, str] = {
    "subject_verb": "verb",
    "passive_subject_verb": "verb",
    "backward_subject_verb": "verb",
    "object_verb_ergative": "verb",
    "object_verb_ergative_zero": "verb",
    "clitic_agent": "verb",
    "clitic_patient": "verb",
    "noun_det": "determiner",
    "det_noun": "noun",
    "adj_noun": "noun",
    "dem_det_noun": "noun",
    "dem_proform_verb": "verb",
    "quantifier_verb": "verb",
    "measure_word_verb": "verb",
    "vocative_imperative": "verb",
}


class AgreementGraph:
    """Graph of agreement dependencies in a sentence.

    Implements Slevanayi's (2001) two-law agreement system:
      Law 1 — Subject-verb (nominative-accusative alignment)
      Law 2 — Object-verb  (ergative alignment, past transitive only)
    """

    # Valid feature names for edge validation
    _VALID_FEATURES = frozenset({
        "person", "number", "tense", "aspect", "case",
        "definiteness", "transitivity",
    })

    def __init__(self, tokens: list[str], features: list[MorphFeatures]):
        self.tokens = tokens
        self.features = features
        self.edges: list[AgreementEdge] = []
        self._edge_keys: set[tuple[int, int, str]] = set()

    def __len__(self) -> int:
        """Return the number of edges in the graph."""
        return len(self.edges)

    def add_edge(self, source: int, target: int,
                 agreement_type: str, features: list[str],
                 law: str = ""):
        # Deduplicate: skip if same source-target-type already exists
        key = (source, target, agreement_type)
        if key in self._edge_keys:
            return
        # Validate feature names
        for f in features:
            if f not in self._VALID_FEATURES:
                logger.warning(
                    "Unknown feature '%s' in edge %s→%s (%s)",
                    f, source, target, agreement_type,
                )
        self._edge_keys.add(key)
        self.edges.append(AgreementEdge(
            source_idx=source,
            target_idx=target,
            agreement_type=agreement_type,
            features=features,
            law=law,
        ))

    def check_agreement(self) -> list[dict]:
        """Check all agreement edges and return violations.
        
        Enforces Law-specific semantics:
          Law 1: subject → verb (person + number must match)
          Law 2: only the object controls verb agreement;
                 agent_non_agreeing edges are informational and skip checking.
          Zero-agreement edges (empty features) are always satisfied.
        """
        violations = []
        for edge in self.edges:
            # Informational edges with no features never violate
            if not edge.features:
                continue
            # Agent non-agreeing edges are traceability-only
            if edge.agreement_type == "agent_non_agreeing":
                continue
            
            src_feat = self.features[edge.source_idx]
            tgt_feat = self.features[edge.target_idx]

            for feat_name in edge.features:
                src_val = getattr(src_feat, feat_name, "")
                tgt_val = getattr(tgt_feat, feat_name, "")

                if src_val and tgt_val and src_val != tgt_val:
                    violations.append({
                        "type": edge.agreement_type,
                        "feature": feat_name,
                        "law": edge.law,
                        "source": (edge.source_idx, self.tokens[edge.source_idx], src_val),
                        "target": (edge.target_idx, self.tokens[edge.target_idx], tgt_val),
                    })

        return violations

    def to_adjacency_matrix(self) -> list[list[int]]:
        """Convert graph to binary adjacency matrix for attention masking."""
        n = len(self.tokens)
        matrix = [[0] * n for _ in range(n)]
        for edge in self.edges:
            matrix[edge.source_idx][edge.target_idx] = 1
        return matrix

    def to_typed_adjacency_matrices(self) -> dict[str, list[list[int]]]:
        """Convert graph to per-edge-type adjacency matrices.

        Returns a dict mapping each agreement_type (e.g. 'subject_verb',
        'clitic_agent') to its own N×N binary matrix. This lets the model
        learn type-specific attention patterns rather than collapsing all
        edges into a single binary mask.
        """
        n = len(self.tokens)
        types: dict[str, list[list[int]]] = {}
        for edge in self.edges:
            t = edge.agreement_type
            if t not in types:
                types[t] = [[0] * n for _ in range(n)]
            types[t][edge.source_idx][edge.target_idx] = 1
        return types

    def edge_type_counts(self) -> dict[str, int]:
        """Count edges by agreement type (useful for diagnostics)."""
        counts: dict[str, int] = {}
        for edge in self.edges:
            counts[edge.agreement_type] = counts.get(edge.agreement_type, 0) + 1
        return counts

    def to_typed_stacked_matrix(self) -> tuple[list[list[list[int]]], list[str]]:
        """Stack per-type adjacency matrices into [num_types, N, N].

        Returns (matrices, type_names) where matrices[k] is the N×N
        binary matrix for type_names[k].  Types follow EDGE_TYPE_ORDER;
        any runtime-only types are appended after the predefined list.
        """
        typed = self.to_typed_adjacency_matrices()
        type_names: list[str] = []
        matrices: list[list[list[int]]] = []
        for t in EDGE_TYPE_ORDER:
            if t in typed:
                matrices.append(typed[t])
                type_names.append(t)
        # Append any types not in the predefined order
        for t in typed:
            if t not in EDGE_TYPE_ORDER:
                matrices.append(typed[t])
                type_names.append(t)
        return matrices, type_names

    def token_roles(self) -> list[set[str]]:
        """Derive grammatical role labels for each token from edges.

        Returns a list (one entry per token) of sets of English role
        strings such as ``{"subject"}``, ``{"verb"}``, or ``set()``
        for tokens not participating in any agreement relation.
        """
        n = len(self.tokens)
        roles: list[set[str]] = [set() for _ in range(n)]
        for e in self.edges:
            src_role = EDGE_ROLE_SOURCE.get(e.agreement_type)
            tgt_role = EDGE_ROLE_TARGET.get(e.agreement_type)
            if src_role and e.source_idx < n:
                roles[e.source_idx].add(src_role)
            if tgt_role and e.target_idx < n:
                roles[e.target_idx].add(tgt_role)
        return roles
