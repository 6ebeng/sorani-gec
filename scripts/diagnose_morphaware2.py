"""Deep diagnostic: check what generate() actually receives from morphaware correct_batch."""
import sys
import torch
sys.path.insert(0, "src")
from src.model.morphology_aware import MorphologyAwareGEC, EDGE_TYPE_ORDER
from src.morphology.analyzer import MorphologicalAnalyzer
from src.morphology.features import FeatureExtractor
from src.morphology.lexicon import SoraniLexicon

device = torch.device("cuda")
lexicon = SoraniLexicon()
analyzer = MorphologicalAnalyzer(use_klpt=False, ahmadi_lexicon=lexicon)
feature_vocab = analyzer.build_feature_vocabulary()
fe = FeatureExtractor(analyzer)

m = MorphologyAwareGEC(
    model_name="google/byt5-small",
    feature_vocab_size=max(len(feature_vocab), 1),
    num_agreement_types=len(EDGE_TYPE_ORDER) + 1,
).to(device)
ckpt = torch.load("results/phase_d/morphaware_seed42/best_model.pt", map_location=device, weights_only=True)
m.load_state_dict(ckpt["model_state_dict"])
m.eval()

src = "زمانی کوردی کرا بەزمانی ڕەسمی لەسەرجەم دام و دەزگاکانی کۆماردا."

# Replicate correct_batch internals manually
morph_t, agr_t = m._build_inference_tensors(src, analyzer, fe)
morph_t = morph_t.to(device)
agr_t = agr_t.to(device)

inputs = m.tokenizer(
    [src], return_tensors="pt", max_length=m.max_length,
    truncation=True, padding="max_length",
)
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    enc_orig = m.backbone.encoder(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
    )
    hs_orig = enc_orig.last_hidden_state.clone()
    print(f"hs_orig: shape={tuple(hs_orig.shape)} mean={hs_orig.mean():.4f} std={hs_orig.std():.4f} hasnan={torch.isnan(hs_orig).any()}")

    hs_morph = m._integrate_morph_features(enc_orig.last_hidden_state.clone(), morph_t)
    print(f"hs_morph: mean={hs_morph.mean():.4f} std={hs_morph.std():.4f} hasnan={torch.isnan(hs_morph).any()}")

    agr_bias = m._build_agreement_bias(agr_t, hs_morph.size(1))
    gate = torch.sigmoid(agr_bias.squeeze(1).max(dim=-1, keepdim=True).values)
    print(f"gate: mean={gate.mean():.4f} max={gate.max():.4f} min={gate.min():.4f}")
    agr_res = m.decoder_agr_proj(hs_morph)
    hs_final = hs_morph + gate * agr_res
    print(f"hs_final: mean={hs_final.mean():.4f} std={hs_final.std():.4f} hasnan={torch.isnan(hs_final).any()}")

    # Generate using FULL morphaware path
    enc_orig.last_hidden_state = hs_final
    out_full = m.backbone.generate(
        encoder_outputs=enc_orig,
        attention_mask=inputs["attention_mask"],
        max_length=m.max_length,
        num_beams=4,
        early_stopping=True,
    )
    print(f"\nFULL morphaware path output ids: {out_full[0][:20].tolist()}")
    print(f"FULL morphaware decoded: {repr(m.tokenizer.decode(out_full[0], skip_special_tokens=True)[:100])}")

    # Generate using ORIGINAL encoder output (no morph mods)
    enc_orig2 = m.backbone.encoder(
        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
    )
    out_no_morph = m.backbone.generate(
        encoder_outputs=enc_orig2,
        attention_mask=inputs["attention_mask"],
        max_length=m.max_length,
        num_beams=4,
        early_stopping=True,
    )
    print(f"\nNO-morph path output ids: {out_no_morph[0][:20].tolist()}")
    print(f"NO-morph decoded: {repr(m.tokenizer.decode(out_no_morph[0], skip_special_tokens=True)[:100])}")

    # Generate using normal backbone.generate from input_ids
    out_native = m.backbone.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=m.max_length,
        num_beams=4,
        early_stopping=True,
    )
    print(f"\nNATIVE backbone.generate output ids: {out_native[0][:20].tolist()}")
    print(f"NATIVE decoded: {repr(m.tokenizer.decode(out_native[0], skip_special_tokens=True)[:100])}")
