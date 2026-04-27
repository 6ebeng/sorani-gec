import sys
import re

content = open('src/errors/pipeline.py', 'r', encoding='utf-8').read()

new_loop = '''import itertools
        from tqdm import tqdm
        
        target_corrupted = int(target_pairs * corruption_ratio)
        target_clean = target_pairs - target_corrupted
        
        self.rng.shuffle(sentences)
        iter_sentences = itertools.cycle(enumerate(sentences))
        
        with tqdm(total=target_pairs, desc="Generating errors") as pbar:
            while len(results) < target_pairs:
                idx, sentence = next(iter_sentences)
                source_id = str(idx)
                
                if stats["corrupted"] < target_corrupted and stats["clean_pairs"] < target_clean:
                    want_corrupt = self.rng.random() < corruption_ratio
                elif stats["corrupted"] < target_corrupted:
                    want_corrupt = True
                else:
                    want_corrupt = False
                    
                if want_corrupt:
                    result = self.process_sentence(sentence)
                    result.source_id = source_id
                    if result.has_errors:
                        # CRIT-6: Validate that injected errors aren't valid words
                        if validator is not None:
                            orig_words = set(result.original.split())
                            corr_words = set(result.corrupted.split())
                            new_tokens = corr_words - orig_words
                            if new_tokens and all(validator.is_correct(w) for w in new_tokens):
                                stats["validation_rejected"] += 1
                                continue  # Reject, try another sentence
                        
                        stats["corrupted"] += 1
                        for err in result.errors:
                            stats["errors_by_type"][err.error_type] = stats["errors_by_type"].get(err.error_type, 0) + 1
                        results.append(result)
                        pbar.update(1)
                    else:
                        continue # Failed to add error, try next sentence (drops source==target)
                else:
                    stats["clean_pairs"] += 1
                    result = ErrorResult(
                        original=sentence, corrupted=sentence, errors=[],
                        source_id=source_id,
                    )
                    results.append(result)
                    pbar.update(1)
        
        self.rng.shuffle(results)
'''

start_idx = content.find('for idx, sentence in enumerate(tqdm(sentences, desc="Generating errors")):')
end_idx = content.find('# Save outputs', start_idx)

if start_idx != -1 and end_idx != -1:
    before = content[:start_idx]
    after = content[end_idx:]
    with open('src/errors/pipeline.py', 'w', encoding='utf-8') as f:
        f.write(before + new_loop + '        ' + after)
    print('Patched pipeline.py successfully')
else:
    print('Failed to find loop boundaries')
