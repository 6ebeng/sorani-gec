#!/bin/bash
# Poll ablation progress until all 12 experiments complete or process dies.
START=$(date +%s)
for i in $(seq 1 80); do
  DONE=$(find /workspace/sorani-gec/results/ablation -name metrics.json 2>/dev/null | wc -l)
  ALIVE=$(pgrep -f 08_ablation.py | wc -l)
  ELAPSED=$(( $(date +%s) - START ))
  echo "[poll $i +${ELAPSED}s] done=$DONE/12 alive=$ALIVE"
  tail -2 /workspace/sorani-gec/results/ablation.log 2>/dev/null | grep -E 'Ablation|Feature|Data|weight|Epoch|curriculum' | tail -1
  if [ "$ALIVE" -eq 0 ]; then
    echo "FINAL_STOP at +${ELAPSED}s"
    break
  fi
  if [ "$DONE" -ge 12 ]; then
    echo "ALL_DONE at +${ELAPSED}s"
    break
  fi
  sleep 90
done
echo "---FINAL STATE---"
find /workspace/sorani-gec/results/ablation -name metrics.json | sort
