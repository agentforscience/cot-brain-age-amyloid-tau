#!/bin/bash
OK=0; FAIL=0
while IFS='|' read -r ID NAME; do
  [ -z "$ID" ] && continue
  OUT="papers/${ID}_${NAME}.pdf"
  if [ -s "$OUT" ]; then echo "SKIP $OUT"; OK=$((OK+1)); continue; fi
  curl -sL --max-time 90 -A "Mozilla/5.0 (research; chicagohailab@gmail.com)" "https://arxiv.org/pdf/${ID}" -o "$OUT"
  if [ -s "$OUT" ] && head -c 5 "$OUT" | grep -q "%PDF"; then
    echo "OK   $(du -h "$OUT" | cut -f1) $OUT"; OK=$((OK+1))
  else
    echo "FAIL $ID"; rm -f "$OUT"; FAIL=$((FAIL+1))
  fi
  sleep 2
done < paper_search_results/selected.txt
echo "=== OK=$OK FAIL=$FAIL ==="
