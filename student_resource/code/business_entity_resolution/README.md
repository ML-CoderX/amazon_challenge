# Business Entity Resolution — pipeline

## Setup
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r code/business_entity_resolution/requirements.txt
```

## Run (from student_resource/, i.e. this file's parent-parent-parent dir)
```bash
python code/business_entity_resolution/src/train.py     # blocks + trains on dataset/train/, saves model.pkl
python code/business_entity_resolution/src/predict.py   # blocks + scores dataset/test/, writes output/*.tsv
python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

## Pipeline
1. **blocking.py** — key-based candidate generation (name prefixes/skeletons,
   rare address tokens, house/plot numbers), processed one country at a time.
   `cap` / `cap_num` control how common a key can be before it's discarded as
   too generic; retune if recall drops or blocking gets too slow.
2. **features.py** — string-similarity, token-overlap, number-overlap, and
   rank/score-margin features per candidate pair.
3. **train.py** — LightGBM, GroupKFold by source1_entity_id (never split
   pairs across folds), threshold tuned directly for macro F0.5 with greedy
   one-to-one assignment (Source1 is deduplicated, so each S2/S3 record is
   used by at most one match).
4. **predict.py** — same blocking + features + ensembled fold models on the
   test set, applies the saved threshold, writes both output files.
5. **metrics.py** — exact per-entity macro F0.5 competition metric, used
   internally by train.py for threshold tuning and validation.

## Known limitations / next steps
- Cap values were tuned on a 3% dev sample; re-check blocking recall at full
  scale and raise cap/cap_num if it drops.
- No embeddings yet — a multilingual model (checked against the MIT/Apache-2.0,
  <=8B constraint) would likely help the non-Latin-name cases blocking still misses.
