"""Stage 1 baseline: no ML, just the blocking score. Takes each S1's best-
scoring candidate (and any other candidate within CLOSE of the top score),
above a minimum floor, no one-to-one assignment. Good for a first, honest
submission while the real classifier trains.

Usage (from student_resource/):
    python code/business_entity_resolution/src/baseline_submit.py
"""
import sys
import pandas as pd

sys.path.insert(0, "code/business_entity_resolution/src")
from blocking import generate_candidates, to_wide_tsv

TE = "dataset/test/"
MIN_SCORE = 3.0   # floor -- tune after seeing the score distribution

def load(fn):
    return pd.read_csv(TE + fn, sep="\t", dtype=str, keep_default_na=False)

def main():
    s1, s2, s3 = load("test_source1.tsv"), load("test_source2.tsv"), load("test_source3.tsv")
    pool = pd.concat([s2, s3], ignore_index=True)

    print("blocking ...")
    cand = generate_candidates(s1, pool, cap=2500, cap_num=20000, topn=50)

    print("writing candidate_pairs.tsv ...")
    to_wide_tsv(cand, s1["entity_id"].tolist(), "output/candidate_pairs.tsv")

    print(f"score distribution:\n{cand['score'].describe()}")

    best = cand[cand["score"] >= MIN_SCORE].sort_values(
        ["source1_entity_id", "score"], ascending=[True, False])
    top1 = best.groupby("source1_entity_id").head(1)   # just the single best match

    print("writing matching_results.tsv (baseline: top-1 candidate above floor) ...")
    grp = top1.set_index("source1_entity_id")["candidate_entity_id"]
    grp = grp.reindex(s1["entity_id"].tolist(), fill_value="")
    grp.rename_axis("source1_entity_id").reset_index(name="matched_entity_ids").to_csv(
        "output/matching_results.tsv", sep="\t", index=False)
    print("done. Now run utils/validate_submission.py before uploading.")

if __name__ == "__main__":
    main()