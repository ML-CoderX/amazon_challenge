"""Generate the final submission files from a trained model.
Usage (from student_resource/):
    python code/business_entity_resolution/src/predict.py
Writes output/matching_results.tsv and output/candidate_pairs.tsv.
"""
import pickle
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "code/business_entity_resolution/src")
from blocking import generate_candidates, to_wide_tsv
from features import compute_features, _lookup
from train import greedy_one_to_one  # reuse the same assignment logic

TE = "dataset/test/"


def load(fn):
    return pd.read_csv(TE + fn, sep="\t", dtype=str, keep_default_na=False)


def main():
    with open("code/business_entity_resolution/src/model.pkl", "rb") as f:
        saved = pickle.load(f)
    models, threshold, feat_cols = saved["models"], saved["threshold"], saved["features"]

    print("loading test data ...")
    s1, s2, s3 = load("test_source1.tsv"), load("test_source2.tsv"), load("test_source3.tsv")
    pool = pd.concat([s2, s3], ignore_index=True)

    print("blocking on test data ...")
    cand = generate_candidates(s1, pool, cap=2500, cap_num=20000, topn=50)

    print("writing candidate_pairs.tsv (blocking output, unscored) ...")
    to_wide_tsv(cand.rename(columns={"candidate_entity_id": "candidate_entity_id"}),
                s1["entity_id"].tolist(), "output/candidate_pairs.tsv")

    print("computing features ...")
    feat = compute_features(cand, _lookup(s1), _lookup(pool))

    print("scoring with ensembled fold models ...")
    probs = np.mean([m.predict_proba(feat[feat_cols])[:, 1] for m in models], axis=0)
    feat["prob"] = probs

    print(f"applying threshold={threshold:.2f} + one-to-one assignment ...")
    above = feat[feat["prob"] >= threshold]
    assigned = greedy_one_to_one(above)

    print("writing matching_results.tsv ...")
    # NOTE: matching_results ids must be a SUBSET of candidate_pairs ids (validator checks this) —
    # since assigned comes directly from `cand`, that's automatically satisfied.
    grp = (assigned.groupby("source1_entity_id")["candidate_entity_id"]
           .apply(lambda s: ",".join(s)))
    grp = grp.reindex(s1["entity_id"].tolist(), fill_value="")
    grp.rename_axis("source1_entity_id").reset_index(name="matched_entity_ids").to_csv(
        "output/matching_results.tsv", sep="\t", index=False)
    print("done.")


if __name__ == "__main__":
    main()