
import argparse
import sys
import pandas as pd

sys.path.insert(0, "code/business_entity_resolution/src")
from blocking import generate_candidates

TR = "dataset/train/"

def load(fn):
    return pd.read_csv(TR + fn, sep="\t", dtype=str, keep_default_na=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-frac", type=float, default=1.0,
                     help="fraction of S1 entities to test on, e.g. 0.1 for a 10%% "
                          "sample -- use this if a full sweep is too slow; the "
                          "winning topn should still transfer to the full run")
    args = ap.parse_args()

    print("loading train data ...")
    s1, s2, s3 = load("train_source1.tsv"), load("train_source2.tsv"), load("train_source3.tsv")
    gt = load("train_ground_truth.tsv")
    if args.sample_frac < 1.0:
        s1 = s1.sample(frac=args.sample_frac, random_state=0).reset_index(drop=True)
        gt = gt[gt["source1_entity_id"].isin(s1["entity_id"])].reset_index(drop=True)
        print(f"sampled down to {len(s1)} S1 entities ({args.sample_frac:.0%})")
    gt["ids"] = gt["matched_entity_ids"].apply(lambda x: [i for i in x.split(",") if i])
    total_true = sum(len(x) for x in gt["ids"])
    pool = pd.concat([s2, s3], ignore_index=True)

    print(f"{total_true} true matches to find across {len(s1)} S1 entities\n")
    print(f"{'topn':>6}  {'recall':>8}  {'avg_cand/S1':>12}  {'total_candidate_rows':>20}")
    for topn in [100, 50, 30, 20, 15, 10]:
        res = generate_candidates(s1, pool, cap=2500, cap_num=20000, topn=topn, verbose=False)
        res_map = res.groupby("source1_entity_id")["candidate_entity_id"].apply(set).to_dict()
        found = sum(i in res_map.get(sid, set())
                    for sid, ids in zip(gt["source1_entity_id"], gt["ids"]) for i in ids)
        avg_cand = len(res) / len(s1)
        print(f"{topn:>6}  {found/total_true:>8.4f}  {avg_cand:>12.1f}  {len(res):>20}")

if __name__ == "__main__":
    main()
