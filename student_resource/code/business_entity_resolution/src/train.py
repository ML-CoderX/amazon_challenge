"""Train the pair classifier: GroupKFold by source1_entity_id, then tune a
global probability threshold for macro F0.5 using greedy one-to-one
assignment (each S2/S3 record used by at most one S1 entity — S1 is the
deduplicated reference source, so this should mainly help precision).

Usage (from student_resource/):
    python code/business_entity_resolution/src/train.py
"""
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

sys.path.insert(0, "code/business_entity_resolution/src")
from blocking import generate_candidates
from features import compute_features, _lookup
from metrics import macro_f05

TR = "dataset/train/"
FEATURES = ["name_ratio", "name_token_sort_ratio", "addr_ratio", "name_jaccard",
            "addr_jaccard", "num_overlap", "num_s1_count", "num_cand_count",
            "name_len_diff", "is_s2", "score", "cand_rank",
            "score_margin_to_top", "n_candidates_for_s1"]


def load(fn):
    return pd.read_csv(TR + fn, sep="\t", dtype=str, keep_default_na=False)


def greedy_one_to_one(df, prob_col="prob"):
    """Each candidate id (S2/S3 record) assigned to at most one S1 entity —
    exploits 'Source1 is deduplicated', should raise precision under F0.5."""
    d = df.sort_values(prob_col, ascending=False)
    used_cand, used_s1 = set(), set()
    keep = []
    for sid, cid, p in zip(d["source1_entity_id"], d["candidate_entity_id"], d[prob_col]):
        if cid in used_cand:
            continue
        keep.append((sid, cid, p))
        used_cand.add(cid)
    return pd.DataFrame(keep, columns=["source1_entity_id", "candidate_entity_id", prob_col])


def main():
    print("loading data ...")
    s1, s2, s3 = load("train_source1.tsv"), load("train_source2.tsv"), load("train_source3.tsv")
    gt = load("train_ground_truth.tsv")
    gt["ids"] = gt["matched_entity_ids"].apply(lambda x: [i for i in x.split(",") if i])
    true_matches = dict(zip(gt["source1_entity_id"], gt["ids"].apply(set)))
    pool = pd.concat([s2, s3], ignore_index=True)

    print("blocking (this is the slow step; expect it to take a while on full data) ...")
    cand = generate_candidates(s1, pool, cap=2500, cap_num=20000, topn=50)
    print(f"candidates: {len(cand)} rows for {cand['source1_entity_id'].nunique()} S1 entities")

    print("computing features ...")
    feat = compute_features(cand, _lookup(s1), _lookup(pool))
    feat["y"] = [
        1 if cid in true_matches.get(sid, set()) else 0
        for sid, cid in zip(feat["source1_entity_id"], feat["candidate_entity_id"])
    ]
    print(f"positive rate in candidates: {feat['y'].mean():.4f}  (blocking recall check: "
          f"{feat['y'].sum()} / {sum(len(v) for v in true_matches.values())} true matches present)")

    groups = feat["source1_entity_id"]
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(feat))
    models = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(feat, feat["y"], groups)):
        tr, va = feat.iloc[tr_idx], feat.iloc[va_idx]
        model = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31,
                                    subsample=0.8, colsample_bytree=0.8, random_state=fold)
        model.fit(tr[FEATURES], tr["y"],
                  eval_set=[(va[FEATURES], va["y"])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[va_idx] = model.predict_proba(va[FEATURES])[:, 1]
        models.append(model)
        print(f"  fold {fold}: best_iter={model.best_iteration_}")

    feat["prob"] = oof

    print("tuning threshold for macro F0.5 (with greedy one-to-one assignment) ...")
    all_s1_ids = s1["entity_id"].tolist()
    best_t, best_f = 0.5, -1
    for t in np.arange(0.1, 0.96, 0.05):
        above = feat[feat["prob"] >= t]
        assigned = greedy_one_to_one(above)
        pred = (assigned.groupby("source1_entity_id")["candidate_entity_id"]
                .apply(set).to_dict())
        f = macro_f05(true_matches, pred, all_s1_ids)
        print(f"  threshold={t:.2f}  macro_F0.5={f:.4f}")
        if f > best_f:
            best_f, best_t = f, t
    print(f"\nBEST threshold={best_t:.2f}  OOF macro_F0.5={best_f:.4f}")

    # refit on all data at this threshold's feature set, save models + threshold
    import pickle
    with open("code/business_entity_resolution/src/model.pkl", "wb") as f:
        pickle.dump({"models": models, "threshold": best_t, "features": FEATURES}, f)
    print("saved model.pkl")


if __name__ == "__main__":
    main()