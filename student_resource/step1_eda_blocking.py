"""Step 1: explore data + measure blocking recall.
Run from student_resource/:  python step1_eda_blocking.py
"""
import re
import numpy as np
import pandas as pd
from unidecode import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer

TR = "dataset/train/"
KS = [10, 20, 30, 50]
KMAX = max(KS)

def rd(p):
    return pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False)

def norm(s):
    s = unidecode(str(s)).lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

s1, s2, s3 = (rd(f"{TR}train_source{i}.tsv") for i in (1, 2, 3))
gt = rd(TR + "train_ground_truth.tsv")

# ---------- EDA ----------
print("rows:", len(s1), len(s2), len(s3))
print(s1["country"].value_counts().to_dict())
gt["ids"] = gt["matched_entity_ids"].apply(lambda x: [i for i in x.split(",") if i])
gt["n"] = gt["ids"].apply(len)
print("singletons: %.1f%%" % (100 * (gt["n"] == 0).mean()),
      "| avg matches: %.2f" % gt["n"].mean(), "| max:", gt["n"].max())

rec = pd.concat([s1, s2, s3]).set_index("entity_id")
print("\n--- sample matched groups ---")
for _, r in gt[gt["n"] > 1].sample(5, random_state=0).iterrows():
    print("\nS1:", rec.loc[r["source1_entity_id"], ["business_name", "business_address", "country"]].tolist())
    for i in r["ids"]:
        print("  ", i, rec.loc[i, ["business_name", "business_address", "country"]].tolist())

# same-country check on true pairs (do NOT hard-code country rules; just informative)
same = [rec.loc[a, "country"] == rec.loc[b, "country"]
        for a, ids in zip(gt["source1_entity_id"], gt["ids"]) for b in ids]
print("\ntrue pairs with same country label: %.1f%%" % (100 * np.mean(same)))

# ---------- Blocking recall ----------
def text(df):
    return (df["business_name"].map(norm) + " " + df["business_address"].map(norm)).tolist()

vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
vec.fit(text(s1) + text(s2) + text(s3))
X1 = vec.transform(text(s1))
tops = {}
for name, s in (("S2", s2), ("S3", s3)):
    Xs = vec.transform(text(s))
    ids = s["entity_id"].values
    k = min(KMAX, len(s))
    out = []
    for a in range(0, X1.shape[0], 500):
        sim = (X1[a:a + 500] @ Xs.T).toarray()
        idx = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(sim, idx, 1), axis=1)
        out.append(ids[np.take_along_axis(idx, order, 1)])
    tops[name] = np.vstack(out)

s1_ids = s1["entity_id"].tolist()
gtd = dict(zip(gt["source1_entity_id"], gt["ids"]))
print("\n--- blocking recall (TF-IDF char 3-5) ---")
for K in KS:
    hit = tot = 0
    for row, sid in enumerate(s1_ids):
        cand = set(tops["S2"][row][:K]) | set(tops["S3"][row][:K])
        true = gtd.get(sid, [])
        tot += len(true)
        hit += sum(t in cand for t in true)
    print(f"K={K:>3}  recall={hit / max(tot, 1):.4f}  avg candidates/S1={2 * K}")