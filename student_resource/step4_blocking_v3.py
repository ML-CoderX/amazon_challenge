"""Step 3: blocking v2 (address-rarity keys, idf-ranked candidates).
"""
import argparse, itertools, os, re, zlib
from collections import Counter
import numpy as np
import pandas as pd
from unidecode import unidecode

ap = argparse.ArgumentParser()
ap.add_argument("--pct", type=int, default=3, help="percent of data kept for dev sample")
ap.add_argument("--cap", type=int, default=100, help="drop blocking keys shared by more than this many records")
ap.add_argument("--same-country", action="store_true", help="only block within identical country labels")
args = ap.parse_args()
TR = "dataset/train/"

def bucket(i):
    return zlib.crc32(i.encode()) % 100

def load(fn, keep_ids=None):
    parts = []
    for ch in pd.read_csv(TR + fn, sep="\t", dtype=str, keep_default_na=False, chunksize=500_000):
        m = ch["entity_id"].map(bucket) < args.pct
        if keep_ids is not None:
            m |= ch["entity_id"].isin(keep_ids)
        parts.append(ch[m])
    return pd.concat(parts, ignore_index=True)

# country labels in the test files (is the label consistent, any new values?)
for i in (1, 2, 3):
    p = f"dataset/test/test_source{i}.tsv"
    if os.path.exists(p):
        c = pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, usecols=["country"])["country"]
        print(f"test S{i} countries:", c.value_counts().to_dict())

# ---------- sample ----------
gt = pd.read_csv(TR + "train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
gt = gt[gt["source1_entity_id"].map(bucket) < args.pct].reset_index(drop=True)
gt["ids"] = gt["matched_entity_ids"].apply(lambda x: [i for i in x.split(",") if i])
keep = {i for l in gt["ids"] for i in l}
s1 = load("train_source1.tsv")
pool = pd.concat([load("train_source2.tsv", keep), load("train_source3.tsv", keep)], ignore_index=True)
print(f"sample: S1={len(s1)} pool={len(pool)}")

# ---------- normalisation ----------
STOP = set("pvt private ltd limited llp llc inc incorporated corp corporation co company "
           "and the of m s ms sarl sas lp plc pty".split())

def norm(s):
    s = unidecode(str(s)).lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def cn(t):
    return re.sub(r"(.)\1+", r"\1", t)

def skel(t):
    return cn(re.sub(r"[aeiouyhw]", "", t))

def addr_toks(addr):
    return {cn(t)[:5] for t in norm(addr).split() if t.isalnum() and len(t) >= 4}

# document frequency of address tokens (over S1 + pool)
DF = Counter()
for a in itertools.chain(s1["business_address"].values, pool["business_address"].values):
    DF.update(addr_toks(a))

def num_toks(addr):
    # digit runs len>=2 from the raw (unnormalised) address: house/plot/flat numbers survive
    # truncation even when everything else is stripped down to city/state.
    return {t for t in re.findall(r"\d{2,}", str(addr))}

def keys(name, addr):
    nt = [t for t in norm(name).split() if t not in STOP and len(t) > 1]
    at = norm(addr).split()
    ks = set()
    P = sorted({cn(t)[:4] for t in nt[:4] if len(t) >= 3})
    S = sorted({skel(t)[:3] for t in nt[:4] if len(skel(t)) >= 2})
    for p in P:
        ks.add("p:" + p)
    for a, b in itertools.combinations(P, 2):
        ks.add(f"pp:{a}|{b}")
    for a, b in itertools.combinations(S, 2):
        ks.add(f"ss:{a}|{b}")
    # address keys from the RAREST tokens (df>=2 so one-off typos are ignored)
    good = sorted((t for t in addr_toks(addr) if DF[t] >= 2), key=lambda t: (DF[t], t))
    R = good[:4]
    for t in R[:2]:
        ks.add("r:" + t)
    for a, b in itertools.combinations(sorted(R), 2):
        ks.add(f"rr:{a}|{b}")
    nums = [t for t in at if any(c.isdigit() for c in t)][:3]
    for n_ in nums:
        for t in R[:3]:
            ks.add(f"a:{n_}|{t}")
    for p in P[:2]:
        for t in R[:3]:
            ks.add(f"pa:{p}|{t}")
    # NEW: bare number key -- last resort when the address is truncated to just
    # city/state + one number, and the name is non-Latin (no P/S tokens match).
    for nnum in num_toks(addr):
        ks.add("n:" + nnum)
    return ks

def key_frame(df):
    ii, kk = [], []
    ctry = df["country"].values
    for i, (n, a) in enumerate(zip(df["business_name"].values, df["business_address"].values)):
        for k in keys(n, a):
            # bare number keys ("n:...") are weak on their own -> always scope by country
            # so a shared 3-digit number doesn't fan out across two different countries.
            scoped = k.startswith("n:") or args.same_country
            ii.append(i)
            kk.append(k + ("#" + ctry[i] if scoped else ""))
    return pd.DataFrame({"i": ii, "k": kk})

print("building keys ...")
f1, fp = key_frame(s1), key_frame(pool)
c1, cp = f1["k"].value_counts(), fp["k"].value_counts()
ok = set(cp[cp <= args.cap].index) & set(c1[c1 <= args.cap].index)
f1, fp = f1[f1["k"].isin(ok)], fp[fp["k"].isin(ok)]
print("usable keys:", len(ok))

w = np.log((len(pool) + 1) / (cp + 1))          # rarer key => bigger weight
pairs = f1.merge(fp, on="k", suffixes=("_s1", "_p"))
pairs["w"] = pairs["k"].map(w).values
g = (pairs.groupby(["i_s1", "i_p"])["w"].sum().rename("score").reset_index()
     .sort_values(["i_s1", "score"], ascending=[True, False]))
g["rank"] = g.groupby("i_s1").cumcount()

# ---------- truth ----------
s1_idx = {e: i for i, e in enumerate(s1["entity_id"])}
p_idx = {e: i for i, e in enumerate(pool["entity_id"])}
tr = [(s1_idx[a], p_idx[b]) for a, ids in zip(gt["source1_entity_id"], gt["ids"])
      for b in ids if a in s1_idx and b in p_idx]
tr = pd.DataFrame(tr, columns=["i_s1", "i_p"]).merge(g[["i_s1", "i_p", "rank"]], how="left", on=["i_s1", "i_p"])
same = (s1["country"].values[tr["i_s1"]] == pool["country"].values[tr["i_p"]]).mean()
print(f"true pairs: {len(tr)} | same country label: {100 * same:.1f}%")

print("\n--- blocking recall v2 ---")
for N in (10, 20, 50, 100, 10**9):
    print(f"top {'all' if N == 10**9 else N:>4} candidates/S1: recall={(tr['rank'] < N).sum() / len(tr):.4f}")
per = g.groupby("i_s1").size().reindex(range(len(s1)), fill_value=0)
print(f"avg candidates/S1: {per.mean():.1f} | S1 with zero candidates: {100 * (per == 0).mean():.1f}%")

miss = tr[tr["rank"].isna()]
print(f"\n--- {min(15, len(miss))} missed true pairs (of {len(miss)}) ---")
for _, r in miss.sample(min(15, len(miss)), random_state=0).iterrows():
    a, b = s1.iloc[int(r.i_s1)], pool.iloc[int(r.i_p)]
    print(f"\nS1  : {a.business_name} | {a.business_address}")
    print(f"MISS: {b.business_name} | {b.business_address}   [{b.entity_id}]")