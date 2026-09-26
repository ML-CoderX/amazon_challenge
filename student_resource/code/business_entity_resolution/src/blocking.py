
import itertools
import math
import os
from collections import Counter, defaultdict
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from normalize import norm, cn, skel, name_tokens, addr_tokens, num_tokens

_WORKER_ADDR_DF = None  # set once per worker process by _init_worker, avoids
                         # re-pickling the (large) address-DF dict on every chunk


def _init_worker(addr_df):
    global _WORKER_ADDR_DF
    _WORKER_ADDR_DF = addr_df


_WORKER_BUCKET_W = None  # set once per worker process by _init_bucket_worker,
                          # avoids re-pickling the (large) weight map per bucket


def _init_bucket_worker(w_dict):
    global _WORKER_BUCKET_W
    _WORKER_BUCKET_W = w_dict


def _merge_bucket(args):
    """Runs in a worker process: the expensive part of one bucket (merge +
    groupby-sum) with no shared state and no truncation -- truncation stays
    in the main process so the fold-and-truncate semantics are byte-for-byte
    identical to the sequential version, just with the merge itself computed
    in parallel across cores."""
    f1b, fpb = args
    if f1b.empty or fpb.empty:
        return pd.DataFrame(columns=["i_s1", "i_p", "score"])
    m = f1b.merge(fpb, on="k", suffixes=("_s1", "_p"))
    if m.empty:
        return pd.DataFrame(columns=["i_s1", "i_p", "score"])
    m["w"] = m["k"].map(_WORKER_BUCKET_W).values
    gb = m.groupby(["i_s1", "i_p"])["w"].sum().rename("score").reset_index()
    return gb


def _merge_bucket_local(args, w):
    """Same computation as _merge_bucket, but for the no-multiprocessing
    fallback path -- takes w directly instead of reading a worker global."""
    f1b, fpb = args
    if f1b.empty or fpb.empty:
        return pd.DataFrame(columns=["i_s1", "i_p", "score"])
    m = f1b.merge(fpb, on="k", suffixes=("_s1", "_p"))
    if m.empty:
        return pd.DataFrame(columns=["i_s1", "i_p", "score"])
    m["w"] = m["k"].map(w).values
    return m.groupby(["i_s1", "i_p"])["w"].sum().rename("score").reset_index()


def _keys_for_chunk(args):
    names, addrs = args
    ii, kk = [], []
    for i, (n, a) in enumerate(zip(names, addrs)):
        for k in build_keys(n, a, _WORKER_ADDR_DF):
            ii.append(i)
            kk.append(k)
    return ii, kk


def compute_addr_df(addresses):
    """Document frequency of address tokens, used to pick 'rare' address
    words as blocking keys (typos with df==1 are ignored)."""
    df = Counter()
    for addr in addresses:
        toks = {cn(t)[:5] for t in norm(addr).split() if t.isalnum() and len(t) >= 4}
        df.update(toks)
    return df


def build_keys(name, addr, addr_df):
    nt = name_tokens(name)
    at = addr_tokens(addr)
    ks = set()

    P = sorted({cn(t)[:4] for t in nt[:4] if len(t) >= 3})
    S = sorted({skel(t)[:3] for t in nt[:4] if len(skel(t)) >= 2})
    for p in P:
        ks.add("p:" + p)
    for a, b in itertools.combinations(P, 2):
        ks.add(f"pp:{a}|{b}")
    for a, b in itertools.combinations(S, 2):
        ks.add(f"ss:{a}|{b}")

    toks = {cn(t)[:5] for t in norm(addr).split() if t.isalnum() and len(t) >= 4}
    good = sorted((t for t in toks if addr_df.get(t, 0) >= 2), key=lambda t: (addr_df[t], t))
    R = good[:4]
    for t in R[:2]:
        ks.add("r:" + t)
    for a, b in itertools.combinations(sorted(R), 2):
        ks.add(f"rr:{a}|{b}")

    nums_alpha = [t for t in at if any(c.isdigit() for c in t)][:3]
    for n_ in nums_alpha:
        for t in R[:3]:
            ks.add(f"a:{n_}|{t}")
    for p in P[:2]:
        for t in R[:3]:
            ks.add(f"pa:{p}|{t}")

    numtoks = num_tokens(addr)
    for nnum in numtoks:
        if len(nnum) >= 3:          # 2-digit numbers alone are too common at full scale
            ks.add("n:" + nnum)     # (safe to relax again if paired with more context)
    for a, b in itertools.combinations(sorted(set(numtoks)), 2):
        ks.add(f"nn:{a}|{b}")       # a PAIR of numbers together is rare even at 2 digits
    return ks


def _key_frame(df, addr_df, n_jobs=None, chunk_size=20_000, label=""):
    """Parallel key generation -- this loop (not the later merge) is the
    actual bottleneck, since it's pure-Python work per record."""
    names = df["business_name"].values
    addrs = df["business_address"].values
    n = len(df)
    if n == 0:
        return pd.DataFrame({"i": [], "k": []})

    n_jobs = n_jobs or max(1, (os.cpu_count() or 2) - 1)
    if n < chunk_size * 2 or n_jobs <= 1:
        # small enough / single-core: skip multiprocessing overhead
        _init_worker(addr_df)   # set the global _keys_for_chunk reads from
        ii, kk = _keys_for_chunk((names, addrs))
        return pd.DataFrame({"i": ii, "k": kk})

    chunks, offsets = [], []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunks.append((names[start:end], addrs[start:end]))
        offsets.append(start)

    all_i, all_k, done = [], [], 0
    with Pool(n_jobs, initializer=_init_worker, initargs=(addr_df,)) as pool:
        for off, (ii, kk) in zip(offsets, pool.imap(_keys_for_chunk, chunks)):
            all_i.append(np.array(ii, dtype=np.int64) + off)
            all_k.extend(kk)
            done += len(ii)
            print(f"    {label} keys: {min(off + chunk_size, n)}/{n} records "
                  f"({done} keys so far)", end="\r")
    print()
    return pd.DataFrame({"i": np.concatenate(all_i) if all_i else np.array([], dtype=np.int64),
                          "k": all_k})


def _topn_chunk(args):
    """Process one contiguous S1 range using only the pool postings for the
    keys used by that range.

    This is the important performance change versus the bucket-fold version:
    every S1 entity is completed in one task, so we never keep a 25M-row
    `running` DataFrame and repeatedly group/trim it after every bucket.
    Scores are exact because all keys for an S1 chunk are joined before the
    final top-n operation.
    """
    (f1_chunk, fp_chunk, topn) = args
    if f1_chunk.empty or fp_chunk.empty:
        return pd.DataFrame(columns=["i_s1", "i_p", "score"])

    m = f1_chunk.merge(
        fp_chunk,
        on="k",
        how="inner",
        sort=False,
        copy=False,
    )
    if m.empty:
        return pd.DataFrame(columns=["i_s1", "i_p", "score"])

    # `w` is already attached to every pool posting, so there is no Series.map
    # here. This is both faster and avoids the old duplicate-index failure.
    gb = (
        m.groupby(["i_s1", "i_p"], sort=False, observed=True)["w"]
        .sum()
        .rename("score")
        .reset_index()
    )
    del m

    if gb.empty:
        return pd.DataFrame(columns=["i_s1", "i_p", "score"])

    # Keep only top-n candidates for each S1 entity. The chunk contains ALL
    # keys for its S1 rows, so this is an exact top-n for those rows.
    gb = gb.sort_values(
        ["i_s1", "score"],
        ascending=[True, False],
        kind="stable",
    )
    gb = gb.groupby("i_s1", sort=False, observed=True).head(topn)
    return gb.reset_index(drop=True)


def _make_s1_ranges(f1, n_s1, pool_counts, target_pairs=750_000):
    """Split S1 rows into balanced ranges using predicted merge-row counts.

    `pool_counts[k]` is the number of pool records for key k. Since each S1
    row has each key at most once, summing these counts gives the exact number
    of pre-groupby rows that its merge will generate.
    """
    if n_s1 == 0 or f1.empty:
        return []

    fi = f1["i"].to_numpy(dtype=np.int64, copy=False)
    fk = f1["k"].to_numpy(dtype=np.int32, copy=False)
    costs = pool_counts[fk]

    # Exact merge work per S1 entity.
    pair_cost = np.bincount(fi, weights=costs, minlength=n_s1).astype(np.int64)

    ranges = []
    start = 0
    acc = 0
    for i, cost in enumerate(pair_cost):
        c = int(cost)
        # Do not split an S1 entity across tasks. If one entity alone is large,
        # it simply becomes a larger task rather than losing any candidates.
        if i > start and acc + c > target_pairs:
            ranges.append((start, i))
            start = i
            acc = c
        else:
            acc += c
    if start < n_s1:
        ranges.append((start, n_s1))
    return ranges


def _pool_postings_for_keys(keys, unique_keys, unique_starts, unique_counts,
                            pool_i, pool_w):
    """Gather pool postings for a set of integer key codes without scanning
    the full pool table."""
    if len(keys) == 0:
        return pd.DataFrame({"k": np.array([], dtype=np.int32),
                             "i_p": np.array([], dtype=np.int64),
                             "w": np.array([], dtype=np.float64)})

    # `unique_keys` is sorted, so searchsorted maps requested keys to their
    # contiguous posting-list ranges in O(len(keys) log len(unique_keys)).
    keys = np.asarray(keys, dtype=np.int32)
    key_pos = np.searchsorted(unique_keys, keys)
    valid = key_pos < len(unique_keys)
    if valid.any():
        valid_pos = np.flatnonzero(valid)
        valid[valid_pos] = unique_keys[key_pos[valid_pos]] == keys[valid_pos]
    key_pos = key_pos[valid]
    keys = keys[valid]
    if len(key_pos) == 0:
        return pd.DataFrame({"k": np.array([], dtype=np.int32),
                             "i_p": np.array([], dtype=np.int64),
                             "w": np.array([], dtype=np.float64)})

    counts = unique_counts[key_pos]
    total = int(counts.sum())
    out_k = np.empty(total, dtype=np.int32)
    out_i = np.empty(total, dtype=np.int64)
    out_w = np.empty(total, dtype=np.float64)

    pos = 0
    for k, p, cnt in zip(keys, key_pos, counts):
        cnt = int(cnt)
        end = pos + cnt
        src = slice(int(unique_starts[p]), int(unique_starts[p]) + cnt)
        out_k[pos:end] = k
        out_i[pos:end] = pool_i[src]
        out_w[pos:end] = pool_w[src]
        pos = end

    return pd.DataFrame({"k": out_k, "i_p": out_i, "w": out_w})


def _candidates_one_country(s1_c, pool_c, cap, cap_num, topn, n_jobs=None):
    """Fast exact blocking within one country.

    The previous implementation merged one hash bucket at a time and then
    repeatedly folded each bucket into a huge `running` DataFrame (~25M rows
    for 259K S1 rows at topn=100). That final fold was single-process and was
    the main reason CPU usage stayed low while runtime exploded.

    This implementation partitions by S1 rows instead. For each S1 chunk it
    gathers only the pool postings for keys actually used by that chunk, merges
    those postings once, aggregates scores once, and keeps top-n immediately.
    Different S1 chunks are independent and run in a ThreadPoolExecutor, so
    pandas' native merge/groupby work can execute concurrently without copying
    the 17M-row pool table into Windows worker processes.
    """
    if len(s1_c) == 0 or len(pool_c) == 0:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "score"])

    print("    computing address token frequencies ...")
    addr_df = compute_addr_df(
        pd.concat([s1_c["business_address"], pool_c["business_address"]])
    )
    f1 = _key_frame(s1_c, addr_df, n_jobs=n_jobs, label="S1")
    fp = _key_frame(pool_c, addr_df, n_jobs=n_jobs, label="pool")
    c1, cp = f1["k"].value_counts(), fp["k"].value_counts()

    def cap_for(k):
        return cap_num if (k.startswith("n:") or k.startswith("nn:")) else cap

    MAX_PAIR_PER_KEY = 20_000
    allk = set(c1.index) | set(cp.index)
    ok = {
        k for k in allk
        if c1.get(k, 0) <= cap_for(k)
        and cp.get(k, 0) <= cap_for(k)
        and c1.get(k, 0) * cp.get(k, 0) <= MAX_PAIR_PER_KEY
    }
    # Only keys that occur in BOTH S1 and pool can generate candidates.
    # `ok` is built from the union of key indexes, so it can also contain
    # S1-only keys. Those keys have no pool weight/postings and must not be
    # assigned an integer code here.
    shared_ok = ok.intersection(c1.index).intersection(cp.index)
    f1 = f1[f1["k"].isin(shared_ok)]
    fp = fp[fp["k"].isin(shared_ok)]
    if f1.empty or fp.empty:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "score"])

    # Integer key codes make merges/groupby much lighter than string keys.
    # Build the codebook ONLY from shared keys so every pool key has a weight
    # entry and the integer codes are guaranteed to be dense [0, n_keys).
    shared_keys = sorted(shared_ok)
    code_of = {k: i for i, k in enumerate(shared_keys)}

    f1 = f1.assign(k=f1["k"].map(code_of).astype("int32"))
    fp = fp.assign(k=fp["k"].map(code_of).astype("int32"))

    w = np.log((len(pool_c) + 1) / (cp.loc[shared_keys] + 1))
    w.index = np.arange(len(shared_keys), dtype=np.int32)
    w_arr = w.to_numpy(dtype=np.float64, copy=False)

    # Keep S1 key rows ordered by entity so ranges can be sliced with two
    # integer positions and no per-task filtering scan.
    f1 = f1.sort_values("i", kind="stable").reset_index(drop=True)
    f1 = f1.rename(columns={"i": "i_s1"})

    # Pool postings are sorted once by key. Then each chunk can fetch its
    # relevant posting lists directly instead of scanning all 17M pool rows.
    fp = fp.rename(columns={"i": "i_p"})
    fp = fp.sort_values("k", kind="stable").reset_index(drop=True)
    pool_k = fp["k"].to_numpy(dtype=np.int32, copy=False)
    pool_i = fp["i_p"].to_numpy(dtype=np.int64, copy=False)
    pool_w = w_arr[pool_k]

    unique_keys, unique_starts, unique_counts = np.unique(
        pool_k, return_index=True, return_counts=True
    )
    pool_counts = np.zeros(len(w_arr), dtype=np.int64)
    pool_counts[unique_keys] = unique_counts

    # f1 was sorted by i_s1, so get row boundaries from entity ids.
    s1_entity_ids = f1["i_s1"].to_numpy(dtype=np.int64, copy=False)
    n_s1 = len(s1_c)
    target_pairs = 750_000
    ranges = _make_s1_ranges(
        f1.rename(columns={"i_s1": "i"}),
        n_s1,
        pool_counts,
        target_pairs=target_pairs,
    )

    if not ranges:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "score"])

    # Convert entity-id ranges to row ranges in f1. Because f1 is sorted by
    # i_s1, searchsorted is O(log n) per boundary.
    f1_i = s1_entity_ids
    range_rows = []
    for ent_start, ent_end in ranges:
        row_start = int(np.searchsorted(f1_i, ent_start, side="left"))
        row_end = int(np.searchsorted(f1_i, ent_end - 1, side="right"))
        range_rows.append((row_start, row_end))

    # Windows multiprocessing would pickle large pandas objects. Threads share
    # the already-built int32 pool arrays and work well here because merge,
    # factorization and groupby execute substantial native code. We materialize
    # only `merge_workers` pool-posting chunks at a time; building every chunk up
    # front would duplicate the giant pool table in memory.
    cpu_jobs = n_jobs or max(1, (os.cpu_count() or 2) - 1)
    merge_workers = max(1, min(int(cpu_jobs), 8, len(range_rows)))
    target_pairs = 750_000
    print(
        f"    S1-chunk merge: {len(range_rows)} chunks, "
        f"{merge_workers} workers, target ~{target_pairs:,} merge rows/chunk ..."
    )

    results = []
    pending = []
    next_idx = 0

    with ThreadPoolExecutor(max_workers=merge_workers) as executor:
        while next_idx < len(range_rows) or pending:
            while next_idx < len(range_rows) and len(pending) < merge_workers:
                row_start, row_end = range_rows[next_idx]
                f1_chunk = f1.iloc[row_start:row_end][["i_s1", "k"]]
                chunk_keys = f1_chunk["k"].unique()
                fp_chunk = _pool_postings_for_keys(
                    chunk_keys,
                    unique_keys,
                    unique_starts,
                    unique_counts,
                    pool_i,
                    pool_w,
                )
                future = executor.submit(_topn_chunk, (f1_chunk, fp_chunk, topn))
                pending.append((next_idx, future))
                next_idx += 1

            # Taking the oldest pending task keeps output deterministic. Other
            # threads continue running while this one finishes.
            idx, future = pending.pop(0)
            result = future.result()
            results.append(result)
            if idx % 10 == 0 or idx == len(range_rows) - 1:
                print(
                    f"    chunk {idx + 1}/{len(range_rows)} "
                    f"({len(result)} top candidates)"
                )

    nonempty = [r for r in results if r is not None and not r.empty]
    if not nonempty:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "score"])

    g = pd.concat(nonempty, ignore_index=True)

    s1_ids = s1_c["entity_id"].values
    p_ids = pool_c["entity_id"].values
    return pd.DataFrame({
        "source1_entity_id": s1_ids[g["i_s1"].to_numpy()],
        "candidate_entity_id": p_ids[g["i_p"].to_numpy()],
        "score": g["score"].to_numpy(),
    })

def generate_candidates(s1_df, pool_df, cap=2500, cap_num=20000, topn=100,
                         n_jobs=None, verbose=True):
    """Main entry point. s1_df / pool_df need columns:
    entity_id, business_name, business_address, country.
    Returns a long DataFrame: source1_entity_id, candidate_entity_id, score
    (one row per candidate pair, topn per S1 entity).

    n_jobs: worker processes for BOTH key generation and the bucket-merge
    loop. Defaults to cpu_count()-1, same convention as _key_frame."""
    out = []
    for c in sorted(s1_df["country"].unique()):
        s1_c = s1_df[s1_df["country"] == c].reset_index(drop=True)
        pool_c = pool_df[pool_df["country"] == c].reset_index(drop=True)
        if verbose:
            print(f"  [{c}] S1={len(s1_c)}  pool={len(pool_c)} ...")
        res = _candidates_one_country(s1_c, pool_c, cap, cap_num, topn, n_jobs=n_jobs)
        if verbose:
            print(f"  [{c}] -> {len(res)} candidate pairs")
        out.append(res)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["source1_entity_id", "candidate_entity_id", "score"])


def to_wide_tsv(candidates_df, s1_ids, path):
    """Write the official candidate_pairs.tsv format: one row per S1 entity,
    even ones with zero candidates."""
    grp = (candidates_df.groupby("source1_entity_id")["candidate_entity_id"]
           .apply(lambda s: ",".join(s)))
    grp = grp.reindex(s1_ids, fill_value="")
    grp.rename_axis("source1_entity_id").reset_index(name="candidate_entity_ids").to_csv(
        path, sep="\t", index=False)