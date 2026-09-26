"""Pair-level features for the candidate-scoring classifier."""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from normalize import norm, name_tokens, num_tokens


def _lookup(df):
    """entity_id -> (name, address, country) for O(1) access."""
    return {r.entity_id: (r.business_name, r.business_address, r.country)
            for r in df.itertuples(index=False)}


def compute_features(candidates_df, s1_lookup, pool_lookup):
    """candidates_df: source1_entity_id, candidate_entity_id, score (from blocking).
    s1_lookup / pool_lookup: dicts from _lookup(). Returns candidates_df with
    feature columns appended (does not mutate the input)."""
    df = candidates_df.copy()

    s1n, s1a = [], []
    can, caa = [], []
    for sid, cid in zip(df["source1_entity_id"], df["candidate_entity_id"]):
        n1, a1, _ = s1_lookup[sid]
        n2, a2, _ = pool_lookup[cid]
        s1n.append(n1); s1a.append(a1); can.append(n2); caa.append(a2)

    name_ratio, addr_ratio, tok_sort = [], [], []
    name_jacc, addr_jacc = [], []
    num_overlap, num_s1, num_c = [], [], []
    len_diff = []
    for n1, a1, n2, a2 in zip(s1n, s1a, can, caa):
        nn1, nn2 = norm(n1), norm(n2)
        na1, na2 = norm(a1), norm(a2)
        name_ratio.append(fuzz.ratio(nn1, nn2) / 100.0)
        tok_sort.append(fuzz.token_sort_ratio(nn1, nn2) / 100.0)
        addr_ratio.append(fuzz.ratio(na1, na2) / 100.0)

        t1, t2 = set(name_tokens(n1)), set(name_tokens(n2))
        name_jacc.append(len(t1 & t2) / len(t1 | t2) if (t1 | t2) else 0.0)
        w1, w2 = set(na1.split()), set(na2.split())
        addr_jacc.append(len(w1 & w2) / len(w1 | w2) if (w1 | w2) else 0.0)

        num1, num2 = set(num_tokens(a1)), set(num_tokens(a2))
        num_s1.append(len(num1)); num_c.append(len(num2))
        num_overlap.append(len(num1 & num2))
        len_diff.append(abs(len(nn1) - len(nn2)))

    df["name_ratio"] = name_ratio
    df["name_token_sort_ratio"] = tok_sort
    df["addr_ratio"] = addr_ratio
    df["name_jaccard"] = name_jacc
    df["addr_jaccard"] = addr_jacc
    df["num_overlap"] = num_overlap
    df["num_s1_count"] = num_s1
    df["num_cand_count"] = num_c
    df["name_len_diff"] = len_diff
    df["is_s2"] = df["candidate_entity_id"].str.startswith("S2-").astype(int)

    # rank / score-margin features (computed per source1 group)
    df = df.sort_values(["source1_entity_id", "score"], ascending=[True, False])
    df["cand_rank"] = df.groupby("source1_entity_id").cumcount()
    top_score = df.groupby("source1_entity_id")["score"].transform("max")
    df["score_margin_to_top"] = top_score - df["score"]
    n_cand = df.groupby("source1_entity_id")["score"].transform("count")
    df["n_candidates_for_s1"] = n_cand

    return df.reset_index(drop=True)
