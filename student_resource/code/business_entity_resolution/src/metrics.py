"""Per-entity macro F0.5 — the exact competition metric.
A Source1 entity with no true matches scores 1.0 if you predict empty,
0.0 if you predict anything. All Source1 entities count in the average,
singletons included.
"""

def f_beta(precision, recall, beta=0.5):
    if precision == 0 and recall == 0:
        return 0.0
    b2 = beta * beta
    denom = b2 * precision + recall
    return 0.0 if denom == 0 else (1 + b2) * precision * recall / denom


def macro_f05(true_matches: dict, pred_matches: dict, all_s1_ids):
    """true_matches / pred_matches: {source1_entity_id: set(candidate_ids)}.
    all_s1_ids: every Source1 id that must be scored (test set or val fold)."""
    scores = []
    for sid in all_s1_ids:
        t = true_matches.get(sid, set())
        p = pred_matches.get(sid, set())
        if not t and not p:
            scores.append(1.0)
            continue
        if not t and p:
            scores.append(0.0)
            continue
        if t and not p:
            scores.append(0.0)
            continue
        tp = len(t & p)
        precision = tp / len(p)
        recall = tp / len(t)
        scores.append(f_beta(precision, recall))
    return sum(scores) / len(scores) if scores else 0.0
