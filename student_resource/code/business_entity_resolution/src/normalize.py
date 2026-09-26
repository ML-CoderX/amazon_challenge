"""Shared text-normalisation helpers for the entity-resolution pipeline."""
import re
from unidecode import unidecode

STOP = set("pvt private ltd limited llp llc inc incorporated corp corporation co company "
           "and the of m s ms sarl sas lp plc pty".split())

def norm(s):
    """Lowercase, transliterate, strip punctuation."""
    s = unidecode(str(s)).lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def cn(t):
    """Collapse repeated letters: MINNNETONKA -> MINETONKA (typo tolerance)."""
    return re.sub(r"(.)\1+", r"\1", t)

def skel(t):
    """Consonant skeleton, helps loosely match transliteration variants."""
    return cn(re.sub(r"[aeiouyhw]", "", t))

def name_tokens(name):
    return [t for t in norm(name).split() if t not in STOP and len(t) > 1]

def addr_tokens(addr):
    return norm(addr).split()

def num_tokens(addr):
    """All digit-runs (len>=2) from the RAW address, e.g. house/plot numbers."""
    return [t for t in re.findall(r"\d{2,}", str(addr))]
