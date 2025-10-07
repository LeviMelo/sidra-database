from __future__ import annotations
from typing import Mapping, Dict

def rrf(ranks: Mapping[int, int], k: float = 60.0) -> Dict[int, float]:
    return {k_: 1.0 / (k + r) for k_, r in ranks.items()}
