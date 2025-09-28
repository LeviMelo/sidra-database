from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping

DEFAULT_WEIGHTS = {"struct": 0.7, "rrf": 0.3}


def rrf(ranks: Mapping[str, int], k: float = 60.0) -> Dict[str, float]:
    return {key: 1.0 / (k + rank) for key, rank in ranks.items()}


@dataclass(frozen=True)
class StructureMatch:
    variable: float
    unit: float
    dims: float
    period: float
    geo: float

    def score(self) -> float:
        return (
            0.40 * self.variable
            + 0.20 * self.unit
            + 0.20 * self.dims
            + 0.10 * self.period
            + 0.10 * self.geo
        )


__all__ = ["rrf", "StructureMatch", "DEFAULT_WEIGHTS"]
