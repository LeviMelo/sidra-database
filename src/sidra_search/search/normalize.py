# src/sidra_search/search/normalize.py
from __future__ import annotations

import re
import unicodedata

_WS = re.compile(r"\s+")
# Keep hyphen as a token-forming char per plan; drop other punctuation.
_PUNCT = re.compile(r"[^\w\s-]", re.UNICODE)

def normalize_basic(s: str | None) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().replace("\u00a0", " ")
    # normalize whitespace around hyphens
    s = _WS.sub(" ", s).strip()
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    # collapse repeated hyphens/spaces like " -  - "
    s = re.sub(r"\s*-\s*", "-", s)
    s = _WS.sub(" ", s).strip()
    return s
