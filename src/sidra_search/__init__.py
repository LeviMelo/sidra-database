"""sidra_search — table-centric search (titles + var/class links + fuzzy)."""

from .db.migrations import apply_search_schema, get_search_schema_version  # noqa: F401
from .db.base_schema import apply_base_schema  # noqa: F401

__all__ = [
    "apply_base_schema",
    "apply_search_schema",
    "get_search_schema_version",
]
