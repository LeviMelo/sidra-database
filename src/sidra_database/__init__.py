"""sidra_database package exports."""
from .config import Settings, get_settings
from .api_client import SidraApiClient, SidraApiError
from .ingest import ingest_agregado, ingest_agregado_sync
from .db import ensure_schema, sqlite_session, create_connection, get_database_path
from .embedding import EmbeddingClient
from .catalog import AgregadoRecord, list_agregados
from .bulk_ingest import (
    BulkIngestionReport,
    discover_agregados_by_coverage,
    ingest_by_coverage,
)
from .discovery import CatalogEntry
from .search import (
    SemanticMatch,
    SemanticResult,
    SearchFilters,
    semantic_search,
    semantic_search_with_metadata,
    hybrid_search,
)

__all__ = [
    "Settings",
    "get_settings",
    "SidraApiClient",
    "SidraApiError",
    "ingest_agregado",
    "ingest_agregado_sync",
    "ensure_schema",
    "sqlite_session",
    "create_connection",
    "get_database_path",
    "EmbeddingClient",
    "AgregadoRecord",
    "list_agregados",
    "BulkIngestionReport",
    "CatalogEntry",
    "discover_agregados_by_coverage",
    "ingest_by_coverage",
    "SemanticMatch",
    "SemanticResult",
    "SearchFilters",
    "semantic_search",
    "semantic_search_with_metadata",
    "hybrid_search",
]



