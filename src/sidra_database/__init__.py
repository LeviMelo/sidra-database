"""sidra_database package exports."""
from .config import Settings, get_settings
from .api_client import SidraApiClient, SidraApiError
from .ingest import ingest_agregado, ingest_agregado_sync
from .db import ensure_schema, sqlite_session, create_connection, get_database_path
from .embedding import EmbeddingClient
from .catalog import AgregadoRecord, list_agregados
from .search import (
    SemanticMatch,
    SemanticResult,
    semantic_search,
    semantic_search_with_metadata,
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
    "SemanticMatch",
    "SemanticResult",
    "semantic_search",
    "semantic_search_with_metadata",
]



