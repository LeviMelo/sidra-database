"""SQLite schema creation helpers for base SIDRA tables."""
from __future__ import annotations

SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS agregados (
        id INTEGER PRIMARY KEY,
        nome TEXT NOT NULL,
        pesquisa TEXT,
        assunto TEXT,
        url TEXT,
        freq TEXT,
        periodo_inicio TEXT,
        periodo_fim TEXT,
        raw_json BLOB NOT NULL,
        fetched_at TEXT NOT NULL,
        municipality_locality_count INTEGER DEFAULT 0,
        covers_national_municipalities INTEGER DEFAULT 0
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS agregados_levels (
        agregado_id INTEGER NOT NULL,
        level_id TEXT NOT NULL,
        level_name TEXT,
        level_type TEXT NOT NULL,
        locality_count INTEGER DEFAULT 0,
        PRIMARY KEY (agregado_id, level_id, level_type),
        FOREIGN KEY (agregado_id) REFERENCES agregados(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS variables (
        id INTEGER PRIMARY KEY,
        agregado_id INTEGER NOT NULL,
        nome TEXT NOT NULL,
        unidade TEXT,
        sumarizacao TEXT,
        text_hash TEXT NOT NULL,
        FOREIGN KEY (agregado_id) REFERENCES agregados(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS classifications (
        id INTEGER NOT NULL,
        agregado_id INTEGER NOT NULL,
        nome TEXT NOT NULL,
        sumarizacao_status INTEGER,
        sumarizacao_excecao TEXT,
        PRIMARY KEY (agregado_id, id),
        FOREIGN KEY (agregado_id) REFERENCES agregados(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS categories (
        agregado_id INTEGER NOT NULL,
        classification_id INTEGER NOT NULL,
        categoria_id INTEGER NOT NULL,
        nome TEXT NOT NULL,
        unidade TEXT,
        nivel INTEGER,
        text_hash TEXT NOT NULL,
        PRIMARY KEY (agregado_id, classification_id, categoria_id),
        FOREIGN KEY (agregado_id, classification_id) REFERENCES classifications(agregado_id, id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS periods (
        agregado_id INTEGER NOT NULL,
        periodo_id TEXT NOT NULL,
        literals TEXT NOT NULL,
        modificacao TEXT,
        PRIMARY KEY (agregado_id, periodo_id),
        FOREIGN KEY (agregado_id) REFERENCES agregados(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS localities (
        agregado_id INTEGER NOT NULL,
        level_id TEXT NOT NULL,
        locality_id TEXT NOT NULL,
        nome TEXT NOT NULL,
        PRIMARY KEY (agregado_id, level_id, locality_id),
        FOREIGN KEY (agregado_id, level_id) REFERENCES agregados_levels(agregado_id, level_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS embeddings (
        entity_type TEXT NOT NULL,
        entity_id TEXT NOT NULL,
        agregado_id INTEGER,
        text_hash TEXT NOT NULL,
        model TEXT NOT NULL,
        dimension INTEGER NOT NULL,
        vector BLOB NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (entity_type, entity_id, model)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ingestion_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        agregado_id INTEGER NOT NULL,
        stage TEXT NOT NULL,
        status TEXT NOT NULL,
        detail TEXT,
        run_at TEXT NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_variables_agregado ON variables(agregado_id)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_categories_agregado ON categories(agregado_id, classification_id)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_localities_agregado ON localities(agregado_id, level_id)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_embeddings_agregado ON embeddings(agregado_id, model)
    """
)

ADDITIONAL_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("agregados", "municipality_locality_count", "INTEGER DEFAULT 0"),
    ("agregados", "covers_national_municipalities", "INTEGER DEFAULT 0"),
    ("agregados_levels", "locality_count", "INTEGER DEFAULT 0"),
)


def _column_exists(connection, table: str, column: str) -> bool:
    cursor = connection.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cursor.fetchall())


def apply_base_schema(connection) -> None:
    """Execute schema statements on the provided SQLite connection."""
    cursor = connection.cursor()
    for stmt in SCHEMA_STATEMENTS:
        cursor.execute(stmt)
    for table, column, definition in ADDITIONAL_COLUMNS:
        if not _column_exists(connection, table, column):
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
    connection.commit()


__all__ = ["apply_base_schema"]
