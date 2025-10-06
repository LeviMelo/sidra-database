"""SQLite schema creation helpers for base SIDRA tables."""
from __future__ import annotations

TABLE_STATEMENTS: tuple[str, ...] = (
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
        id          INTEGER NOT NULL,
        agregado_id INTEGER NOT NULL,
        nome        TEXT NOT NULL,
        unidade     TEXT,
        sumarizacao TEXT,
        text_hash   TEXT NOT NULL,
        PRIMARY KEY (agregado_id, id),
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
        -- normalized/typed period for precise filtering
        periodo_ord INTEGER,          -- sortable ordinal (YYYY00/ YYYYMM / YYYYMMDD)
        periodo_kind TEXT,            -- 'Y' | 'YM' | 'YMD' | 'UNK'
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
    CREATE TABLE IF NOT EXISTS ingestion_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        agregado_id INTEGER NOT NULL,
        stage TEXT NOT NULL,
        status TEXT NOT NULL,
        detail TEXT,
        run_at TEXT NOT NULL
    )
    """,
)

INDEX_STATEMENTS: tuple[str, ...] = (
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
    CREATE UNIQUE INDEX IF NOT EXISTS u_agregados_levels_pair
    ON agregados_levels(agregado_id, level_id)
    """,
    # new — now safe because columns exist:
    """
    CREATE INDEX IF NOT EXISTS idx_periods_agregado_ord
    ON periods(agregado_id, periodo_ord)
    """,
)

ADDITIONAL_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("agregados", "municipality_locality_count", "INTEGER DEFAULT 0"),
    ("agregados", "covers_national_municipalities", "INTEGER DEFAULT 0"),
    ("agregados_levels", "locality_count", "INTEGER DEFAULT 0"),
    # NEW: periods ordering/typed kind
    ("periods", "periodo_ord", "INTEGER"),
    ("periods", "periodo_kind", "TEXT"),
)

EXTRA_INDEXES: tuple[str, ...] = (
    # requires periodo_ord to exist
    "CREATE INDEX IF NOT EXISTS idx_periods_ord ON periods(agregado_id, periodo_ord)",
)

def _column_exists(connection, table: str, column: str) -> bool:
    cursor = connection.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cursor.fetchall())


def apply_base_schema(connection) -> None:
    """Execute schema statements on the provided SQLite connection."""
    cursor = connection.cursor()

    # 1) Tables
    for stmt in TABLE_STATEMENTS:
        cursor.execute(stmt)

    # 2) Additive columns (for already-existing DBs)
    for table, column, definition in ADDITIONAL_COLUMNS:
        if not _column_exists(connection, table, column):
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
            
    # create extra indexes after ensuring columns exist
    for stmt in EXTRA_INDEXES:
        cursor.execute(stmt)
        
    # 3) Indexes (safe now that columns exist)
    for stmt in INDEX_STATEMENTS:
        cursor.execute(stmt)

    connection.commit()



__all__ = ["apply_base_schema"]
