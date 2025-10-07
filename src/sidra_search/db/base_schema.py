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
        periodo_ord INTEGER,
        periodo_kind TEXT,
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
    "CREATE INDEX IF NOT EXISTS idx_variables_agregado ON variables(agregado_id)",
    "CREATE INDEX IF NOT EXISTS idx_categories_agregado ON categories(agregado_id, classification_id)",
    "CREATE INDEX IF NOT EXISTS idx_localities_agregado ON localities(agregado_id, level_id)",
    "CREATE UNIQUE INDEX IF NOT EXISTS u_agregados_levels_pair ON agregados_levels(agregado_id, level_id)",
    "CREATE INDEX IF NOT EXISTS idx_periods_agregado_ord ON periods(agregado_id, periodo_ord)",
)

def apply_base_schema(connection) -> None:
    cur = connection.cursor()
    for stmt in TABLE_STATEMENTS:
        cur.execute(stmt)
    for stmt in INDEX_STATEMENTS:
        cur.execute(stmt)
    connection.commit()
