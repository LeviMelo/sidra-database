from __future__ import annotations

DDL: tuple[str, ...] = (
    # meta kv (shared)
    """
    CREATE TABLE IF NOT EXISTS meta_kv(
      key TEXT PRIMARY KEY,
      value TEXT NOT NULL
    )
    """,
    # link keys
    """
    CREATE TABLE IF NOT EXISTS name_keys (
      kind TEXT NOT NULL,     -- 'var' | 'class' | 'cat'
      key  TEXT NOT NULL,     -- normalized
      raw  TEXT NOT NULL,     -- original
      UNIQUE(kind, key, raw)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS link_var (
      var_key    TEXT NOT NULL,
      table_id   INTEGER NOT NULL,
      variable_id INTEGER NOT NULL,
      UNIQUE(var_key, table_id, variable_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS link_class (
      class_key TEXT NOT NULL,
      table_id  INTEGER NOT NULL,
      class_id  INTEGER NOT NULL,
      UNIQUE(class_key, table_id, class_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS link_cat (
      class_key  TEXT NOT NULL,
      cat_key    TEXT NOT NULL,
      table_id   INTEGER NOT NULL,
      class_id   INTEGER NOT NULL,
      category_id INTEGER NOT NULL,
      UNIQUE(class_key, cat_key, table_id, class_id, category_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS link_var_class (
      var_key   TEXT NOT NULL,
      class_key TEXT NOT NULL,
      table_id  INTEGER NOT NULL,
      variable_id INTEGER NOT NULL,
      class_id    INTEGER NOT NULL,
      UNIQUE(var_key, class_key, table_id, variable_id, class_id)
    )
    """,
    # FTS for titles (no extra index allowed on virtual table)
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS table_titles_fts
    USING fts5(table_id UNINDEXED, title, survey, subject, tokenize='unicode61')
    """,
    # Embeddings table (generic, reused for table titles)
    """
    CREATE TABLE IF NOT EXISTS embeddings (
      entity_type TEXT NOT NULL,   -- 'agregado'
      entity_id   TEXT NOT NULL,   -- table_id as text
      agregado_id INTEGER,         -- optional convenience (same as entity_id)
      text_hash   TEXT NOT NULL,
      model       TEXT NOT NULL,
      dimension   INTEGER NOT NULL,
      vector      BLOB NOT NULL,
      created_at  TEXT NOT NULL,
      PRIMARY KEY (entity_type, entity_id, model)
    )
    """,
)

INDEXES: tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS idx_link_var_key   ON link_var(var_key)",
    "CREATE INDEX IF NOT EXISTS idx_link_class_key ON link_class(class_key)",
    "CREATE INDEX IF NOT EXISTS idx_link_cat_keys  ON link_cat(class_key, cat_key)",
    "CREATE INDEX IF NOT EXISTS idx_link_var_class ON link_var_class(var_key, class_key)",
    "CREATE INDEX IF NOT EXISTS idx_embeddings_agregado ON embeddings(agregado_id, model)",
)

def apply_search_schema(connection) -> None:
    cur = connection.cursor()
    for stmt in DDL:
        cur.execute(stmt)
    for stmt in INDEXES:
        cur.execute(stmt)
    connection.commit()
