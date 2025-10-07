# plan_sidra_search_reboot.md

A compact, no-nonsense plan to rebuild the SIDRA table search so it’s fast, small, and predictable—without “value atoms” or heavyweight indexing.

---

## 0) Outcome (what the tool will do)

A single CLI that returns **tables** by:

* **Title** (semantic + optional lexical on titles)
* **Variables / Classifications** (names, not IDs; fuzzy by default)
* Optional **coverage** filters (`(N3>=27) OR (N6>=5000)`)

**Identity = names.** We normalize names to keys (Portuguese preserved).
**Pairing is real.** If you ask for a variable + classes, we only return tables where that variable is actually broken down by those classes.

No “value atoms.” No global IDs. No unit ontologies.

---

## 1) Scope / Non-goals

**In scope**

* SQLite metadata store for tables, variables, classes, categories
* Lightweight “link_*” tables to support AND semantics + pairing
* In-RAM 3-gram fuzzy for variables/classes
* Optional FTS5 for titles
* Optional title embeddings (cosine)
* Coverage filter execution

**Out of scope**

* Building value-atoms, neighbors, or vector stores for values
* Cross-table data aggregation or unit conversions
* Category “fuzzy” (kept strict if used at all)

---

## 2) Architecture (simple & small)

* **Metadata tables:** `variables`, `classifications`, `categories` with **table-local** SIDRA IDs and **normalized keys**.
* **Link tables (inverted indexes):** `link_var`, `link_class`, `link_cat`, `link_var_class` (names → tables; and var↔class pairing).
* **Titles:**

  * Optional FTS5 for lexical title search
  * Optional embeddings for semantic title search
* **Fuzzy:** 3-gram TF-IDF cosine **in memory** at query time; no disk grams by default.
* **Config toggles:** indexing for titles (FTS/embeddings) on/off; fuzzy thresholds.

No VA tables, no variable fingerprints, no “neighbors” required for table discovery.

---

## 3) Normalization (deterministic)

`norm(s)`:

* lowercase
* strip accents/diacritics
* trim
* collapse internal whitespace
* drop trivial punctuation (commas, periods, parens when empty); keep token-forming hyphens if present

Keys:

* `var_key = norm(variable_name)`
* `class_key = norm(class_name)`
* `cat_key = norm(category_name)`

---

## 4) Schema (minimal)

```sql
-- Tables
CREATE TABLE IF NOT EXISTS agregados(
  id INTEGER PRIMARY KEY,
  nome TEXT,
  pesquisa TEXT,
  assunto TEXT,
  periodo_inicio INTEGER,
  periodo_fim INTEGER
);

CREATE TABLE IF NOT EXISTS variables(
  table_id INTEGER,
  variable_id INTEGER,          -- SIDRA variable id (table-local)
  name TEXT NOT NULL,
  unit TEXT,
  var_key TEXT NOT NULL,
  PRIMARY KEY(table_id, variable_id)
);

CREATE TABLE IF NOT EXISTS classifications(
  table_id INTEGER,
  class_id INTEGER,             -- SIDRA class id (table-local)
  name TEXT NOT NULL,
  class_key TEXT NOT NULL,
  PRIMARY KEY(table_id, class_id)
);

CREATE TABLE IF NOT EXISTS categories(
  table_id INTEGER,
  class_id INTEGER,
  category_id INTEGER,          -- SIDRA category id (table-local)
  name TEXT NOT NULL,
  cat_key TEXT NOT NULL,
  PRIMARY KEY(table_id, class_id, category_id)
);

-- Links (fast set intersections + pairing proof)
CREATE TABLE IF NOT EXISTS link_var(
  var_key TEXT,
  table_id INTEGER,
  variable_id INTEGER,
  PRIMARY KEY(var_key, table_id, variable_id)
);

CREATE TABLE IF NOT EXISTS link_class(
  class_key TEXT,
  table_id INTEGER,
  class_id INTEGER,
  PRIMARY KEY(class_key, table_id, class_id)
);

CREATE TABLE IF NOT EXISTS link_cat(
  class_key TEXT,               -- normalized class name
  cat_key TEXT,                 -- normalized category name
  table_id INTEGER,
  class_id INTEGER,
  category_id INTEGER,
  PRIMARY KEY(class_key, cat_key, table_id, class_id, category_id)
);

CREATE TABLE IF NOT EXISTS link_var_class(
  var_key TEXT,
  class_key TEXT,
  table_id INTEGER,
  variable_id INTEGER,
  class_id INTEGER,
  PRIMARY KEY(var_key, class_key, table_id, variable_id, class_id)
);

-- Optional: coverage
CREATE TABLE IF NOT EXISTS coverage(
  table_id INTEGER PRIMARY KEY,
  N1_count INTEGER, N2_count INTEGER, N3_count INTEGER, N6_count INTEGER
);

-- Optional: titles search helpers
CREATE VIRTUAL TABLE IF NOT EXISTS table_titles_fts
USING fts5(table_title, content='');

CREATE TABLE IF NOT EXISTS title_embeddings(
  table_id INTEGER PRIMARY KEY,
  -- store as JSON array string or BLOB of float32s
  embed BLOB
);
```

**Note on IDs:** `variable_id`, `class_id`, `category_id` are **from SIDRA**, but **always qualified by `table_id`**; never reused across tables.

---

## 5) Ingest (per table)

For each `table_id`:

1. **Extract** variables, classes, categories from your cached JSON/API.
2. **Normalize**: compute `var_key`, `class_key`, `cat_key`.
3. **Insert** into `variables`, `classifications`, `categories`.
4. **Build links**:

   * `link_var(var_key, table_id, variable_id)`
   * `link_class(class_key, table_id, class_id)`
   * `link_cat(class_key, cat_key, table_id, class_id, category_id)`
   * For each **actual schema pairing** (variable × classification actually used in the table):
     `link_var_class(var_key, class_key, table_id, variable_id, class_id)`
5. **Titles**: upsert into `table_titles_fts` and (if enabled) compute/store embedding.
6. **Coverage**: upsert per your probe.

No VAs. No fingerprints. No neighbors.

---

## 6) Query model

### A) Title-only

* **Recall**: semantic cosine (embeddings) ∪ FTS5 BM25 hits (if enabled)
* **Rank**: RRF of (semantic rank, BM25 rank). Or weighted sum with cosine.

### B) Structural (`--var`, `--class`) without `--title`

* **Expand** each provided var/class by **3-gram fuzzy** (in RAM) into candidate keys with cosine scores.
* **Filter** tables by set intersections on `link_*`, and **enforce pairing** with `link_var_class`.
* **Rank** with **structure score** (best variable per table that satisfies all classes):

  ```
  structure = 0.55 * var_cos
            + 0.35 * mean(class_cos_j)
            + 0.10 * exact_match_bonus
  ```

### C) Combined (`--title` + structural)

* **Filter** = structural only (title doesn’t add recall here).
* **Rank** = combine:

  * **Multiplicative** (structure dominates): `final = structure * (1 + w * title_score)`, e.g. `w=0.35`
  * or **Additive**: `final = 0.75*structure + 0.25*title_score`

### D) Coverage

* Apply after structural filtering (simple SQL on `coverage`).

---

## 7) 3-gram fuzzy (in RAM)

* **At startup** (or first search), build dicts:

  * `all_var_names = {var_key: raw_name}` and cached 3-gram vectors
  * `all_class_names = {class_key: raw_name}` and cached 3-gram vectors
* **expand_trigram(query_str, topK=20, τ=threshold)**:

  * build query 3-gram vector
  * cosine vs cached vectors
  * return topK ≥ τ as `(key, cos)`

**Threshold defaults:**

* `var_th = 0.90`
* `class_th = 0.85`
* (Allow `--no-fuzzy` to force exact only, but keep fuzzy default ON.)

No disk grams by default. (Optional future: persist grams/IDF if you want faster warm-starts.)

---

## 8) Structural filter (precise and fast)

Given:

* `VAR_REQS = [q1, q2, ...]` (0–many)
* `CLASS_REQS = [c1, c2, …]` (0–many)

Expand each to:

* `V_cands = {(var_key, cos)}`
* `Cj_cands = list of {(class_key, cos)}`

**Tables that survive:**

1. If any `VAR_REQS` present:

   * Start from `T = { tables that have var_key ∈ V_cands via link_var }`

2. For each requested class j:

   * Keep only tables where the **same variable** also pairs with **some** `class_key ∈ Cj_cands` via `link_var_class`.

3. If only `CLASS_REQS` present (no variable):

   * `T` = intersection over j of tables in `link_class` for any key ∈ `Cj_cands[j]`
   * (No pairing constraint needed because there’s no variable to pair with.)

4. Apply coverage filter if given.

This enforces **real schema constraints** without exploding storage.

---

## 9) Ranking (details)

**Structure score (per table)**

* Choose the **best variable** in that table that satisfies all class constraints.
* `s_var`: cosine of that var match
* `s_classes`: mean of the best cosine per requested class
* `exact boosts`: +0.10 for exact var key; +0.05 per exact class key
* `structure = 0.55*s_var + 0.35*s_classes + exact_boosts`

**Title score**

* If `--title`:

  * **Semantic** cosine from `title_embeddings`
  * (Optional) fold BM25 from FTS5 into a mini-RRF with semantic
  * Normalize to 0–1

**Final**

* With structure only: `final = structure`
* With title too: multiplicative or additive combine (see §6C)

---

## 10) CLI UX

```
python -m sidra_va.cli search tables
  [--title "texto"]        # alias: --q (kept for backward compat)
  [--var   "nome da variável"]   # repeatable (AND); fuzzy by default
  [--class "nome da classificação"]  # repeatable (AND); fuzzy by default
  [--coverage "(N3>=27) OR (N6>=5000)"]
  [--no-fuzzy | --var-th 0.90 | --class-th 0.85]
  [--limit 50]
```

**Output**

```
{table_id} | {title} | years:{start}–{end} | cov:N3={x}/N6={y}
matches: [var≈ "..."] [class≈ "..."] [...]
```

Use `≈` for fuzzy; omit for exact.

---

## 11) Implementation checklist

**Kill/deprecate**

* `value_atoms*`, `value_index.py`, `neighbors.py`, `variable_fingerprints` (and any CLI that builds them)

**Schema migrations**

* Add columns: `var_key`, `class_key`, `cat_key`
* Create `link_*` tables
* Optional: `table_titles_fts`, `title_embeddings`, `coverage`
* Remove VA tables if you want a clean DB (or leave; just unused)

**Ingest**

* Update ingestion to compute normalized keys and fill `variables`, `classifications`, `categories` + `link_*`
* Implement var×class pairing extraction and fill `link_var_class`
* Titles: upsert into FTS (optional) and embeddings (optional)
* Coverage: upsert

**Search**

* `search_tables.py`:

  * Trigram expander (RAM cache)
  * Structural filter using `link_*`
  * Structure score
  * Title scorer (semantic; optional FTS5 BM25)
  * Final combiner
* `cli.py`:

  * `--title/--q`, `--var`, `--class`, `--coverage`, thresholds, `--no-fuzzy`, `--limit`

**Type fixes (Pylance)**

* Define concrete dataclasses / TypedDicts for results (`TableHit`) and use them in signatures
* Avoid assigning `List[Tuple[str,float]]` into `List[TableHit]`
* Fix “Never is not iterable” by refining union types / narrowing `Optional` before iterating

---

## 12) Migration / sanity SQL

**Backfill keys (one-off if you already have names)**

```sql
-- After adding columns var_key/class_key/cat_key:
UPDATE variables
SET var_key = :norm(name);

UPDATE classifications
SET class_key = :norm(name);

UPDATE categories
SET cat_key = :norm(name);
```

(`:norm` is done in Python; then write back.)

**Spot checks**

```sql
-- What variable names exist that include "pessoas" (raw LIKE, just for eyeballing)?
SELECT table_id, variable_id, name
FROM variables
WHERE lower(name) LIKE '%pessoas%';

-- Confirm keys exist:
SELECT DISTINCT var_key FROM variables LIMIT 20;

-- Which tables claim a var_key:
SELECT table_id FROM link_var WHERE var_key='pessoas que dirigem os estabelecimentos agropecuarios';

-- Prove pairing exists:
SELECT * FROM link_var_class
WHERE var_key='pessoas que dirigem os estabelecimentos agropecuarios'
  AND class_key='grupo de idade';
```

---

## 13) Testing & acceptance

* **Unit**: normalization, trigram cosine, structural filter with synthetic tiny DB
* **CLI e2e**:

  * `--title` only returns reasonable things
  * `--var "pessoas que dirigem..."` returns table 1022
  * `--var "pessoas" (fuzzy)` returns tables with “Pessoas …”
  * `--var ... --class "grupo de idade"` keeps only tables where that variable is actually broken by that class
  * Coverage filters include/exclude as expected
* **Perf**: ≤100 ms per structured query on ~100–500 tables with in-RAM trigram

Acceptance = passes tests above and DB size stays small (no VA bloat).

---

## 14) Config toggles (env or CLI)

* `SIDRA_ENABLE_FTS_TITLES=0/1` (default 0)
* `SIDRA_ENABLE_TITLE_EMBEDDINGS=0/1` (default 0)
* `SIDRA_VAR_TH=0.90`, `SIDRA_CLASS_TH=0.85`
* CLI `--no-fuzzy`, `--var-th`, `--class-th` override

---

## 15) Logging / explain

* Log expansions: chosen `(var_key, cos)`, `(class_key, cos)`
* Log which `table_id` survived each filter step
* `--explain` flag prints the match rationale per table (keys and cosines; exact flags)

---

## 16) Rollback

* This plan **does not** mutate data values—only metadata tables.
* If needed, revert by dropping `link_*`, keys, and CLI paths; the old DB remains intact (minus VA builders you remove).

---

## 17) Why this will work

* **High recall** from trigram fuzzy (variables/classes)
* **High precision** from enforcing real var↔class pairing (`link_var_class`)
* **Low storage**: a few small link tables; optional FTS/embeddings
* **Clear UX**: title for intent; structure for correctness; simple knobs

---

## 18) Pseudocode anchors

```python
def expand_trigram(query: str, keys: dict[str, str], topk=20, th=0.85) -> list[tuple[str,float]]:
    # keys: {key -> raw_name}, pre-cached 3-gram vectors for values
    qvec = grams(query)
    scored = [(k, cosine(qvec, cache[k])) for k in keys]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [(k, s) for (k, s) in scored[:topk] if s >= th]

def structural_filter(var_reqs: list[str], class_reqs: list[str]) -> set[int]:
    V = union(expand_trigram(v, all_var_names, th=VAR_TH) for v in var_reqs)  # list of (var_key, cos)
    C = [expand_trigram(c, all_class_names, th=CLASS_TH) for c in class_reqs] # list[list (class_key, cos)]

    if V:
        T = {t for (vk, _) in V for (t, _) in fetch_link_var(vk)}  # link_var lookup
        for j, Cj in enumerate(C):
            # keep tables where same variable pairs with some class_key in Cj
            T = {
              t for t in T
              if any(pair_exists(vk, ck, t) for (vk, _) in V for (ck, _) in Cj)
            }
    else:
        # classes only: intersect tables per class set
        sets = [{t for (ck, _) in Cj for (t, _) in fetch_link_class(ck)} for Cj in C]
        T = set.intersection(*sets) if sets else all_tables()

    return T
```

(Where `pair_exists(vk, ck, t)` checks `link_var_class` for `(vk, ck, table_id=t)` and fetch helpers are small SQL lookups.)

---

**Bottom line:**
We’re replacing the sprawling VA approach with a **tight, name-key + links** design: tiny to store, fast to query, and faithful to how people search.

# 19) Codebase migration & refactor guide (from current `sidra_va`)

This section turns the **plan** into a concrete, file-by-file refactor/migration checklist. We’ll **start a clean package**, move over only what we need, and delete the VA baggage.

---

## 19.1 New package layout

We’ll create a new top-level package and keep the old one around only until we finish cutover.

```
src/
  sidra_search/
    __init__.py
    config.py                  # env + defaults (kept, trimmed)
    db/
      __init__.py
      session.py               # create_connection(), sqlite_session(), ensure_full_schema()
      base_schema.py           # base SIDRA tables (from old base_schema.py)
      search_schema.py         # FTS on titles, link tables, meta_kv versioning
      migrations.py            # bump + apply versions (v4+ for search)
    net/
      __init__.py
      api_client.py            # from old api_client.py unchanged (rename module path)
    ingest/
      __init__.py
      ingest_table.py          # from old ingest_base.py (trimmed to only base metadata + titles embedding)
      bulk.py                  # from old bulk_ingest.py (trimmed)
      links.py                 # from old links.py (rewritten “actual pairing” hook; see 19.5)
    search/
      __init__.py
      normalize.py             # single normalization function (moved from synonyms.normalize_basic)
      fuzzy3gram.py            # from old fuzzy.py (renamed, in-RAM TF-IDF trigrams)
      tables.py                # from old search_tables.py (rewritten to remove VA deps, use titles FTS)
      title_rank.py            # semantic + lexical title scoring utils
    cli/
      __init__.py
      main.py                  # from old cli.py, slimmed to only ingest/list/show/search tables/diagnostics
    util/
      __init__.py
      time.py, text.py         # tiny helpers if/when we need them
```

> We’ll **deprecate** the old `sidra_va` package and leave a tiny `__init__` shim that prints a message directing users to `sidra_search`. (See 19.10 Cutover.)

---

## 19.2 Remove entirely (delete files & CLI paths)

These are VA-only or dead weight and must go:

* `value_index.py`
* `search_va.py`
* `embed.py` (VA embeddings)
* `neighbors.py`
* `fingerprints.py`
* `plan_sidra_search_unified_cli_name_keys.md` (superseded by this plan)
* Any VA-specific tests/fixtures

Also **delete** all CLI subcommands that touch those (see 19.8).

---

## 19.3 Keep (move/rename) with light edits

* `api_client.py` → `sidra_search/net/api_client.py` (no behavior change)

* `base_schema.py` → `sidra_search/db/base_schema.py` (unchanged except imports)

* `config.py` → `sidra_search/config.py` (trim env keys to what we use; see 19.11)

* `discovery.py` → `sidra_search/ingest/bulk.py` (integrate discovery helpers here)

* `coverage.py` → `sidra_search/search/coverage.py` or keep at root; update imports

* `catalog.py` → `sidra_search/cli/support_catalog.py` (optional), or fold listing into CLI

* `diagnostics_base.py` → `sidra_search/cli/support_diag.py` (keep only base-metadata checks)

* `db.py` → split into:

  * `sidra_search/db/session.py` (connection helpers)
  * remove `apply_va_schema` references (see 19.4)

* `links.py` → `sidra_search/ingest/links.py` (adjust as in 19.5)

* `fuzzy.py` → `sidra_search/search/fuzzy3gram.py` (rename symbols, no DB grams)

* `search_tables.py` → `sidra_search/search/tables.py` (rewrite FTS part; see 19.6)

* `ingest_base.py` → `sidra_search/ingest/ingest_table.py` (VA bits stripped; keep table-level embeddings; see 19.7)

* `schema_migrations.py` → **split**:

  * `sidra_search/db/migrations.py` (new, v4+ for search)
  * `sidra_search/db/search_schema.py` (DDL for links + titles FTS)

* `scoring.py` → keep only `rrf()` (move to `sidra_search/search/title_rank.py`)

* `synonyms.py` → **keep only** `normalize_basic()` and simple helpers; remove synonym maps/import/export (see 19.8/19.11)

---

## 19.4 Database schema (new “search” migration v4)

We stop using VA tables for discovery. Add a new **search schema** migration without touching base SIDRA tables.

**Create** `sidra_search/db/search_schema.py`:

* **Link tables** (already exist in v3): `name_keys`, `link_var`, `link_class`, `link_cat`, `link_var_class`.

  * Keep the same structure; we’ll populate via the new `ingest/links.py`.
* **Titles FTS** (new; replaces use of `value_atoms_fts`):

  ```sql
  CREATE VIRTUAL TABLE IF NOT EXISTS table_titles_fts
  USING fts5(table_id UNINDEXED, title, survey, subject, tokenize='unicode61');
  CREATE INDEX IF NOT EXISTS idx_titles_fts_table_id ON table_titles_fts(table_id);
  ```
* **Optional**: `title_embeddings(table_id INTEGER PRIMARY KEY, model TEXT, dimension INT, vector BLOB)`

  * Or continue to use existing generic `embeddings` table with `entity_type='agregado'` (preferred — already in DB).
* **Meta KV**: reuse `meta_kv` table managed in migrations.

**Create** `sidra_search/db/migrations.py`:

* `SEARCH_SCHEMA_VERSION = 4`
* `apply_search_schema(conn)`:

  * Ensure `meta_kv` exists.
  * If current `< 4`, create `table_titles_fts` (and optional `title_embeddings` if you decide not to reuse `embeddings`), ensure link indexes exist.
  * `bump_schema_version(..., to=4)` (reuse key `sidra_search_schema_version` or reuse existing `sidra_va_schema_version`—pick one and stick with it; recommend **new key** to isolate from VA).

**Edit** `sidra_search/db/session.py`:

* `ensure_full_schema()` now calls:

  * `apply_base_schema(conn)`
  * `apply_search_schema(conn)` (not VA)

---

## 19.5 Link building (realistic pairing & normalization)

Move and rewrite `links.py` → `sidra_search/ingest/links.py`:

* **Normalization**: import a single source of truth:

  * `from sidra_search/search/normalize import normalize_basic`
* **Insert**:

  * `name_keys(kind, key, raw)` on first sight of each `raw` string.
  * `link_var(var_key, table_id, variable_id)` from `variables`.
  * `link_class(class_key, table_id, class_id)` from `classifications`.
  * `link_cat(class_key, cat_key, table_id, class_id, category_id)` from `categories`.
* **Var×Class pairing**:

  * Keep the **current cross-product fallback** (all vars × all classes present), **but**:

    * Add a **hook** `_infer_var_class_pairs(metadata_json)` that returns the actual subset if we can parse it (future work: classification summarização/exceção often hints applicability).
    * If the hook returns a non-empty subset, use it; else use cross-product.
* **API**:

  * `build_links_for_table(table_id: int) -> LinkCounts`
  * `build_links_for_all(concurrency: int = 4) -> dict[int, LinkCounts]` (unchanged signature)
* **Call site**:

  * At the **end of ingestion** for a table (see 19.7), call `build_links_for_table(table_id)` automatically.

---

## 19.6 Title ranking (remove VA FTS dependency)

Rewrite `search_tables.py` → `sidra_search/search/tables.py`:

* **Delete** the part that queries `value_atoms_fts`. Replace with:

  ```sql
  SELECT table_id FROM table_titles_fts
  WHERE table_titles_fts MATCH ?
  ```

  Where `?` is a sanitized token query (use `normalize_basic` + join with spaces). Keep the **distinct** and limit logic, then map into a **rank** (1-based).

* **Semantic ranking**: keep current cosine flow, but **only** against `embeddings` rows with `entity_type='agregado'` and `entity_id=<table_id>` (already implemented).

* **RRF**: unchanged; combine lexical(rank) + semantic(rank) into `rrf_scores`.

* **Structural filtering**:

  * Keep var/class fuzzy expansion code paths you already have (they’re good).
  * Keep the “enforce var×class co-occurrence **only for strict originals**” logic (you already do this right).

* **Coverage filter**: unchanged (uses `agregados_levels` and `coverage.py`).

* **Output**: `TableHit` dataclass retained.

> Result: **no dependency** on VA tables at all.

---

## 19.7 Ingestion (slim, table-centric)

Rewrite `ingest_base.py` → `sidra_search/ingest/ingest_table.py`:

* **Keep**:

  * Fetch metadata + periods + localities (unchanged).
  * Insert into `agregados`, `variables`, `classifications`, `categories`, `periods`, `agregados_levels`, `localities` (existing code is solid).
  * Compute and store **table-level embeddings** using the existing `embeddings` table with `entity_type='agregado'`. (You already do this.)
* **Add**:

  * Populate `table_titles_fts` per table (delete old rows for this `table_id`, then insert one row with `{table_id, title, survey, subject}`).
* **Remove**:

  * Any VA-target building/embedding code paths (already mostly isolated behind flags).
* **Call `links`**:

  * After the insert transaction commits, call `build_links_for_table(table_id)` (synchronously or via a small task pool; either is fine for our scale).

---

## 19.8 CLI: new, slim commands

Rewrite `cli.py` → `sidra_search/cli/main.py` with only:

* `db migrate` → calls `apply_base_schema` + `apply_search_schema` and prints versions
* `db stats` → counts only **base** tables + link tables + titles FTS rows
* `ingest <ids...>` → ingest tables; flags: `--concurrent`, `--skip-embeddings` (still controls agregados embeddings)
* `ingest-coverage` → keep (unchanged behavior)
* `index build-links [--ids … | --all]` → keep (uses new module path)
* `list` → keep
* `show table <id>` → keep
* `search tables` → keep, but:

  * `--q/--title` for title query (lexical and/or semantic)
  * `--var`, `--class` (repeatable) with fuzzy on by default
  * `--coverage` (boolean expression)
  * `--no-fuzzy`, `--var-th`, `--class-th`, `--semantic`
* `diagnostics` → keep **only** base metadata checks (remove VA smoke tests)

**Delete CLI subcommands**:

* `index build-va`, `index embed-va`, `index rebuild-fts` (VA)
* `search va` (VA)
* `link neighbors` (VA)
* `index synonyms import/export` (drop; keep only normalization)

---

## 19.9 Fuzzy 3-gram module hardening

`sidra_search/search/fuzzy3gram.py` (from `fuzzy.py`):

* Keep **in-RAM** corpus of normalized names (vars/classes).
* Build corpus on first call; add `reset_cache()` function that **must** be called when:

  * New tables are ingested
  * Links rebuilt
* Public API:

  * `similar_keys(kind: Literal["var","class"], query: str, *, threshold: float, top_k:int=20) -> list[tuple[key,score]]`
  * Uses `normalize_basic` consistently (moved to a single module).

---

## 19.10 Cutover & compatibility

* Keep the old package `sidra_va` with a **minimal stub**:

  * `__init__.py` prints a deprecation warning and re-exports `get_settings` and the CLI entry pointing to `sidra_search.cli.main`.
  * Optionally keep a hidden alias `sidra-va` CLI for a couple of releases that just calls the new CLI.
* Add a **one-time migration command**:

  ```
  sidra-search db migrate
  sidra-search index build-links --all
  sidra-search ingest-coverage ... (if needed)
  sidra-search search tables --q "..." ...
  ```
* Provide an **optional prune** command to drop VA tables (docs only; not required):

  * Tables to prune: `value_atoms`, `value_atom_dims`, `value_atoms_fts`, `variable_fingerprints`, `synonyms` (if not used), any VA indexes.

---

## 19.11 Config/env cleanup

In `sidra_search/config.py`:

* Keep:

  * `embedding_api_url`, `embedding_model` (used for **table** embeddings & semantic title search)
  * `sidra_base_url`, request timeouts/retries
  * `database_timeout`, `municipality_national_threshold`
* Remove:

  * Any VA-specific toggles (none obvious beyond the above)
* Add:

  * `SIDRA_SEARCH_ENABLE_TITLES_FTS=1` (default 1)
  * `SIDRA_SEARCH_ENABLE_TITLE_EMBEDDINGS=1` (default 1)

---

## 19.12 Precise edit list (by current file)

* `src/sidra_va/__init__.py`

  * Replace exports: **remove** VA exports. Add a deprecation notice or re-export `sidra_search` CLI.

* `src/sidra_va/api_client.py`

  * **Move** to `sidra_search/net/api_client.py`. Update imports in all moved modules.

* `src/sidra_va/base_schema.py`

  * **Move** to `sidra_search/db/base_schema.py` unchanged.

* `src/sidra_va/schema_migrations.py`

  * **Freeze** at current state (for historical VA). Do **not** call from new code.
  * Create new `sidra_search/db/search_schema.py` and `migrations.py` as in 19.4.

* `src/sidra_va/db.py`

  * **Replace** usage across code with `sidra_search/db/session.py`.
  * New `ensure_full_schema()` should call `apply_base_schema` + `apply_search_schema`.

* `src/sidra_va/ingest_base.py`

  * **Copy** to `sidra_search/ingest/ingest_table.py`.
  * Remove all references to VA embeddings and VA tables.
  * Add: FTS upsert into `table_titles_fts` (delete by `table_id`, then insert).
  * At the end, call `build_links_for_table(table_id)` (new module path).
  * Keep table-level embeddings (`embeddings` with `entity_type='agregado'`).

* `src/sidra_va/bulk_ingest.py`

  * **Copy** to `sidra_search/ingest/bulk.py`.
  * Remove references to VA embedding generation; keep `generate_embeddings` flag to control **table** embeddings (not VA).
  * Keep coverage probing logic.

* `src/sidra_va/links.py`

  * **Move** to `sidra_search/ingest/links.py`.
  * Replace normalization import with `sidra_search/search/normalize.py`.
  * Implement “pairing hook” (see 19.5). Keep cross-product fallback.

* `src/sidra_va/fuzzy.py`

  * **Move/rename** to `sidra_search/search/fuzzy3gram.py`.
  * Ensure it uses **only** `variables.nome` & `classifications.nome` via `normalize_basic`.

* `src/sidra_va/search_tables.py`

  * **Move** to `sidra_search/search/tables.py`.
  * **Delete** queries to `value_atoms_fts`. Use `table_titles_fts`.
  * Keep structural filtering as is.
  * Keep optional semantic ranking (embeddings on `agregado`).

* `src/sidra_va/coverage.py`

  * **Move** to `sidra_search/search/coverage.py` (adjust imports).

* `src/sidra_va/synonyms.py`

  * **Split** or **shrink**: keep **only** `normalize_basic` (and tiny helpers). Delete synonym map + CLI import/export.
  * Update all modules to import normalization from `sidra_search/search/normalize.py` (single place).

* `src/sidra_va/cli.py`

  * **Rebuild** as `sidra_search/cli/main.py` with commands listed in 19.8.
  * **Remove**: `cmd_index_build` (VA), `cmd_index_embed`, `cmd_index_rebuild_fts`, `cmd_search_va`, `cmd_link_neighbors`, `cmd_synonyms_*`, VA stats in `cmd_db_stats`, VA smoke tests in `cmd_diagnostics`.
  * **Keep/Adjust**: `cmd_ingest`, `cmd_ingest_coverage`, `cmd_list_agregados`, `cmd_show_table`, `cmd_search_tables` (updated to call new `tables.search_tables`), `cmd_db_migrate`, `cmd_db_stats` (base/link/fts stats only), `cmd_repair_missing` (still valid).

* `src/sidra_va/embedding_client.py`

  * **Move** to `sidra_search/net/embedding_client.py` (unchanged); still used for table embeddings & semantic ranking.

* `src/sidra_va/scoring.py`

  * **Move** `rrf()` into `sidra_search/search/title_rank.py`. Delete `StructureMatch` (VA-specific).

* `src/sidra_va/catalog.py`, `src/sidra_va/diagnostics_base.py`

  * Keep but **move** into `sidra_search/cli/support_*.py` or inline into CLI.

* **Delete** entirely:

  * `value_index.py`, `search_va.py`, `embed.py`, `neighbors.py`, `fingerprints.py`, VA CLI code paths.

---

## 19.13 Tests to write/update

1. **Normalization**

   * Round-trip examples (accents, hyphens, whitespace) → key stability.

2. **Link building**

   * For a tiny fixture table, ensure `name_keys`, `link_var`, `link_class`, `link_cat`, `link_var_class` fill as expected.
   * When re-ingesting the same table, links are **replaced** (no duplicates).

3. **Fuzzy**

   * `similar_keys("var", "pessoas", th=.85)` returns the expected keys.
   * Cache rebuild after ingest (call `reset_cache()` in test).

4. **Title FTS**

   * After ingest, `table_titles_fts` has one row per table; lexical query returns expected table IDs ordered.

5. **Semantic title**

   * With a fake embedding client (deterministic vectors), cosine ranks are computed and combined with RRF.

6. **Search end-to-end**

   * `--var pessoas --class sexo` → returns only tables that satisfy **enforced var×class** (strict originals).
   * Fuzzy expansions broaden recall; “≈” markers appear in `why`.
   * Coverage filters include/exclude as expected.

7. **CLI smoke**

   * `db migrate`, `ingest`, `index build-links --all`, `search tables --q ...` succeed on fixtures.

---

## 19.14 One-time migration (operator checklist)

1. Upgrade code and install.

2. Run:

   ```
   sidra-search db migrate
   sidra-search index build-links --all
   ```

3. (Optional) Re-embed table titles if you want semantic title search:

   ```
   sidra-search ingest --skip-embeddings   # if you want to defer
   # or re-run ingest for existing tables to populate table-level embeddings
   ```

4. Validate searches:

   ```
   sidra-search search tables --var "pessoas" --class "residência"
   sidra-search search tables --q "agrotóxicos"
   ```

5. (Optional) Prune VA tables after verifying (manual SQL or separate admin script).

---

## 19.15 Known gaps & TODO hooks

* **Var×Class actual pairing inference**: The hook exists; initial behavior is cross-product. Add a parser over `classificacoes[*].sumarizacao` / `excecao` to narrow pairs where metadata allows.
* **Caching**: `fuzzy3gram` exposes `reset_cache()`; call it after a batch ingest/build-links if you keep the process warm.
* **Embeddings**: If the external service is down, fail **softly** (skip semantic ranking; still run lexical and structural).

---

## 19.16 “Definition of Done” for the migration

* No module in the new CLI imports **any** VA file or table.
* `search tables` produces identical/better results without `value_atoms*`.
* Database contains `table_titles_fts` and populated rows for all ingested tables.
* Link tables exist and are populated for all ingested tables.
* Old VA commands are gone; help text is concise and up to date.
* The repo tree matches 19.1.

---

This is everything you need to stand up the **clean search-only** codebase and retire VA baggage with minimal risk.
