# SIDRA Search — Unified CLI + Name-Key Linking

## 0) Goal (what we’re building)

A **CLI search** that returns **tables** from SIDRA by:

* free-text on table titles,
* **variables**, **classifications**, and optional **categories** (individually or combined),
* with optional **coverage** filters (e.g., `(N3>=27) OR (N6>=5000)`).

We identify “the same thing across tables” by **names**, not IDs. We do **strict** name equality by default and allow **fuzzy** name matches for variables and classes (categories are strict).

We **do not** special-case units, percent vs. absolute, etc. (We show them; users can filter by unit text if they want.)

---

## 1) Core principles

* **Names, not IDs, define identity across tables.** Treat SIDRA numeric IDs as **table-local** (for API calls only).
* **Normalization** gives stable keys:

  * `var_key = norm(var_name)`, `class_key = norm(class_name)`, `cat_key = norm(category_name)`.
  * `norm` = lowercase, strip accents, trim, collapse spaces, drop trivial punctuation.
* **Links, not joins.** Build fast inverted indexes (`link_*`) from normalized names to tables.
* **AND semantics.** Multiple constraints must all hold. If a variable and classes are provided, each class must be a **real breakdown** of that variable in the table.

---

## 2) Minimal storage we need (metadata only)

Keep your existing `agregados` table. Add/ensure these (names + normalized keys; keep original SIDRA IDs to call the API later):

* `variables(table_id, variable_id, name, unit, var_key)`
* `classifications(table_id, class_id, name, class_key)`
* `categories(table_id, class_id, category_id, name, cat_key)`

**Inverted indexes (link tables):**

* `link_var(var_key, table_id, variable_id)`
* `link_class(class_key, table_id, class_id)`
* `link_cat(class_key, cat_key, table_id, class_id, category_id)`
* `link_var_class(var_key, class_key, table_id, variable_id, class_id)`
  *(This is crucial to prove a class actually breaks a given variable in that table.)*

**Search helpers:**

* `table_titles_fts`: FTS index on table titles (and optional description).
* Optional: `title_embedding` per table (if you’re already generating embeddings).

**Coverage table:**
`coverage(table_id, N1_count, N2_count, N3_count, N6_count, ...)` produced by your probe step.

---

## 3) Normalization spec (deterministic)

* Lowercase.
* Remove accents/diacritics.
* Trim leading/trailing spaces.
* Collapse internal whitespace to single space.
* Drop punctuation that has no semantic effect (commas, periods, parentheses around nothing, duplicated dashes). Keep hyphens that are part of tokens if they matter (but normalize multiple “ - ” variants to a single space).
* Do **not** translate words. Keep Portuguese as-is.

Examples:

* `"Pessoas que frequentavam escola ou creche"` → `pessoas que frequentavam escola ou creche`
* `"Grupo de idade"` / `"Idade"` remain distinct. Fuzzy may bridge them; strict will not.

---

## 4) Building the indexes (during ingest)

For each table:

1. Extract variables, classifications, categories.
2. Compute `var_key`, `class_key`, `cat_key`.
3. Insert into `variables`, `classifications`, `categories`.
4. Create links:

   * `link_var(var_key, table_id, variable_id)`
   * `link_class(class_key, table_id, class_id)`
   * `link_cat(class_key, cat_key, table_id, class_id, category_id)`
   * For each variable × classification actually paired in the table schema, add `link_var_class(var_key, class_key, table_id, variable_id, class_id)`.
5. Update `table_titles_fts`.
6. Update `coverage` (from your probe results).

---

## 5) CLI (one unified command)

```
python -m sidra_va.cli search
  [--q "free text"]           # semantic/lexical on table titles
  [--var "variable name"]     # may repeat (AND)
  [--class "class[:category]"]# may repeat (AND); category optional
  [--coverage "(N3>=27) OR (N6>=5000)"]
  [--no-fuzzy | --var-th 0.90 | --class-th 0.85]
  [--limit 50]
```

* **`--q`**: hybrid rank on titles (embedding + BM25/TF-IDF).
* **`--var`**: match by variable name; strict + fuzzy (threshold configurable).
* **`--class`**: match by classification; optional `:category`. Strict + fuzzy for the **class name**, **category strict**.
* **Multiple `--class`** allowed.
  If `--var` is present, the table must have each `(--var, --class)` pair in **link_var_class**.
  If a class has `:category`, require that category via **link_cat**.
* **Coverage** is a post-filter.
* **Fuzzy** defaults: var ≥ **0.90**, class ≥ **0.85**. Disable with `--no-fuzzy`.

---

## 6) Query flow (fast and predictable)

**Step A – Interpret & normalize inputs**

* Normalize all provided names using `norm()`.
* For each var/class, compute both strict key and candidates via fuzzy (if enabled).

**Step B – Candidate set (set intersections)**

* Start from all tables.
* If `--var` present: intersect with union(strict, fuzzy) from `link_var`.
* For each `--class`:

  * If class only: intersect with union(strict, fuzzy) from `link_class`.
  * If `:category`: intersect with **strict** `link_cat`.
* If **both var and classes** are present: intersect with `link_var_class` for each `(var, class)` (strict or fuzzy class name resolves to one or more `class_key` candidates).
* Apply `--coverage` boolean expression on `coverage` table.

**Step C – Ranking**
For remaining candidates, compute:

* **Title score** vs `--q` (if provided): embedding cosine + BM25; combine via RRF or weighted sum.
* **Structure score:**

  * * strong bonus for **strict** matches on var/class.
  * * smaller bonus for **fuzzy** matches.
  * * small bonus if **all requested classes** appear for the **same variable** (when `--var` used).
* Final score = Title score (if `--q`) + Structure score.
* Sort desc, cap by `--limit`.

---

## 7) Output format (each result = one table)

```
{table_id} | {title} | years:{start}–{end} | cov:N3={x}/N6={y}
matches: [var: "..."] [class: "..."] [class: "..." : "category"] ...
```

* Mark fuzzy with `≈`: e.g., `[class≈ "grupo de idade"]`.
* Show unit next to variable in brackets if helpful: `[var: "... (Pessoas)"]`.
* Always show which parts matched so users can copy exact names.

---

## 8) Examples (realistic)

**A) Broad idea first**

```
search --q "educação por faixa etária" --limit 20
```

**B) Same variable across tables**

```
search --var "Pessoas que frequentavam escola ou creche"
```

**C) Variable broken by a class (no category)**

```
search --var   "Pessoas que frequentavam escola ou creche" \
       --class "Grupo de idade"
```

**D) Add another class**

```
search --var   "Pessoas que frequentavam escola ou creche" \
       --class "Nível de ensino ou curso que frequentavam" \
       --class "Situação do domicílio"
```

**E) Pin categories (single bins)**

```
search --var   "Pessoas que frequentavam escola ou creche" \
       --class "Situação do domicílio:Urbana" \
       --class "Grupo de idade:4 anos"
```

**F) Coverage**

```
search --var "Pessoas que frequentavam escola ou creche" \
       --class "Grupo de idade" \
       --coverage "(N6>=5000)"
```

**G) Tighten or kill fuzzy**

```
search --var "pessoal ocupado total" --no-fuzzy
# or
search --var "pessoal ocupado total" --var-th 0.92
```

---

## 9) Fuzzy matching (simple, contained)

* **Where**: variables and classes only.
  Categories are **strict** (short labels; fuzziness adds noise).
* **How**: character **3-gram TF-IDF cosine** on raw names (pre-norm string). Cache vectors.
* **Thresholds**: var ≥ 0.90 (default), class ≥ 0.85 (default). CLI overrides allowed.
* **Tie-breaking**: prefer strict exact key; then highest cosine. Cap to top-K (e.g., 10) before set intersections for speed.

---

## 10) Migration from the “global ID” mistake

1. **Schema**: ensure variable/class/category rows are keyed by `(table_id, *_id)` (or internal rowid) — not global.
2. **Re-ingest metadata** (or backfill from JSON cache) to repopulate `variables`, `classifications`, `categories`.
3. **Build links**: fill `link_var`, `link_class`, `link_cat`, `link_var_class`.
4. **Recompute coverage** using your probe.
5. **FTS/embeddings**: rebuild for titles.

**Sanity checks**

* Count how many tables have ≥1 variable/class after rebuild.
* Spot-check:

  * `SELECT * FROM link_var WHERE var_key='pessoas que frequentavam escola ou creche'` → should list all relevant tables.
  * `SELECT * FROM link_var_class WHERE var_key='...' AND class_key='grupo de idade'` → proves the pairing exists.
  * Random tables: verify classes/categories present.

---

## 11) Non-goals / constraints

* No manual taxonomies, no unit families, no “percent vs absolute” re-labeling.
* No web UI—**CLI only**.
* We don’t aggregate data values here; we’re **finding tables** (and where specific var/class(/cat) combinations exist).

---

## 12) What this enables (user intent)

* Start with **idea** (`--q`), then pivot to **structure** (`--var`, `--class`) without switching modes.
* Find **the same variable** across many tables—optionally constrained by classes/categories and coverage.
* Identify which tables truly expose **var + multiple classes** together (the `link_var_class` guarantees this).

---

## 13) Optional future (if needed later)

* Add fuzzy for **categories** behind a flag (rarely useful).
* Add `--unit "Pessoas"` filter (plain string filter; still no unit ontology).
* Add `explain` command to show which links made a match pass.

---

**Bottom line:**

* **Names = identity across tables.**
* **Links = fast set intersections** to honor AND semantics.
* **One CLI** handles free-text and structural filters together.
* **Fuzzy is minimal and controllable** (vars/classes only).
