from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from array import array
from dataclasses import asdict
from typing import Sequence

import orjson

from ..config import get_settings
from ..db.session import ensure_full_schema, sqlite_session
from ..ingest.bulk import ingest_by_coverage
from ..ingest.ingest_table import _canonical_table_text  # reuse identical text
from ..ingest.ingest_table import ingest_table
from ..ingest.links import build_links_for_table
from ..net.embedding_client import EmbeddingClient
from ..search.fuzzy3gram import reset_cache
from ..search.tables import SearchArgs, search_tables
from ..search.where_expr import parse_where_expr


CLI_MANUAL = """\
sidra-search manual
====================

Common commands:
  python -m sidra_search.cli db migrate
      Ensure both base and search schemas are present in the local SQLite database.

  python -m sidra_search.cli ingest-coverage --coverage "N3 OR (N6>=5000)" --limit 10
      Discover tables that satisfy a coverage expression (boolean logic over N-level counts),
      probing SIDRA as needed, and ingest them locally. Combine with --survey-contains or
      --subject-contains to narrow the catalog textually.

  python -m sidra_search.cli build-links --all
      Rebuild search link tables (variables, classifications, categories) for every ingested table.

  python -m sidra_search.cli embed-titles --only-missing
      Refresh semantic embeddings for table titles. Required before using --semantic search.

  python -m sidra_search.cli search --q 'title~"taxa" AND (N6>=5000)'
      Search with the unified boolean query language (facets + coverage). Combine with --semantic
      to blend embeddings with lexical and structural ranking when TITLE literals are present.
      SURVEY and SUBJECT terms act as catalog filters only. For categories, use cat~"Nome" to
      match any class containing that category, or cat~"Class::Nome" to require the exact
      class/category pair. "Contains" (~) checks use normalized, accent-stripped substring
      comparisons, while internal prefilters on VAR/CLASS/CAT link keys stay exact for precision.
      Add --explain to show match reasons.

Legacy flags like --title/--var/--class/--coverage are still accepted but translate into --q behind the scenes.

Flags worth noting:
  --json            Output structured JSON for scripting.
  --debug-fuzzy     Inspect fuzzy expansions for variables/classes during search.
  --no-fuzzy        Disable fuzzy expansions (exact key matching only).

Environment hints:
  SIDRA_SEARCH_ENABLE_TITLE_EMBEDDINGS=1 enables semantic title ranking once embeddings exist.
  SIDRA_DATABASE_PATH can be set to relocate the SQLite database file.

Run `python -m sidra_search.cli <command> --help` for command-specific options.
"""

def _print_json(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


# ---------------------------
# Manual / help
# ---------------------------
def _cmd_manual(_args: argparse.Namespace) -> None:
    print(CLI_MANUAL)


# ---------------------------
# DB admin
# ---------------------------
def _cmd_db_migrate(args: argparse.Namespace) -> None:
    ensure_full_schema()
    print("Database schema ensured (base + search).")


def _cmd_db_stats(args: argparse.Namespace) -> None:
    ensure_full_schema()
    with sqlite_session() as conn:
        counts = {}
        def c(table: str) -> int:
            try:
                return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            except Exception:
                return 0

        counts["agregados"] = c("agregados")
        counts["variables"] = c("variables")
        counts["classifications"] = c("classifications")
        counts["categories"] = c("categories")
        counts["periods"] = c("periods")
        counts["agregados_levels"] = c("agregados_levels")
        counts["localities"] = c("localities")
        counts["name_keys"] = c("name_keys")
        counts["link_var"] = c("link_var")
        counts["link_class"] = c("link_class")
        counts["link_cat"] = c("link_cat")
        counts["link_var_class"] = c("link_var_class")
        counts["table_titles_fts"] = c("table_titles_fts")
        try:
            counts["embeddings_agregado"] = int(
                conn.execute(
                    "SELECT COUNT(*) FROM embeddings WHERE entity_type='agregado'"
                ).fetchone()[0]
            )
        except Exception:
            counts["embeddings_agregado"] = 0

    _print_json(counts)


# ---------------------------
# Ingest
# ---------------------------
def _cmd_ingest(args: argparse.Namespace) -> None:
    ensure_full_schema()
    async def run():
        for tid in args.table_ids:
            try:
                await ingest_table(int(tid))
                print(f"ingested {tid}")
            except Exception as exc:
                print(f"failed {tid}: {exc}")
    asyncio.run(run())


def _cmd_ingest_coverage(args: argparse.Namespace) -> None:
    ensure_full_schema()
    report = asyncio.run(
        ingest_by_coverage(
            coverage=args.coverage,
            subject_contains=args.subject_contains,
            survey_contains=args.survey_contains,
            limit=args.limit,
            concurrency=args.concurrent,
            probe_concurrent=args.probe_concurrent,
        )
    )
    _print_json(asdict(report))


# ---------------------------
# Index / links
# ---------------------------
def _all_table_ids() -> list[int]:
    with sqlite_session() as conn:
        rows = conn.execute("SELECT id FROM agregados ORDER BY id").fetchall()
        return [int(r[0]) for r in rows]

def _cmd_build_links(args: argparse.Namespace) -> None:
    ensure_full_schema()
    table_ids = _all_table_ids() if args.all else args.table_ids
    if not table_ids:
        print("No table IDs provided.")
        return
    for tid in table_ids:
        c = build_links_for_table(int(tid))
        print(f"{tid}: vars={c.vars} classes={c.classes} cats={c.cats} var×class={c.var_class}")
    reset_cache()
    print("fuzzy cache reset")


# ---------------------------
# Search
# ---------------------------
def _quote_literal(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', "\\\"")


def _legacy_query(args: argparse.Namespace) -> str | None:
    parts: list[str] = []
    if args.title:
        parts.append(f'title~"{_quote_literal(args.title)}"')
    for v in args.var or ():
        parts.append(f'var~"{_quote_literal(v)}"')
    for spec in args.cls or ():
        raw = spec.strip()
        if not raw:
            continue
        if ":" in raw:
            class_part, cat_part = raw.split(":", 1)
            class_part = class_part.strip()
            cat_part = cat_part.strip()
        else:
            class_part, cat_part = raw, ""
        if class_part:
            parts.append(f'class~"{_quote_literal(class_part)}"')
        if class_part and cat_part:
            combo = f"{class_part}::{cat_part}"
            parts.append(f'cat~"{_quote_literal(combo)}"')
    if args.coverage:
        parts.append(f"({args.coverage})")
    if args.survey_contains:
        parts.append(f'survey~"{_quote_literal(args.survey_contains)}"')
    if args.subject_contains:
        parts.append(f'subject~"{_quote_literal(args.subject_contains)}"')
    if parts:
        return " AND ".join(parts)
    return None


def _cmd_search_tables(args: argparse.Namespace) -> None:
    ensure_full_schema()

    q_expr = args.q.strip() if args.q and args.q.strip() else None
    legacy_expr = _legacy_query(args)
    if legacy_expr:
        print("NOTE: --title/--var/--class/--coverage flags are deprecated; use --q instead.")
        q_expr = f"({q_expr}) AND ({legacy_expr})" if q_expr else legacy_expr

    where = None
    if q_expr:
        try:
            where = parse_where_expr(q_expr)
        except Exception as exc:  # pragma: no cover - user input error
            print(f"invalid query (--q): {exc}")
            sys.exit(1)

    sargs = SearchArgs(
        q=q_expr,
        where=where,
        limit=max(1, args.limit),
        allow_fuzzy=not args.no_fuzzy,
        var_th=float(args.var_th),
        class_th=float(args.class_th),
        semantic=bool(args.semantic),
        debug_fuzzy=bool(args.debug_fuzzy),
    )

    emb = EmbeddingClient() if args.semantic else None
    hits = asyncio.run(search_tables(sargs, embedding_client=emb))
    if args.json:
        _print_json([
            {
                "id": h.table_id,
                "title": h.title,
                "period_start": h.period_start,
                "period_end": h.period_end,
                "n3": h.n3,
                "n6": h.n6,
                "why": h.why,
                "score": h.score,
                "rrf_score": h.rrf_score,
                "struct_score": h.struct_score,
            }
            for h in hits
        ])
        return
    if not hits:
        print("No results.")
        return
    with sqlite_session() as _conn:
        for h in hits:
            period = ""
            if h.period_start or h.period_end:
                if h.period_start and h.period_end and h.period_start != h.period_end:
                    period = f" | {h.period_start}–{h.period_end}"
                else:
                    period = f" | {h.period_start or h.period_end}"
            cov = f" | N3={h.n3} N6={h.n6}" if (h.n3 or h.n6) else ""
            print(f"{h.table_id}: {h.title}{period}{cov}")

            if args.show_classes:
                names = [r[0] for r in _conn.execute(
                    "SELECT nome FROM classifications WHERE agregado_id=? ORDER BY id LIMIT 3", (h.table_id,)
                ).fetchall()]
                if names:
                    print("  classes:", "; ".join(names))

            if args.explain and h.why:
                print("  matches:", " ".join(f"[{w}]" for w in h.why))
                print(f"  score={h.score:.3f} (struct={h.struct_score:.3f}, rrf={h.rrf_score:.3f})")


# ---------------------------
# Embed
# ---------------------------

def _vec_to_blob(vec):
    arr = array("f", (float(x) for x in vec))
    return arr.tobytes()

def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def _cmd_embed_titles(args: argparse.Namespace) -> None:
    ensure_full_schema()
    s = get_settings()
    model = args.model or s.embedding_model
    only_missing = bool(args.only_missing)
    limit = args.limit if args.limit and args.limit > 0 else None

    emb = EmbeddingClient()

    # Iterate agregados and upsert embeddings if missing/stale
    with sqlite_session() as conn:
        rows = conn.execute(
            "SELECT id, raw_json FROM agregados ORDER BY id"
        ).fetchall()

    count = 0
    updated = 0
    for r in rows:
        if limit is not None and count >= limit:
            break
        tid = int(r["id"])
        try:
            md = orjson.loads(r["raw_json"])
        except Exception:
            continue

        text = _canonical_table_text(md)
        if not text.strip():
            continue

        text_hash = _sha256_text(text)

        # Check existing
        with sqlite_session() as conn:
            cur = conn.execute(
                "SELECT text_hash FROM embeddings WHERE entity_type='agregado' AND entity_id=? AND model=?",
                (str(tid), model),
            ).fetchone()
            existing_hash = cur["text_hash"] if cur else None

        if existing_hash == text_hash:
            count += 1
            continue
        if only_missing and existing_hash is not None:
            count += 1
            continue

        # Compute vector
        try:
            vec = emb.embed_text(text, model=model)
        except Exception as exc:
            print(f"embed failed for {tid}: {exc}")
            count += 1
            continue

        # Upsert
        with sqlite_session() as conn:
            with conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO embeddings(
                      entity_type, entity_id, agregado_id, text_hash, model, dimension, vector, created_at
                    ) VALUES(?,?,?,?,?,?,?,datetime('now'))
                    """,
                    (
                        "agregado",
                        str(tid),
                        tid,
                        text_hash,
                        model,
                        len(vec),
                        _vec_to_blob(vec),
                    ),
                )
        updated += 1
        count += 1

    print(f"embed-titles: scanned={count}, updated={updated}, model={model}")



def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sidra-search", description="Table-centric search & ingestion")
    p.add_argument("--manual", action="store_true", help="Print a concise CLI usage guide and exit")
    sub = p.add_subparsers(dest="cmd")

    # db
    dbmig = sub.add_parser("db", help="Database utilities")
    dbsub = dbmig.add_subparsers(dest="db_cmd")

    db_migrate = dbsub.add_parser("migrate", help="Apply base + search schema")
    db_migrate.set_defaults(func=_cmd_db_migrate)

    db_stats = dbsub.add_parser("stats", help="Show row counts for key tables")
    db_stats.set_defaults(func=_cmd_db_stats)

    # ingest single/many
    ig = sub.add_parser("ingest", help="Ingest one or more table IDs")
    ig.add_argument("table_ids", type=int, nargs="+")
    ig.set_defaults(func=_cmd_ingest)

    # ingest by coverage discovery
    ic = sub.add_parser("ingest-coverage", help="Discover by coverage and ingest")
    ic.add_argument("--coverage", required=True, help='Boolean expr, e.g. "N3 OR (N6>=5000)"')
    ic.add_argument("--subject-contains", dest="subject_contains", help="Filter catalog by subject name substring")
    ic.add_argument("--survey-contains", dest="survey_contains", help="Filter catalog by survey name substring")
    ic.add_argument("--limit", type=int, default=None)
    ic.add_argument("--concurrent", type=int, default=8, help="ingestion concurrency")
    ic.add_argument("--probe-concurrent", type=int, default=None, help="coverage probe concurrency (defaults to --concurrent)")
    ic.set_defaults(func=_cmd_ingest_coverage)

    # build links
    bl = sub.add_parser("build-links", help="(Re)build link indexes for tables")
    bl.add_argument("table_ids", type=int, nargs="*", help="Specific table IDs")
    bl.add_argument("--all", action="store_true", help="Process all ingested tables")
    bl.set_defaults(func=_cmd_build_links)

    # search
    st = sub.add_parser("search", help="Search tables with unified boolean queries")
    st.add_argument(
        "--q",
        dest="q",
        help="Unified boolean query (e.g. 'title~\"taxa\" AND (N6>=5000)')",
    )
    st.add_argument(
        "--title",
        dest="title",
        help='[deprecated] Title filter; prefer --q \'title~"..."\'',
    )
    st.add_argument(
        "--survey-contains",
        dest="survey_contains",
        help="[deprecated] Survey substring filter; use --q",
    )
    st.add_argument(
        "--subject-contains",
        dest="subject_contains",
        help="[deprecated] Subject substring filter; use --q",
    )
    st.add_argument(
        "--var",
        dest="var",
        action="append",
        help="[deprecated] Variable name (repeatable); use --q",
    )
    st.add_argument(
        "--class",
        dest="cls",
        action="append",
        help="[deprecated] Class or 'Class:Category' (repeatable); use --q",
    )
    st.add_argument(
        "--coverage",
        help="[deprecated] Coverage expression; include directly in --q",
    )
    st.add_argument("--limit", type=int, default=20)
    st.add_argument("--no-fuzzy", action="store_true")
    st.add_argument("--var-th", type=float, default=0.74)
    st.add_argument("--class-th", type=float, default=0.78)
    st.add_argument("--semantic", action="store_true", help="use semantic title ranking (requires embeddings)")
    st.add_argument("--explain", action="store_true", help="print match rationale and scores")
    st.add_argument("--json", action="store_true")
    st.add_argument("--show-classes", action="store_true", help="List up to 3 classification names for each hit")
    st.add_argument("--debug-fuzzy", action="store_true", help="Print fuzzy expansions and candidate counts")
    st.set_defaults(func=_cmd_search_tables)

    # embeddings backfill (ADD BEFORE RETURN)
    et = sub.add_parser("embed-titles", help="(Re)embed table titles (idempotent)")
    et.add_argument("--model", help="Embedding model name (defaults to settings)")
    et.add_argument("--only-missing", action="store_true", help="Skip rows that already have an embedding, even if text changed")
    et.add_argument("--limit", type=int, default=None, help="Limit number of tables processed")
    et.set_defaults(func=_cmd_embed_titles)

    return p

def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if getattr(args, "manual", False):
        _cmd_manual(args)
        return

    # Support nested: "db migrate"/"db stats"
    if getattr(args, "cmd", None) == "db" and not hasattr(args, "func"):
        parser.parse_args(["db", "-h"])
        return

    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
