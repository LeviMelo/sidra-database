from __future__ import annotations

import asyncio
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from array import array

from ..config import get_settings
from ..db.migrations import apply_search_schema
from ..db.session import create_connection
from ..net.embedding_client import EmbeddingClient
from ..search.normalize import normalize_basic
from ..search.title_rank import rrf
from .where_eval import eval_where
from .where_expr import WhereNode, iter_contains_literals, _Contains, _StrLit, _And, _Or, _Not 
from ..search.fuzzy3gram import similar_keys


@dataclass(frozen=True)
class TableHit:
    table_id: int
    title: str
    period_start: str | None
    period_end: str | None
    n3: int
    n6: int
    why: List[str]
    score: float
    rrf_score: float
    struct_score: float


@dataclass(frozen=True)
class SearchArgs:
    q: str | None
    where: WhereNode | None
    limit: int
    allow_fuzzy: bool
    var_th: float
    class_th: float
    semantic: bool
    debug_fuzzy: bool


@dataclass
class _LiteralQuery:
    raw: str
    normalized: str
    scores: Dict[str, float]


@dataclass
class _TableContext:
    title: str
    title_norm: str
    survey: str
    survey_norm: str
    subject: str
    subject_norm: str
    vars: List[str]
    classes: List[str]
    cats: List[str]
    cat_any: Set[str]
    class_cat_map: Dict[str, Set[str]]
    var_class_map: Dict[str, Set[str]]
    coverage_counts: Dict[str, int]
    period_years: Set[int]
    period_start: str | None
    period_end: str | None
    n3: int
    n6: int

    @property
    def vars_set(self) -> Set[str]:
        return set(self.vars)

    @property
    def classes_set(self) -> Set[str]:
        return set(self.classes)

    @property
    def cats_set(self) -> Set[str]:
        return set(self.cats)


@dataclass(frozen=True)
class _CatRequirements:
    loose: Set[str]
    strict: Dict[str, Set[str]]
    strict_total: int
    has_any: bool


_YEAR_RE = re.compile(r"(\d{4})")


def _fts_query(text: str) -> str:
    toks = [t for t in normalize_basic(text).split() if t]
    return " ".join(toks)


def _positive_literals(where: WhereNode | None) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    if not where:
        return out
    for field, polarity, text in iter_contains_literals(where):
        if not polarity:
            continue
        if not text:
            continue
        out.setdefault(field.upper(), []).append(text)
    return out

def _strip_title_contains(node: WhereNode) -> WhereNode:
    """
    Return a copy of the WHERE AST where TITLE~(...) constraints are replaced by a tautology.
    This lets us keep other filters (period, var/class/cat, survey/subject) while
    allowing the semantic title layer to take over when lexical title contains fails.
    """
    if isinstance(node, _Contains) and str(node.field).upper() == "TITLE":
        # any non-empty _StrLit evaluates to True in eval_where
        return _StrLit("1")

    if isinstance(node, _Not):
        return _Not(_strip_title_contains(node.node))
    if isinstance(node, _And):
        return _And(_strip_title_contains(node.left), _strip_title_contains(node.right))
    if isinstance(node, _Or):
        return _Or(_strip_title_contains(node.left), _strip_title_contains(node.right))
    # all other nodes unchanged
    return node


def _build_literal_queries(
    field: str,
    literals: Iterable[str],
    *,
    allow_fuzzy: bool,
    threshold: float,
) -> List[_LiteralQuery]:
    seen: Set[str] = set()
    queries: List[_LiteralQuery] = []
    for lit in literals:
        norm = normalize_basic(lit)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        scores: Dict[str, float] = {norm: 1.0}
        if allow_fuzzy:
            kind = "var" if field == "VAR" else "class"
            for key, score in similar_keys(kind, lit, threshold=threshold, top_k=12):
                scores[key] = max(scores.get(key, 0.0), float(score))
        queries.append(_LiteralQuery(raw=lit, normalized=norm, scores=scores))
    return queries


def _prefilter_link_exact(
    conn,
    table: str,
    column: str,
    needles: Iterable[str],
) -> Set[int]:
    values = [normalize_basic(n) for n in needles if normalize_basic(n)]
    if not values:
        return set()
    unique = sorted(set(values))
    ids: Set[int] = set()
    chunk = 500
    for idx in range(0, len(unique), chunk):
        batch = unique[idx : idx + chunk]
        placeholders = ",".join("?" for _ in batch)
        rows = conn.execute(
            f"SELECT DISTINCT table_id FROM {table} WHERE {column} IN ({placeholders})",
            batch,
        ).fetchall()
        ids.update(int(r[0]) for r in rows)
    return ids


def _prefilter_agregado_text(
    conn,
    column: str,
    needles: Iterable[str],
) -> Set[int]:
    # NOTE: still a scan, but cheap relative to per-table context loads.
    normalized = [normalize_basic(n) for n in needles if normalize_basic(n)]
    if not normalized:
        return set()
    rows = conn.execute(f"SELECT id, {column} FROM agregados").fetchall()
    ids: Set[int] = set()
    for row in rows:
        value = normalize_basic(str(row[column] or ""))
        if any(needle in value for needle in normalized):
            ids.add(int(row["id"]))
    return ids


def _prefilter_title_fts(conn, literals: Iterable[str]) -> Set[int]:
    settings = get_settings()
    if not settings.enable_titles_fts:
        return set()

    # Build a broad OR query over individual tokens; also include full phrases
    terms: List[str] = []
    for lit in literals:
        norm = normalize_basic(lit)
        if not norm:
            continue
        tokens = [t for t in norm.split() if t]
        if not tokens:
            continue
        # OR all tokens for recall
        terms.extend(tokens)
        # keep the full phrase too (acts like a bonus signal)
        if len(tokens) > 1:
            terms.append(" ".join(tokens))

    if not terms:
        return set()

    # De-dupe and quote phrases for FTS
    uniq: List[str] = list(dict.fromkeys(terms))
    pieces = [f'"{t}"' if " " in t else t for t in uniq]
    query = " OR ".join(pieces)

    rows = conn.execute(
        "SELECT DISTINCT table_id FROM table_titles_fts WHERE table_titles_fts MATCH ?",
        (query,),
    ).fetchall()
    return {int(r[0]) for r in rows}


def _extract_years(pid: Any, pord: Any) -> Set[int]:
    years: Set[int] = set()
    for value in (pid, pord):
        if value is None:
            continue
        text = str(value)
        match = _YEAR_RE.search(text)
        if match:
            year = int(match.group(1))
            if 1500 <= year <= 2100:
                years.add(year)
    return years


# ------------- NEW: bulk context loader (batched) -------------
def _bulk_load_contexts(conn, table_ids: Sequence[int], *, batch_size: int = 400) -> Dict[int, _TableContext]:
    """
    Load _TableContext for many tables in batches with a handful of queries per batch,
    instead of 6–7 queries per table. This is the core perf fix.
    """
    if not table_ids:
        return {}

    out: Dict[int, _TableContext] = {}
    norm = normalize_basic

    for start in range(0, len(table_ids), batch_size):
        batch = list(table_ids[start : start + batch_size])
        ph = ",".join("?" for _ in batch)

        # 1) headers
        hdr_rows = conn.execute(
            f"""
            SELECT id, nome, pesquisa, assunto, periodo_inicio, periodo_fim
            FROM agregados
            WHERE id IN ({ph})
            """,
            batch,
        ).fetchall()
        # seed contexts
        for r in hdr_rows:
            tid = int(r["id"])
            title = str(r["nome"] or "")
            survey = str(r["pesquisa"] or "")
            subject = str(r["assunto"] or "")
            out[tid] = _TableContext(
                title=title,
                title_norm=norm(title),
                survey=survey,
                survey_norm=norm(survey),
                subject=subject,
                subject_norm=norm(subject),
                vars=[],
                classes=[],
                cats=[],
                cat_any=set(),
                class_cat_map={},
                var_class_map={},
                coverage_counts={},
                period_years=set(),
                period_start=r["periodo_inicio"],
                period_end=r["periodo_fim"],
                n3=0,
                n6=0,
            )

        if not out:
            continue  # none of these batch ids exist

        # 2) link_var
        for r in conn.execute(
            f"SELECT table_id, var_key FROM link_var WHERE table_id IN ({ph})",
            batch,
        ):
            tid = int(r["table_id"])
            if tid not in out:
                continue
            v = norm(str(r["var_key"] or ""))
            if v:
                out[tid].vars.append(v)

        # 3) link_class
        for r in conn.execute(
            f"SELECT table_id, class_key FROM link_class WHERE table_id IN ({ph})",
            batch,
        ):
            tid = int(r["table_id"])
            if tid not in out:
                continue
            c = norm(str(r["class_key"] or ""))
            if c:
                out[tid].classes.append(c)

        # 4) link_var_class
        for r in conn.execute(
            f"SELECT table_id, var_key, class_key FROM link_var_class WHERE table_id IN ({ph})",
            batch,
        ):
            tid = int(r["table_id"])
            if tid not in out:
                continue
            v = norm(str(r["var_key"] or ""))
            c = norm(str(r["class_key"] or ""))
            if v and c:
                out[tid].var_class_map.setdefault(v, set()).add(c)

        # 5) link_cat
        for r in conn.execute(
            f"SELECT table_id, class_key, cat_key FROM link_cat WHERE table_id IN ({ph})",
            batch,
        ):
            tid = int(r["table_id"])
            if tid not in out:
                continue
            ck_raw = str(r["class_key"] or "")
            cat_raw = str(r["cat_key"] or "")
            ck = norm(ck_raw)
            cat = norm(cat_raw)
            if not cat:
                continue
            ctx = out[tid]
            ctx.cat_any.add(cat)
            ctx.cats.append(cat)
            if ck:
                ctx.cats.append(f"{ck}::{cat}")
                ctx.class_cat_map.setdefault(ck, set()).add(cat)

        # 6) coverage / levels
        for r in conn.execute(
            f"SELECT agregado_id, level_id, locality_count FROM agregados_levels WHERE agregado_id IN ({ph})",
            batch,
        ):
            tid = int(r["agregado_id"])
            if tid not in out:
                continue
            lvl = str(r["level_id"]).upper() if r["level_id"] else ""
            cnt = int(r["locality_count"] or 0)
            if lvl:
                out[tid].coverage_counts[lvl] = cnt
                if lvl == "N3":
                    out[tid].n3 = cnt
                elif lvl == "N6":
                    out[tid].n6 = cnt

        # 7) periods → years
        for r in conn.execute(
            f"SELECT agregado_id, periodo_id, periodo_ord FROM periods WHERE agregado_id IN ({ph})",
            batch,
        ):
            tid = int(r["agregado_id"])
            if tid not in out:
                continue
            out[tid].period_years |= _extract_years(r["periodo_id"], r["periodo_ord"])

        # de-dup preserve order for vars/classes/cats (like original)
        for tid, ctx in out.items():
            ctx.vars = list(dict.fromkeys(v for v in ctx.vars if v))
            ctx.classes = list(dict.fromkeys(c for c in ctx.classes if c))
            ctx.cats = list(dict.fromkeys(ctx.cats))

    return out


def _parse_cat_requirements(cat_literals: Iterable[str]) -> _CatRequirements:
    loose: Set[str] = set()
    strict: Dict[str, Set[str]] = {}
    has_any = False
    for lit in cat_literals:
        raw = lit.strip()
        if not raw:
            continue
        has_any = True
        if "::" in raw:
            class_part, cat_part = raw.split("::", 1)
            class_norm = normalize_basic(class_part)
            cat_norm = normalize_basic(cat_part)
            if class_norm and cat_norm:
                strict.setdefault(class_norm, set()).add(cat_norm)
        else:
            cat_norm = normalize_basic(raw)
            if cat_norm:
                loose.add(cat_norm)
    strict_total = sum(len(v) for v in strict.values())
    return _CatRequirements(loose=loose, strict=strict, strict_total=strict_total, has_any=has_any)


def _first_strict_pair(ctx: _TableContext, cat_req: _CatRequirements) -> Optional[str]:
    for class_key, cats in cat_req.strict.items():
        have = ctx.class_cat_map.get(class_key, set())
        for cat in cats:
            if cat in have:
                return f"{class_key}::{cat}"
    return None


def _aggregate_var_scores(queries: List[_LiteralQuery]) -> Tuple[Dict[str, float], Set[str]]:
    scores: Dict[str, float] = {}
    exact: Set[str] = set()
    for q in queries:
        if q.normalized:
            exact.add(q.normalized)
        for key, value in q.scores.items():
            scores[key] = max(scores.get(key, 0.0), float(value))
    return scores, exact


def _build_class_groups(
    queries: List[_LiteralQuery],
) -> Tuple[List[Dict[str, float]], List[str]]:
    groups: List[Dict[str, float]] = []
    exact: List[str] = []
    for q in queries:
        if not q.scores:
            continue
        groups.append({k: float(v) for k, v in q.scores.items()})
        exact.append(q.normalized)
    return groups, exact


def _structural_for_table(
    ctx: _TableContext,
    *,
    var_scores: Dict[str, float],
    var_exact: Set[str],
    class_groups: List[Dict[str, float]],
    class_exact: List[str],
    cat_req: _CatRequirements,
) -> Optional[Tuple[float, List[str]]]:
    # Enforce loose category presence
    for cat in cat_req.loose:
        if cat not in ctx.cat_any:
            return None

    # Enforce strict category presence per class
    for class_key, cats in cat_req.strict.items():
        have = ctx.class_cat_map.get(class_key, set())
        if not cats.issubset(have):
            return None

    has_vars = bool(var_scores)
    has_classes = bool(class_groups)

    strict_exact = cat_req.strict_total

    if has_vars:
        present_vars = [vk for vk in ctx.vars if vk in var_scores]
        if not present_vars:
            return None

        best_struct = -1.0
        best_why: List[str] = []

        for vk in present_vars:
            cls_picks: List[Tuple[str, float, bool]] = []
            required_classes = set(cat_req.strict.keys())
            var_allowed_total = ctx.var_class_map.get(vk, set())
            if required_classes and not required_classes.issubset(var_allowed_total):
                continue
            if has_classes:
                allowed_classes = var_allowed_total
                if not allowed_classes:
                    continue
                ok = True
                for idx, group in enumerate(class_groups):
                    if not group:
                        ok = False
                        break
                    candidates = set(group.keys()) & allowed_classes & ctx.classes_set
                    if not candidates:
                        ok = False
                        break
                    filtered = [
                        ck
                        for ck in candidates
                        if cat_req.strict.get(ck, set()).issubset(ctx.class_cat_map.get(ck, set()))
                    ]
                    if not filtered:
                        ok = False
                        break
                    best_ck = max(filtered, key=lambda ck: group.get(ck, 0.0))
                    cls_picks.append(
                        (
                            best_ck,
                            group.get(best_ck, 0.0),
                            bool(class_exact[idx]) and best_ck == class_exact[idx],
                        )
                    )
                if not ok:
                    continue

            var_score = var_scores.get(vk, 0.0)
            if has_classes and len(class_groups) > 0:
                avg_class = sum(score for (_ck, score, _ex) in cls_picks) / len(class_groups)
            else:
                avg_class = 0.0
            exact_boost = 0.0
            if vk in var_exact:
                exact_boost += 0.10
            exact_boost += 0.05 * sum(1 for (_ck, _s, ex) in cls_picks if ex)
            exact_boost += 0.03 * strict_exact
            struct = 0.55 * var_score + 0.35 * avg_class + exact_boost
            struct = min(1.0, struct)

            why: List[str] = []
            label = "=" if vk in var_exact else "≈"
            why.append(f'var{label}"{vk}"')
            for ck, _score, ex in cls_picks:
                clabel = "=" if ex else "≈"
                why.append(f'class{clabel}"{ck}"')
            if cls_picks:
                why.append("var×class")
            if cat_req.has_any:
                if cat_req.strict:
                    strict_hit = _first_strict_pair(ctx, cat_req)
                    if strict_hit:
                        why.append(f'cat="{strict_hit}"')
                why.append("cat")

            if struct > best_struct:
                best_struct = struct
                best_why = why

        if best_struct < 0:
            return None
        return best_struct, best_why

    if has_classes:
        picks: List[Tuple[str, float, bool]] = []
        for idx, group in enumerate(class_groups):
            if not group:
                return None
            candidates = set(group.keys()) & ctx.classes_set
            if not candidates:
                return None
            filtered = [
                ck
                for ck in candidates
                if cat_req.strict.get(ck, set()).issubset(ctx.class_cat_map.get(ck, set()))
            ]
            if not filtered:
                return None
            best_ck = max(filtered, key=lambda ck: group.get(ck, 0.0))
            picks.append(
                (
                    best_ck,
                    group.get(best_ck, 0.0),
                    bool(class_exact[idx]) and best_ck == class_exact[idx],
                )
            )
        avg_class = sum(score for (_ck, score, _ex) in picks) / len(class_groups)
        exact_boost = 0.05 * sum(1 for (_ck, _s, ex) in picks if ex) + 0.03 * strict_exact
        struct = 0.35 * avg_class + exact_boost
        struct = min(1.0, struct)
        why = [f'class{"=" if ex else "≈"}"{ck}"' for (ck, _s, ex) in picks]
        if cat_req.has_any:
            if cat_req.strict:
                strict_hit = _first_strict_pair(ctx, cat_req)
                if strict_hit:
                    why.append(f'cat="{strict_hit}"')
            why.append("cat")
        return struct, why

    # No var/class filters; only categories may contribute exact boost
    exact_boost = 0.03 * strict_exact
    struct = min(1.0, exact_boost)
    why: List[str] = []
    if cat_req.has_any:
        if cat_req.strict:
            strict_hit = _first_strict_pair(ctx, cat_req)
            if strict_hit:
                why.append(f'cat="{strict_hit}"')
        why.append("cat")
    return struct, why


async def search_tables(
    args: SearchArgs,
    *,
    embedding_client: EmbeddingClient | None = None,
) -> List[TableHit]:
    conn = create_connection()
    try:
        apply_search_schema(conn)

        where_ast = args.where
        positives = _positive_literals(where_ast)

        var_queries = _build_literal_queries(
            "VAR",
            positives.get("VAR", []),
            allow_fuzzy=args.allow_fuzzy,
            threshold=args.var_th,
        )
        class_queries = _build_literal_queries(
            "CLASS",
            positives.get("CLASS", []),
            allow_fuzzy=args.allow_fuzzy,
            threshold=args.class_th,
        )
        cat_literals = positives.get("CAT", [])

        var_scores, var_exact = _aggregate_var_scores(var_queries)
        class_groups, class_exact = _build_class_groups(class_queries)
        cat_req = _parse_cat_requirements(cat_literals)

        if args.debug_fuzzy:
            top_vars = [
                f"{k}:{s:.2f}"
                for q in var_queries
                for k, s in sorted(q.scores.items(), key=lambda x: x[1], reverse=True)[:5]
            ]
            print("[fuzzy] var:", ", ".join(top_vars) or "(none)")
            for idx, q in enumerate(class_queries):
                top_cls = sorted(q.scores.items(), key=lambda x: x[1], reverse=True)[:5]
                print(
                    f"[fuzzy] class[{idx}]:",
                    ", ".join(f"{k}:{s:.2f}" for k, s in top_cls) or "(none)",
                )

        candidates: Optional[Set[int]] = None
        pre_counts: Dict[str, int] = {}

        # Prefilters (positive-only; safe and already in your design)
        var_hints = {normalize_basic(x) for x in positives.get("VAR", []) if normalize_basic(x)}
        if var_hints:
            ids = _prefilter_link_exact(conn, "link_var", "var_key", var_hints)
            if ids:
                candidates = set(ids) if candidates is None else candidates & ids
            pre_counts["var"] = len(ids)

        class_hints = {normalize_basic(x) for x in positives.get("CLASS", []) if normalize_basic(x)}
        if class_hints:
            ids = _prefilter_link_exact(conn, "link_class", "class_key", class_hints)
            if ids:
                candidates = set(ids) if candidates is None else candidates & ids
            pre_counts["class"] = len(ids)

        # categories: loose and strict
        cat_hints = set(cat_req.loose)
        for cats in cat_req.strict.values():
            cat_hints.update(cats)
        if cat_hints:
            ids = _prefilter_link_exact(conn, "link_cat", "cat_key", cat_hints)
            if ids:
                candidates = set(ids) if candidates is None else candidates & ids
            pre_counts["cat"] = len(ids)

        title_hints = positives.get("TITLE", [])
        if title_hints:
            ids = _prefilter_title_fts(conn, title_hints)
            if ids:
                candidates = set(ids) if candidates is None else candidates & ids
            pre_counts["titlefts"] = len(ids)

        survey_hints = positives.get("SURVEY", [])
        if survey_hints:
            ids = _prefilter_agregado_text(conn, "pesquisa", survey_hints)
            if ids:
                candidates = set(ids) if candidates is None else candidates & ids
            pre_counts["survey"] = len(ids)

        subject_hints = positives.get("SUBJECT", [])
        if subject_hints:
            ids = _prefilter_agregado_text(conn, "assunto", subject_hints)
            if ids:
                candidates = set(ids) if candidates is None else candidates & ids
            pre_counts["subject"] = len(ids)

        # Fallback universe if hints produced nothing
        if candidates is None:
            rows = conn.execute("SELECT id FROM agregados").fetchall()
            candidates = {int(r[0]) for r in rows}

        if args.debug_fuzzy and pre_counts:
            print(
                "[pre] candidates≈{} (hints: var={}, class={}, cat={}, titlefts={}, survey={}, subject={})".format(
                    len(candidates),
                    pre_counts.get("var", 0),
                    pre_counts.get("class", 0),
                    pre_counts.get("cat", 0),
                    pre_counts.get("titlefts", 0),
                    pre_counts.get("survey", 0),
                    pre_counts.get("subject", 0),
                )
            )

        # --------- NEW: bulk-load contexts for all candidates (batched) ---------
        cand_list_all = sorted(candidates)
        ctx_by_id = _bulk_load_contexts(conn, cand_list_all, batch_size=400)

        # where-eval filter (re-uses same contexts; no per-table DB work)
        if where_ast:
            filtered: Set[int] = set()
            for tid in cand_list_all:
                ctx = ctx_by_id.get(tid)
                if not ctx:
                    continue
                if eval_where(where_ast, table_ctx=ctx.__dict__):
                    filtered.add(tid)
            candidates = filtered
        
        # --- Semantic fallback:
        # If TITLE literals exist and the lexical WHERE wiped out all candidates,
        # re-evaluate the WHERE with TITLE contains removed (keep all other filters).
        title_literals = positives.get("TITLE", [])
        if not candidates and args.semantic and title_literals:
            where_no_title = _strip_title_contains(where_ast) if where_ast else None
            if where_no_title:
                filtered2: Set[int] = set()
                for tid in cand_list_all:
                    ctx = ctx_by_id.get(tid)
                    if not ctx:
                        continue
                    if eval_where(where_no_title, table_ctx=ctx.__dict__):
                        filtered2.add(tid)
                candidates = filtered2

        if not candidates:
            if args.semantic:
                print("semantic: no tables matched filters; skipping embeddings")
            return []

        # ---------- Title ranking (lexical + semantic) ----------
        lexical_ranks: Dict[int, int] = {}
        semantic_ranks: Dict[int, int] = {}

        title_literals = positives.get("TITLE", [])
        title_text = " ".join(dict.fromkeys(title_literals)) if title_literals else ""

        if title_text and get_settings().enable_titles_fts:
            q = _fts_query(title_text)
            if q:
                rows = conn.execute(
                    "SELECT table_id FROM table_titles_fts WHERE table_titles_fts MATCH ?",
                    (q,),
                ).fetchall()
                order: List[int] = []
                seen: Set[int] = set()
                for r in rows:
                    tid = int(r[0])
                    if tid in candidates and tid not in seen:
                        seen.add(tid)
                        order.append(tid)
                    if len(order) >= args.limit * 5:
                        break
                lexical_ranks = {tid: idx + 1 for idx, tid in enumerate(order)}

        semantic_enabled = False
        semantic_client = embedding_client if embedding_client else None
        if args.semantic:
            settings = get_settings()
            if not title_text.strip():
                print("[semantic] no title literals in query; skipping embeddings")
            elif not settings.enable_title_embeddings:
                print("semantic disabled: ENABLE_TITLE_EMBEDDINGS=0")
            else:
                model = semantic_client.model if semantic_client else settings.embedding_model
                row = conn.execute(
                    "SELECT COUNT(*) FROM embeddings WHERE entity_type='agregado' AND model=?",
                    (model,),
                ).fetchone()
                count = int(row[0]) if row else 0
                if count == 0:
                    print("semantic: no table embeddings found; run `embed-titles`")
                else:
                    semantic_enabled = True
                    if semantic_client is None:
                        semantic_client = EmbeddingClient(model=model)
        if title_text and semantic_enabled and semantic_client:
            try:
                qvec = await asyncio.to_thread(
                    lambda: semantic_client.embed_text(title_text, model=semantic_client.model)
                )
            except Exception as exc:
                print(f"semantic: embeddings request failed: {exc} (continuing without)")
                qvec = None
            if qvec:
                ordered = sorted(
                    candidates,
                    key=lambda tid: lexical_ranks.get(tid, float("inf")),
                )
                cap = max(200, args.limit * 10)
                ordered = ordered[:cap]
                if ordered:
                    placeholders = ",".join("?" for _ in ordered)
                    sql = (
                        "SELECT entity_id, dimension, vector "
                        "FROM embeddings WHERE entity_type='agregado' AND model=? AND entity_id IN ("
                        + placeholders
                        + ")"
                    )
                    cur = conn.execute(sql, (semantic_client.model, *ordered))
                    sims: List[Tuple[int, float]] = []
                    for row in cur.fetchall():
                        tid = int(row["entity_id"])
                        blob = row["vector"]
                        dim = int(row["dimension"])
                        arr = array("f")
                        arr.frombytes(blob)
                        vec = list(arr)
                        if dim and len(vec) > dim:
                            vec = vec[:dim]
                        if not vec:
                            continue
                        dot = sum(a * b for a, b in zip(qvec, vec))
                        norm_q = math.sqrt(sum(a * a for a in qvec))
                        norm_t = math.sqrt(sum(a * a for a in vec))
                        sim = 0.0 if norm_q == 0 or norm_t == 0 else dot / (norm_q * norm_t)
                        if sim > 0:
                            sims.append((tid, sim))
                    sims.sort(key=lambda x: x[1], reverse=True)
                    semantic_ranks = {tid: idx + 1 for idx, (tid, _s) in enumerate(sims[: args.limit * 5])}

        combined_ranks = dict(lexical_ranks)
        combined_ranks.update(semantic_ranks)
        rrf_scores = rrf(combined_ranks, k=60.0)

        # ---------- Assemble hits (no extra DB — reuse contexts) ----------
        hits: List[TableHit] = []
        for tid in sorted(candidates):
            ctx = ctx_by_id.get(tid)
            if not ctx:
                continue
            struct_info = _structural_for_table(
                ctx,
                var_scores=var_scores,
                var_exact=var_exact,
                class_groups=class_groups,
                class_exact=class_exact,
                cat_req=cat_req,
            )
            if struct_info is None:
                continue
            struct, why = struct_info

            rrf_score = float(rrf_scores.get(tid, 0.0))
            final = 0.75 * struct + 0.25 * rrf_score

            hits.append(
                TableHit(
                    table_id=tid,
                    title=ctx.title,
                    period_start=ctx.period_start,
                    period_end=ctx.period_end,
                    n3=ctx.n3,
                    n6=ctx.n6,
                    why=why,
                    score=final,
                    rrf_score=rrf_score,
                    struct_score=struct,
                )
            )

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[: max(1, int(args.limit))]
    finally:
        conn.close()
