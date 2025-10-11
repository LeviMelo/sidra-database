from __future__ import annotations

from typing import Any, Sequence

from .normalize import normalize_basic
from .where_expr import (
    WhereNode,
    _And,
    _Cmp,
    _Contains,
    _Not,
    _Or,
    _PeriodCmp,
    _PeriodRange,
    _StrLit,
)


def _cmp(op: str, left: int, right: int) -> bool:
    return {
        ">=": left >= right,
        ">": left > right,
        "<=": left <= right,
        "<": left < right,
        "==": left == right,
        "!=": left != right,
    }[op]


def _years_overlap(years: Sequence[int], start: int, end: int) -> bool:
    if not years:
        return False
    for y in years:
        if start <= y <= end:
            return True
    return False


def _field_values(table_ctx: dict[str, Any], field: str) -> list[str]:
    field = field.upper()
    if field == "TITLE":
        return [table_ctx.get("title_norm", "")]
    if field == "SURVEY":
        return [table_ctx.get("survey_norm", "")]
    if field == "SUBJECT":
        return [table_ctx.get("subject_norm", "")]
    if field == "VAR":
        return table_ctx.get("vars", [])
    if field == "CLASS":
        return table_ctx.get("classes", [])
    if field == "CAT":
        return table_ctx.get("cats", [])
    return []


def _eval_contains_expr(node: WhereNode, haystack: str) -> bool:
    if isinstance(node, _StrLit):
        needle = normalize_basic(node.text)
        if not needle:
            return True
        return needle in haystack
    if isinstance(node, _Not):
        return not _eval_contains_expr(node.node, haystack)
    if isinstance(node, _And):
        return _eval_contains_expr(node.left, haystack) and _eval_contains_expr(node.right, haystack)
    if isinstance(node, _Or):
        return _eval_contains_expr(node.left, haystack) or _eval_contains_expr(node.right, haystack)
    raise TypeError(f"Unexpected node inside contains: {node!r}")


def _eval_contains(node: _Contains, table_ctx: dict[str, Any]) -> bool:
    values = _field_values(table_ctx, node.field)
    if not values:
        return False

    # Simple literal case
    if isinstance(node.needle, _StrLit):
        raw = node.needle.text
        if not raw:
            return True  # empty literal is trivially true

        # Special handling for categories: allow strict "Class::Cat" and loose "Cat"
        if node.field.upper() == "CAT":
            if "::" in raw:
                # strict pair
                class_part, cat_part = raw.split("::", 1)
                ck = normalize_basic(class_part)
                cat = normalize_basic(cat_part)
                if not ck or not cat:
                    return False
                target = f"{ck}::{cat}"
                # accept exact strict pair or any value that clearly contains both class and cat tokens
                return any(v == target or (ck in v and cat in v) for v in values)
            else:
                # loose category name — try both normal and "flattened" representations (treat '::' like space)
                needle = normalize_basic(raw)
                if not needle:
                    return True
                flattened = [v.replace("::", " ") for v in values]
                return any(needle in v for v in values) or any(needle in v for v in flattened)

        # Default contains (accent/punct-insensitive substring)
        needle = normalize_basic(raw)
        if not needle:
            return True
        return any(needle in v for v in values)

    # Composite contains expression: evaluate tree with substring semantics
    for value in values:
        if _eval_contains_expr(node.needle, value):
            return True
    return False


def eval_where(node: WhereNode, *, table_ctx: dict[str, Any]) -> bool:
    def _eval(n: WhereNode) -> bool:
        if isinstance(n, _Not):
            return not _eval(n.node)
        if isinstance(n, _And):
            return _eval(n.left) and _eval(n.right)
        if isinstance(n, _Or):
            return _eval(n.left) or _eval(n.right)
        if isinstance(n, _Cmp):
            counts = table_ctx.get("coverage_counts", {})
            value = int(counts.get(n.ident, 0))
            return _cmp(n.op, value, n.number)
        if isinstance(n, _Contains):
            return _eval_contains(n, table_ctx)
        if isinstance(n, _PeriodCmp):
            years = sorted(table_ctx.get("period_years", set()))
            if not years:
                return False
            if n.op == "==":
                return n.year in years
            if n.op == ">=":
                return years[-1] >= n.year
            if n.op == ">":
                return years[-1] > n.year
            if n.op == "<=":
                return years[0] <= n.year
            if n.op == "<":
                return years[0] < n.year
            raise ValueError(f"Unknown period comparator {n.op}")
        if isinstance(n, _PeriodRange):
            years = sorted(table_ctx.get("period_years", set()))
            return _years_overlap(years, n.start, n.end)
        if isinstance(n, _StrLit):
            # Should not reach top-level string literals
            return bool(normalize_basic(n.text))
        raise TypeError(f"Unsupported node: {n!r}")

    return _eval(node)


__all__ = ["eval_where"]
