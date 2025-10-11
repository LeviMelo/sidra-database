from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Tuple


@dataclass(frozen=True)
class _Tok:
    kind: str
    value: str
    pos: int


def _tokens(text: str) -> Iterator[_Tok]:
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if ch in "()":
            yield _Tok("LP" if ch == "(" else "RP", ch, i)
            i += 1
            continue
        if ch == "[":
            yield _Tok("LBRACK", ch, i)
            i += 1
            continue
        if ch == "]":
            yield _Tok("RBRACK", ch, i)
            i += 1
            continue
        if ch == "~":
            yield _Tok("TILDE", ch, i)
            i += 1
            continue
        if ch == "." and i + 1 < n and text[i + 1] == ".":
            yield _Tok("RANGE", "..", i)
            i += 2
            continue
        if ch == '"':
            start = i
            i += 1
            buf: list[str] = []
            while i < n:
                cur = text[i]
                if cur == "\\":
                    i += 1
                    if i >= n:
                        raise SyntaxError(f"Unterminated string starting at {start}")
                    esc = text[i]
                    if esc not in {'"', "\\"}:
                        buf.append(esc)
                    else:
                        buf.append(esc)
                    i += 1
                    continue
                if cur == '"':
                    i += 1
                    break
                buf.append(cur)
                i += 1
            else:
                raise SyntaxError(f"Unterminated string starting at {start}")
            yield _Tok("STR", "".join(buf), start)
            continue
        if i + 1 < n:
            two = text[i : i + 2]
            if two in (">=", "<=", "==", "!=", "&&"):
                yield _Tok("OP" if two != "&&" else "AND", two, i)
                i += 2
                continue
            if two == "||":
                yield _Tok("OR", two, i)
                i += 2
                continue
        if ch in ("<", ">", "="):
            yield _Tok("OP", ch, i)
            i += 1
            continue
        if ch.isalpha() or ch == "_":
            start = i
            i += 1
            while i < n and (text[i].isalnum() or text[i] == "_"):
                i += 1
            word = text[start:i].upper()
            if word in {"AND", "OR", "NOT", "IN"}:
                yield _Tok(word, word, start)
            else:
                yield _Tok("ID", word, start)
            continue
        if ch.isdigit():
            start = i
            i += 1
            while i < n and text[i].isdigit():
                i += 1
            yield _Tok("NUM", text[start:i], start)
            continue
        if ch == "!":
            yield _Tok("NOT", "!", i)
            i += 1
            continue
        raise SyntaxError(f"Unexpected {ch!r} at {i}")
    yield _Tok("EOF", "", n)


@dataclass(frozen=True)
class _Cmp:
    op: str
    ident: str
    number: int


@dataclass(frozen=True)
class _Not:
    node: Any


@dataclass(frozen=True)
class _And:
    left: Any
    right: Any


@dataclass(frozen=True)
class _Or:
    left: Any
    right: Any


@dataclass(frozen=True)
class _StrLit:
    text: str


@dataclass(frozen=True)
class _Contains:
    field: str
    needle: Any


@dataclass(frozen=True)
class _PeriodCmp:
    op: str
    year: int


@dataclass(frozen=True)
class _PeriodRange:
    start: int
    end: int


WhereNode = Any


class _Parser:
    def __init__(self, text: str) -> None:
        self._it = iter(_tokens(text))
        self.cur = next(self._it)

    def _eat(self, kind: str) -> _Tok:
        if self.cur.kind != kind:
            raise SyntaxError(f"Expected {kind}, got {self.cur.kind} at {self.cur.pos}")
        tok = self.cur
        self.cur = next(self._it)
        return tok

    def parse(self) -> WhereNode:
        node = self._expr()
        if self.cur.kind != "EOF":
            raise SyntaxError(f"Unexpected {self.cur.kind} at {self.cur.pos}")
        return node

    def _expr(self) -> WhereNode:
        node = self._and()
        while self.cur.kind == "OR":
            self._eat("OR")
            node = _Or(node, self._and())
        return node

    def _and(self) -> WhereNode:
        node = self._unary()
        while self.cur.kind == "AND":
            self._eat("AND")
            node = _And(node, self._unary())
        return node

    def _unary(self) -> WhereNode:
        if self.cur.kind == "NOT":
            self._eat("NOT")
            return _Not(self._unary())
        return self._primary()

    def _primary(self) -> WhereNode:
        if self.cur.kind == "LP":
            self._eat("LP")
            node = self._expr()
            self._eat("RP")
            return node
        if self.cur.kind == "ID":
            return self._field_expr()
        raise SyntaxError(f"Unexpected {self.cur.kind} at {self.cur.pos}")

    def _field_expr(self) -> WhereNode:
        ident = self._eat("ID").value
        if ident == "PERIOD":
            return self._period_expr()
        if self.cur.kind == "TILDE":
            self._eat("TILDE")
            return _Contains(ident, self._contains_rhs())
        if self.cur.kind != "OP":
            return _Cmp(">=", ident, 1)
        op = self._eat("OP").value
        if op == "=":
            op = "=="
        number = int(self._eat("NUM").value)
        return _Cmp(op, ident, number)

    def _period_expr(self) -> WhereNode:
        # Support: PERIOD IN [YYYY..YYYY]
        if self.cur.kind == "IN":
            self._eat("IN")
            self._eat("LBRACK")
            start = int(self._eat("NUM").value)
            self._eat("RANGE")
            end = int(self._eat("NUM").value)
            self._eat("RBRACK")
            if end < start:
                raise SyntaxError("period range end < start")
            return _PeriodRange(start, end)

        # Support: PERIOD {>, >=, ==, !=, <, <=} YYYY
        if self.cur.kind != "OP":
            raise SyntaxError("Expected comparator or IN after PERIOD")
        op = self._eat("OP").value
        if op == "=":
            op = "=="
        year = int(self._eat("NUM").value)
        return _PeriodCmp(op, year)

    def _contains_rhs(self) -> WhereNode:
        if self.cur.kind == "STR":
            return _StrLit(self._eat("STR").value)
        if self.cur.kind == "LP":
            self._eat("LP")
            node = self._contains_expr()
            self._eat("RP")
            return node
        raise SyntaxError("Expected string or group after ~")

    def _contains_expr(self) -> WhereNode:
        node = self._contains_and()
        while self.cur.kind == "OR":
            self._eat("OR")
            node = _Or(node, self._contains_and())
        return node

    def _contains_and(self) -> WhereNode:
        node = self._contains_unary()
        while self.cur.kind == "AND":
            self._eat("AND")
            node = _And(node, self._contains_unary())
        return node

    def _contains_unary(self) -> WhereNode:
        if self.cur.kind == "NOT":
            self._eat("NOT")
            return _Not(self._contains_unary())
        return self._contains_primary()

    def _contains_primary(self) -> WhereNode:
        if self.cur.kind == "LP":
            self._eat("LP")
            node = self._contains_expr()
            self._eat("RP")
            return node
        if self.cur.kind == "STR":
            return _StrLit(self._eat("STR").value)
        raise SyntaxError(f"Expected string literal in contains expression at {self.cur.pos}")


def parse_where_expr(text: str) -> WhereNode:
    text = text.strip()
    if not text:
        raise SyntaxError("empty query")
    return _Parser(text).parse()


def extract_fields(node: WhereNode) -> set[str]:
    fields: set[str] = set()

    def _walk(n: WhereNode) -> None:
        if isinstance(n, _Contains):
            fields.add(n.field)
            _walk(n.needle)
        elif isinstance(n, _Cmp):
            fields.add(n.ident)
        elif isinstance(n, (_PeriodCmp, _PeriodRange)):
            fields.add("PERIOD")
        elif isinstance(n, _StrLit):
            return
        elif isinstance(n, _Not):
            _walk(n.node)
        elif isinstance(n, (_And, _Or)):
            _walk(n.left)
            _walk(n.right)

    _walk(node)
    return fields


def iter_contains_literals(node: WhereNode, *, positive: bool = True) -> Iterator[Tuple[str, bool, str]]:
    def _walk(n: WhereNode, polarity: bool) -> Iterator[Tuple[str, bool, str]]:
        if isinstance(n, _Not):
            yield from _walk(n.node, not polarity)
        elif isinstance(n, _And) or isinstance(n, _Or):
            yield from _walk(n.left, polarity)
            yield from _walk(n.right, polarity)
        elif isinstance(n, _Contains):
            yield from _walk_contains(n.field, n.needle, polarity)
        elif isinstance(n, (_Cmp, _PeriodCmp, _PeriodRange, _StrLit)):
            return
        else:
            return

    def _walk_contains(field: str, needle: WhereNode, polarity: bool) -> Iterator[Tuple[str, bool, str]]:
        if isinstance(needle, _StrLit):
            yield field, polarity, needle.text
            return
        if isinstance(needle, _Not):
            yield from _walk_contains(field, needle.node, not polarity)
            return
        if isinstance(needle, (_And, _Or)):
            yield from _walk_contains(field, needle.left, polarity)
            yield from _walk_contains(field, needle.right, polarity)
            return

    yield from _walk(node, positive)


__all__ = [
    "WhereNode",
    "_Cmp",
    "_Not",
    "_And",
    "_Or",
    "_StrLit",
    "_Contains",
    "_PeriodCmp",
    "_PeriodRange",
    "parse_where_expr",
    "extract_fields",
    "iter_contains_literals",
]
