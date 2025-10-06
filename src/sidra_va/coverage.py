# src/sidra_va/coverage.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Optional


# ------- Tokenizer -------

@dataclass(frozen=True)
class _Tok:
    kind: str  # 'ID','NUM','OP','LP','RP','AND','OR','NOT','EOF'
    value: str
    pos: int


def _tokens(s: str) -> Iterator[_Tok]:
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch.isspace():
            i += 1
            continue
        if ch in '()':
            yield _Tok('LP' if ch == '(' else 'RP', ch, i)
            i += 1
            continue
        # multi-char operators
        if i + 1 < n:
            two = s[i:i+2]
            if two in ('>=', '<=', '==', '!=', '&&'):
                yield _Tok('OP' if two != '&&' else 'AND', two, i)
                i += 2
                continue
        # single-char operators
        if ch in ('<', '>', '='):
            yield _Tok('OP', ch, i)
            i += 1
            continue
        # identifiers and keywords
        if ch.isalpha() or ch == '_':
            start = i
            i += 1
            while i < n and (s[i].isalnum() or s[i] in ('_',)):
                i += 1
            word = s[start:i].upper()
            if word in ('AND', 'OR', 'NOT'):
                yield _Tok(word, word, start)
            else:
                yield _Tok('ID', word, start)
            continue
        # numbers
        if ch.isdigit():
            start = i
            i += 1
            while i < n and s[i].isdigit():
                i += 1
            yield _Tok('NUM', s[start:i], start)
            continue
        # symbolic OR / NOT
        if ch == '|':
            # accept '||' as OR; single '|' treated as error
            if i + 1 < n and s[i+1] == '|':
                yield _Tok('OR', '||', i)
                i += 2
                continue
        if ch == '!':
            yield _Tok('NOT', '!', i)
            i += 1
            continue
        raise SyntaxError(f"Unexpected character {ch!r} at {i}")
    yield _Tok('EOF', '', n)


# ------- AST -------

@dataclass(frozen=True)
class _Cmp:
    op: str   # '>=','>','<=','<','==','!='
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


# ------- Parser -------

class _Parser:
    def __init__(self, text: str) -> None:
        self._iter = iter(_tokens(text))
        self.cur: _Tok = next(self._iter)

    def _eat(self, kind: str) -> _Tok:
        if self.cur.kind != kind:
            raise SyntaxError(f"Expected {kind}, got {self.cur.kind} at {self.cur.pos}")
        tok = self.cur
        self.cur = next(self._iter)
        return tok

    def parse(self) -> Any:
        node = self._expr()
        if self.cur.kind != 'EOF':
            raise SyntaxError(f"Unexpected token {self.cur.kind} at {self.cur.pos}")
        return node

    def _expr(self) -> Any:
        # or := and (OR and)*
        node = self._and()
        while self.cur.kind == 'OR':
            self._eat('OR')
            node = _Or(node, self._and())
        return node

    def _and(self) -> Any:
        # and := unary (AND unary)*
        node = self._unary()
        while self.cur.kind == 'AND':
            self._eat('AND')
            node = _And(node, self._unary())
        return node

    def _unary(self) -> Any:
        # unary := NOT unary | primary
        if self.cur.kind == 'NOT':
            self._eat('NOT')
            return _Not(self._unary())
        return self._primary()

    def _primary(self) -> Any:
        # primary := '(' expr ')' | cmp
        if self.cur.kind == 'LP':
            self._eat('LP')
            node = self._expr()
            self._eat('RP')
            return node
        return self._cmp()

    def _cmp(self) -> Any:
        # cmp := ID OP NUM
        ident = self._eat('ID').value
        op = self._eat('OP').value
        if op == '=':
            op = '=='
        num_tok = self._eat('NUM')
        number = int(num_tok.value)
        return _Cmp(op, ident, number)


def parse_coverage_expr(text: str) -> Any:
    """
    Parse a coverage expression like:
        (N3>=27) AND (N6>=5000 OR N4>10)
    Supports: AND/OR/NOT or symbolic &&, ||, ! and parentheses.
    Comparisons: >, >=, <, <=, ==, != against integer numbers.
    Identifiers are uppercased (e.g., n3 -> N3).
    Returns an AST suitable for eval_coverage()/extract_levels().
    """
    return _Parser(text).parse()


def extract_levels(node: Any) -> set[str]:
    """Collect all identifiers used in the expression (e.g., {'N3','N6'})."""
    out: set[str] = set()
    def _walk(n: Any) -> None:
        if isinstance(n, _Cmp):
            out.add(n.ident)
        elif isinstance(n, (_Not,)):
            _walk(n.node)
        elif isinstance(n, (_And, _Or)):
            _walk(n.left); _walk(n.right)
    _walk(node)
    return out


def eval_coverage(node: Any, counts: dict[str, int]) -> bool:
    """Evaluate the expression against a {level_code -> count} mapping."""
    def _cmp(op: str, a: int, b: int) -> bool:
        if op == '>=': return a >= b
        if op == '>':  return a > b
        if op == '<=': return a <= b
        if op == '<':  return a < b
        if op == '==': return a == b
        if op == '!=': return a != b
        raise ValueError(f"Unsupported op {op}")
    def _ev(n: Any) -> bool:
        if isinstance(n, _Cmp):
            val = int(counts.get(n.ident, 0))
            return _cmp(n.op, val, n.number)
        if isinstance(n, _Not):
            return not _ev(n.node)
        if isinstance(n, _And):
            return _ev(n.left) and _ev(n.right)
        if isinstance(n, _Or):
            return _ev(n.left) or _ev(n.right)
        raise TypeError(f"Bad node {type(n)}")
    return _ev(node)
