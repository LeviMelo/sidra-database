from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterator

@dataclass(frozen=True)
class _Tok:
    kind: str
    value: str
    pos: int

def _tokens(s: str) -> Iterator[_Tok]:
    i, n = 0, len(s)
    while i < n:
        ch = s[i]
        if ch.isspace(): i += 1; continue
        if ch in '()':
            yield _Tok('LP' if ch=='(' else 'RP', ch, i); i += 1; continue
        if i+1<n:
            two = s[i:i+2]
            if two in ('>=','<=','==','!=','&&'): yield _Tok('OP' if two!='&&' else 'AND', two, i); i += 2; continue
        if ch in ('<','>','='): yield _Tok('OP', ch, i); i += 1; continue
        if ch.isalpha() or ch=='_':
            start=i; i+=1
            while i<n and (s[i].isalnum() or s[i]=='_'): i+=1
            word = s[start:i].upper()
            if word in ('AND','OR','NOT'): yield _Tok(word, word, start)
            else: yield _Tok('ID', word, start)
            continue
        if ch.isdigit():
            start=i; i+=1
            while i<n and s[i].isdigit(): i+=1
            yield _Tok('NUM', s[start:i], start); continue
        if ch=='|':
            if i+1<n and s[i+1]=='|': yield _Tok('OR','||',i); i+=2; continue
        if ch=='!': yield _Tok('NOT','!',i); i+=1; continue
        raise SyntaxError(f"Unexpected {ch!r} at {i}")
    yield _Tok('EOF','',n)

@dataclass(frozen=True)
class _Cmp: op: str; ident: str; number: int
@dataclass(frozen=True)
class _Not: node: Any
@dataclass(frozen=True)
class _And: left: Any; right: Any
@dataclass(frozen=True)
class _Or: left: Any; right: Any

class _Parser:
    def __init__(self, text: str) -> None:
        self._it = iter(_tokens(text)); self.cur = next(self._it)
    def _eat(self, kind: str) -> _Tok:
        if self.cur.kind != kind: raise SyntaxError(f"Expected {kind}, got {self.cur.kind} at {self.cur.pos}")
        t = self.cur; self.cur = next(self._it); return t
    def parse(self) -> Any:
        n = self._expr()
        if self.cur.kind != 'EOF': raise SyntaxError(f"Unexpected {self.cur.kind} at {self.cur.pos}")
        return n
    def _expr(self) -> Any:
        n = self._and()
        while self.cur.kind == 'OR':
            self._eat('OR'); n = _Or(n, self._and())
        return n
    def _and(self) -> Any:
        n = self._unary()
        while self.cur.kind == 'AND':
            self._eat('AND'); n = _And(n, self._unary())
        return n
    def _unary(self) -> Any:
        if self.cur.kind=='NOT': self._eat('NOT'); return _Not(self._unary())
        return self._primary()
    def _primary(self) -> Any:
        if self.cur.kind=='LP':
            self._eat('LP'); n=self._expr(); self._eat('RP'); return n
        return self._cmp()
    def _cmp(self) -> Any:
        ident = self._eat('ID').value
        op = self._eat('OP').value
        if op == '=': op = '=='
        number = int(self._eat('NUM').value)
        return _Cmp(op, ident, number)

def parse_coverage_expr(text: str) -> Any: return _Parser(text).parse()

def extract_levels(node: Any) -> set[str]:
    out: set[str] = set()
    def _w(n: Any) -> None:
        if isinstance(n, _Cmp): out.add(n.ident)
        elif isinstance(n, _Not): _w(n.node)
        elif isinstance(n, (_And,_Or)): _w(n.left); _w(n.right)
    _w(node)
    return out

def eval_coverage(node: Any, counts: dict[str,int]) -> bool:
    def _cmp(op: str, a: int, b: int) -> bool:
        return {'>=':a>=b,'>':a>b,'<=':a<=b,'<':a<b,'==':a==b,'!=':a!=b}[op]
    def _ev(n: Any) -> bool:
        if isinstance(n,_Cmp): return _cmp(n.op, int(counts.get(n.ident,0)), n.number)
        if isinstance(n,_Not): return not _ev(n.node)
        if isinstance(n,_And): return _ev(n.left) and _ev(n.right)
        if isinstance(n,_Or): return _ev(n.left) or _ev(n.right)
        raise TypeError(f"Bad node {n}")
    return _ev(node)
