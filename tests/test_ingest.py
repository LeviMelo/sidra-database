from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from sidra_database import ingest_agregado, ensure_schema, get_database_path
from sidra_database.db import create_connection


class FakeSidraClient:
    def __init__(self) -> None:
        self._localities = {
            ("Administrativo", "N1"): [
                {"id": "1", "nome": "Brasil", "nivel": {"nome": "Brasil"}}
            ],
            ("Administrativo", "N3"): [
                {"id": "11", "nome": "Rondônia", "nivel": {"nome": "Unidade da Federação"}},
                {"id": "12", "nome": "Acre", "nivel": {"nome": "Unidade da Federação"}},
            ],
        }

    async def fetch_metadata(self, agregado_id: int):
        assert agregado_id == 1234
        return {
            "id": agregado_id,
            "nome": "Tabela teste",
            "pesquisa": "Pesquisa demo",
            "assunto": "Demografia",
            "URL": "https://sidra.ibge.gov.br/tabela/1234",
            "periodicidade": {"frequencia": "anual", "inicio": 2000, "fim": 2020},
            "nivelTerritorial": {
                "Administrativo": ["N1", "N3"],
                "Especial": [],
                "IBGE": [],
            },
            "variaveis": [
                {"id": 99, "nome": "População", "unidade": "Pessoas", "sumarizacao": ["nivelTerritorial"]}
            ],
            "classificacoes": [
                {
                    "id": 1,
                    "nome": "Sexo",
                    "sumarizacao": {"status": True, "excecao": []},
                    "categorias": [
                        {"id": 0, "nome": "Total", "unidade": None, "nivel": 0},
                        {"id": 4, "nome": "Homens", "unidade": None, "nivel": 1},
                    ],
                }
            ],
        }

    async def fetch_periods(self, agregado_id: int):
        assert agregado_id == 1234
        return [
            {"id": "2000", "literals": ["2000"], "modificacao": "01/01/2021"},
            {"id": "2020", "literals": ["2020"], "modificacao": "01/01/2021"},
        ]

    async def fetch_localities(self, agregado_id: int, level: str):
        assert agregado_id == 1234
        for (level_type, level_code), payload in self._localities.items():
            if level_code == level:
                return payload
        return []


@pytest.fixture(autouse=True)
def override_db_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_path = tmp_path / "sidra-test.db"
    monkeypatch.setenv("SIDRA_DATABASE_PATH", str(db_path))
    yield
    if db_path.exists():
        db_path.unlink()


@pytest.mark.asyncio
async def test_ingest_agregado_writes_metadata():
    ensure_schema()
    client = FakeSidraClient()
    await ingest_agregado(1234, client=client)

    conn = create_connection()
    try:
        row = conn.execute("SELECT nome, pesquisa, assunto FROM agregados WHERE id = 1234").fetchone()
        assert row["nome"] == "Tabela teste"
        assert row["pesquisa"] == "Pesquisa demo"

        variables = conn.execute("SELECT COUNT(*) FROM variables WHERE agregado_id = 1234").fetchone()[0]
        assert variables == 1

        categories = conn.execute("SELECT COUNT(*) FROM categories WHERE agregado_id = 1234").fetchone()[0]
        assert categories == 2

        periods = conn.execute("SELECT periodo_id FROM periods WHERE agregado_id = 1234 ORDER BY periodo_id").fetchall()
        assert [p["periodo_id"] for p in periods] == ["2000", "2020"]

        localities = conn.execute(
            "SELECT COUNT(*) FROM localities WHERE agregado_id = 1234 AND level_id = ?", ("N3",)
        ).fetchone()[0]
        assert localities == 2
    finally:
        conn.close()


def test_get_database_path_uses_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    target = tmp_path / "alt.db"
    monkeypatch.setenv("SIDRA_DATABASE_PATH", str(target))
    ensure_schema()
    path = get_database_path()
    assert path == target.resolve()
