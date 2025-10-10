#!/usr/bin/env bash
set -euo pipefail

echo "== sanity: pick DB in repo root =="
export SIDRA_DATABASE_PATH="$(pwd)/sidra.db"
python -m sidra_search.cli db stats

echo
echo "== 0) CLI manual quick view =="
python -m sidra_search.cli --manual | head -n 20

echo
echo "== A) unified boolean search examples =="
python -m sidra_search.cli search --q '(class~"sexo" OR var~"pessoas") AND (N6>=5000 OR N3==27)' --limit 10 --debug-fuzzy || true

echo
python -m sidra_search.cli search --q 'class~("sexo" AND NOT "agro")' --limit 10 --explain || true

echo
python -m sidra_search.cli search --q 'survey~"censo"' --limit 10 || true

echo
python -m sidra_search.cli search --q 'subject~"agro"' --limit 10 || true

echo
python -m sidra_search.cli search --q 'period in [2010..2019] AND subject~"educação"' --limit 10 || true

echo
python -m sidra_search.cli search --q 'var~"pessoas" AND class~"sexo"' --limit 10 --explain || true

echo
python -m sidra_search.cli search --q 'var~"pessoas" AND class~"região"' --limit 10 --explain || true

echo
python -m sidra_search.cli search --q 'cat~"Cor ou raça::Branca"' --limit 10 || true

echo
python -m sidra_search.cli search --q 'cat~"Branca"' --limit 10 || true

echo
echo "== B) semantic title ranking diagnostics =="
SIDRA_SEARCH_ENABLE_TITLE_EMBEDDINGS=0 python -m sidra_search.cli search --q 'title~"teste"' --semantic --limit 5 || true
python -m sidra_search.cli search --q 'survey~"censo"' --semantic --limit 5 || true

echo
echo "== C) back-compat translation still works =="
python -m sidra_search.cli search --title "pessoas indígenas" --coverage "(N6>=5000) OR N3" --limit 5 || true
python -m sidra_search.cli search --q 'title~"pessoas indígenas" AND ((N6>=5000) OR N3)' --limit 5 || true

echo
echo "== D) smoke ingest/build remains available =="
python -m sidra_search.cli ingest-coverage --coverage "N3 OR (N6>=5000)" --limit 3 || true
python -m sidra_search.cli build-links --all || true

echo
echo "== E) manual reminder =="
python -m sidra_search.cli --manual | head -n 20

echo
echo "== F) parser errors are surfaced =="
python -m sidra_search.cli search --q 'N6>>=5' 2>&1 | head -n 3 || true

echo
echo "== Smoke complete (inspect outputs for sanity) =="
