#!/usr/bin/env bash
set -euo pipefail

echo "== sanity: pick DB in repo root =="
export SIDRA_DATABASE_PATH="$(pwd)/sidra.db"
python -m sidra_search.cli db stats

echo
echo "== A) subject/survey filters =="
python -m sidra_search.cli ingest-coverage \
  --coverage "N3 OR (N6>=5000)" \
  --survey-contains "Censo" \
  --subject-contains "Agro" \
  --limit 10

python -m sidra_search.cli ingest-coverage \
  --coverage "N3 OR (N6>=5000)" \
  --survey-contains "Censo" \
  --limit 10

echo
echo "== B) link counts after ingest (should be nonzero) =="
python -m sidra_search.cli build-links --all
python -m sidra_search.cli db stats
sqlite3 "$SIDRA_DATABASE_PATH" "SELECT COUNT(*) FROM link_var;"
sqlite3 "$SIDRA_DATABASE_PATH" "SELECT COUNT(*) FROM link_class;"
sqlite3 "$SIDRA_DATABASE_PATH" "SELECT COUNT(*) FROM link_var_class;"

echo
echo "== C) verify known var×class pairs exist at SQL level =="
sqlite3 "$SIDRA_DATABASE_PATH" "
SELECT DISTINCT table_id
FROM link_var_class
WHERE var_key LIKE '%pessoas%' AND class_key='sexo';"

sqlite3 "$SIDRA_DATABASE_PATH" "
SELECT DISTINCT table_id
FROM link_var_class
WHERE var_key LIKE '%pessoas%' AND var_key LIKE '%indigen%'
  AND class_key LIKE '%ensino%';"

echo
echo "== D) search: single-token typo should succeed (token-level merge) =="
python -m sidra_search.cli search --var "pessoal" --class "sexo" --limit 10 --explain --debug-fuzzy

echo
echo "== E) search: multi-token misspelling still ok (via blended scorer) =="
python -m sidra_search.cli search --var "pssoas" --class "sexo" --limit 10 --explain --debug-fuzzy

echo
echo "== F) search: classes-only should work (no crash) =="
python -m sidra_search.cli search --class "sexo" --limit 10 --explain

echo
echo "== G) search: Class:Category strict filter =="
# Try common category 'Branca' under 'Cor ou raça' (if present in your ingest)
python -m sidra_search.cli search --class "Cor ou raça:Branca" --limit 10 --explain || true

echo
echo "== H) title FTS (lexical) =="
python -m sidra_search.cli search --title "frequencia escolar indígenas" --limit 10 --explain

echo
echo "== I) coverage filter on search results (should only show N6 >= 5000) =="
python -m sidra_search.cli search \
  --var "pessoas" --class "sexo" \
  --coverage "(N6>=5000)" \
  --limit 10 --explain

echo
echo "== J) debug mismatch drill: show top fuzzy keys directly =="
# Ensure we actually see 'pessoas' among var fuzzy candidates for 'pessoal'
python -m sidra_search.cli search --var "pessoal" --limit 3 --debug-fuzzy

echo
echo "== K) negative test: impossible coverage expression -> handled error =="
set +e
python -m sidra_search.cli ingest-coverage --coverage "N6>>=5" --limit 1 2>&1 | sed -n '1,5p'
set -e

echo
echo "== L) FTS row per table guaranteed =="
sqlite3 "$SIDRA_DATABASE_PATH" "SELECT COUNT(*), COUNT(DISTINCT table_id) FROM table_titles_fts;"

echo
echo "== M) pairing sanity: pick a returned table and show its classes =="
# Replace 10058 with any ID from earlier results:
sqlite3 "$SIDRA_DATABASE_PATH" "SELECT DISTINCT class_key FROM link_class WHERE table_id=10058 ORDER BY 1 LIMIT 20;"

echo
echo "== All tests ran. Scan the outputs above for: =="
echo "  - No 'No results.' where a pair exists in SQL"
echo "  - Fuzzy lists for 'pessoal' include at least one 'pessoas ...' key"
echo "  - Class-only search prints results (no TypeError)"
echo "  - Coverage-constrained search shows N6 >= 5000 in lines"
