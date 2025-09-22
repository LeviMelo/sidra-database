# SQLite Schema Draft

## Tables

### agregados
- `id` INTEGER PRIMARY KEY
- `nome` TEXT NOT NULL
- `pesquisa` TEXT
- `assunto` TEXT
- `url` TEXT
- `freq` TEXT  -- periodicidade.frequencia
- `periodo_inicio` TEXT
- `periodo_fim` TEXT
- `raw_json` BLOB NOT NULL  -- cached metadata payload
- `fetched_at` TEXT NOT NULL  -- ISO timestamp
- `municipality_locality_count` INTEGER DEFAULT 0  -- number of municipalities (N6) captured
- `covers_national_municipalities` INTEGER DEFAULT 0  -- 1 when municipality coverage exceeds the configured threshold

### agregados_levels
- `agregado_id` INTEGER REFERENCES agregados(id)
- `level_id` TEXT NOT NULL  -- e.g. N1, N2
- `level_name` TEXT
- `level_type` TEXT NOT NULL  -- Administrativo/Especial/IBGE
- `locality_count` INTEGER DEFAULT 0  -- number of localities returned for this level
- PRIMARY KEY (`agregado_id`, `level_id`, `level_type`)

### variables
- `id` INTEGER PRIMARY KEY
- `agregado_id` INTEGER NOT NULL REFERENCES agregados(id)
- `nome` TEXT NOT NULL
- `unidade` TEXT
- `sumarizacao` TEXT  -- JSON list
- `text_hash` TEXT NOT NULL  -- for embedding invalidation

### classifications
- `id` INTEGER NOT NULL
- `agregado_id` INTEGER NOT NULL REFERENCES agregados(id)
- `nome` TEXT NOT NULL
- `sumarizacao_status` INTEGER
- `sumarizacao_excecao` TEXT  -- JSON list
- PRIMARY KEY (`agregado_id`, `id`)

### categories
- `agregado_id` INTEGER NOT NULL REFERENCES agregados(id)
- `classification_id` INTEGER NOT NULL
- `categoria_id` INTEGER NOT NULL
- `nome` TEXT NOT NULL
- `unidade` TEXT
- `nivel` INTEGER
- `text_hash` TEXT NOT NULL
- PRIMARY KEY (`agregado_id`, `classification_id`, `categoria_id`)
- FOREIGN KEY (`agregado_id`, `classification_id`) REFERENCES classifications(`agregado_id`, `id`)

### periods
- `agregado_id` INTEGER NOT NULL REFERENCES agregados(id)
- `periodo_id` TEXT NOT NULL
- `literals` TEXT NOT NULL  -- JSON array from API
- `modificacao` TEXT
- PRIMARY KEY (`agregado_id`, `periodo_id`)

### localities
- `agregado_id` INTEGER NOT NULL REFERENCES agregados(id)
- `level_id` TEXT NOT NULL
- `locality_id` TEXT NOT NULL
- `nome` TEXT NOT NULL
- PRIMARY KEY (`agregado_id`, `level_id`, `locality_id`)
- FOREIGN KEY (`agregado_id`, `level_id`) REFERENCES agregados_levels(`agregado_id`, `level_id`)

### embeddings
- `entity_type` TEXT NOT NULL  -- table/variable/classification/category
- `entity_id` TEXT NOT NULL  -- composed key per type
- `agregado_id` INTEGER
- `text_hash` TEXT NOT NULL
- `model` TEXT NOT NULL
- `dimension` INTEGER NOT NULL
- `vector` BLOB NOT NULL
- `created_at` TEXT NOT NULL
- PRIMARY KEY (`entity_type`, `entity_id`, `model`)

### ingestion_log
- `id` INTEGER PRIMARY KEY AUTOINCREMENT
- `agregado_id` INTEGER NOT NULL
- `stage` TEXT NOT NULL  -- metadata/localities/periods/values
- `status` TEXT NOT NULL  -- success/failure
- `detail` TEXT
- `run_at` TEXT NOT NULL

## Index Suggestions
- `variables` index on (`agregado_id`)
- `categories` index on (`agregado_id`, `classification_id`)
- `localities` index on (`agregado_id`, `level_id`)
- `embeddings` index on (`agregado_id`, `model`)

## Notes
- Store text hashes as SHA256 hex strings to trigger embedding refresh.
- Keep raw metadata JSON in `agregados.raw_json` for auditing and replays.
- Locality catalogs can be large; consider chunked insert with transactions.
- Municipality coverage flags are derived from the locality counts (level `N6`) using a configurable threshold during ingestion.
- Future extension: add `data_availability` summary table if we persist observation counts.
