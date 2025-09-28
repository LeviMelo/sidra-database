from __future__ import annotations

import asyncio
from array import array

from sidra_database.db import create_connection, ensure_schema
from sidra_database.embedding import EmbeddingClient

from .schema_migrations import apply_va_schema
from .utils import sha256_text, utcnow_iso


def _vector_to_blob(vector) -> bytes:
    arr = array("f", (float(value) for value in vector))
    return arr.tobytes()


async def embed_vas_for_agregados(
    agregado_ids: list[int] | None,
    *,
    concurrency: int = 6,
    model: str | None = None,
    embedding_client: EmbeddingClient | None = None,
) -> dict[str, int]:
    client = embedding_client or EmbeddingClient(model=model)
    model_name = model or client.model

    def _collect_targets():
        conn = create_connection()
        try:
            ensure_schema(conn)
            apply_va_schema(conn)
            if agregado_ids is None:
                cursor = conn.execute("SELECT va_id, agregado_id, text FROM value_atoms")
            else:
                placeholder = ",".join("?" for _ in agregado_ids)
                cursor = conn.execute(
                    f"SELECT va_id, agregado_id, text FROM value_atoms WHERE agregado_id IN ({placeholder})",
                    tuple(agregado_ids),
                )
            rows = cursor.fetchall()
            return [(row[0], row[1], row[2]) for row in rows]
        finally:
            conn.close()

    targets = await asyncio.to_thread(_collect_targets)
    if not targets:
        return {"embedded": 0, "skipped": 0, "failed": 0}

    semaphore = asyncio.Semaphore(concurrency)
    stats = {"embedded": 0, "skipped": 0, "failed": 0}

    async def _process(target):
        va_id, agregado_id, text = target
        text_hash = sha256_text(text)

        def _should_embed() -> bool:
            conn = create_connection()
            try:
                ensure_schema(conn)
                apply_va_schema(conn)
                row = conn.execute(
                    "SELECT text_hash FROM embeddings WHERE entity_type = 'va' AND entity_id = ? AND model = ?",
                    (va_id, model_name),
                ).fetchone()
                if row and row[0] == text_hash:
                    return False
                return True
            finally:
                conn.close()

        should_embed = await asyncio.to_thread(_should_embed)
        if not should_embed:
            stats["skipped"] += 1
            return

        async with semaphore:
            try:
                vector = await asyncio.to_thread(client.embed_text, text, model=model_name)
            except Exception:
                stats["failed"] += 1
                return

            def _persist() -> None:
                conn = create_connection()
                try:
                    ensure_schema(conn)
                    apply_va_schema(conn)
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO embeddings (
                            entity_type, entity_id, agregado_id, text_hash, model, dimension, vector, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            "va",
                            va_id,
                            agregado_id,
                            text_hash,
                            model_name,
                            len(vector),
                            _vector_to_blob(vector),
                            utcnow_iso(),
                        ),
                    )
                    conn.commit()
                finally:
                    conn.close()

            await asyncio.to_thread(_persist)
            stats["embedded"] += 1

    await asyncio.gather(*[_process(target) for target in targets])
    return stats


__all__ = ["embed_vas_for_agregados"]
