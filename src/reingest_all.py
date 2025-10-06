from sidra_database.db import create_connection
from sidra_database.ingest import ingest_agregado
import asyncio

async def main():
    conn = create_connection()
    ids = [int(r[0]) for r in conn.execute("SELECT id FROM agregados ORDER BY id").fetchall()]
    conn.close()
    print(f"{len(ids)} agregados found")

    sem = asyncio.Semaphore(6)  # concurrency
    async def worker(tid):
        async with sem:
            try:
                await ingest_agregado(tid, generate_embeddings=False)
            except Exception as e:
                print(f"Failed {tid}: {e}")

    await asyncio.gather(*(worker(tid) for tid in ids))

if __name__ == "__main__":
    asyncio.run(main())
