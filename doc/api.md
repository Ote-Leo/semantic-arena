# API Reference

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health` | Report backend, model, Solr, and Meilisearch availability. |
| `GET /api/engines` | List available engines and their status. |
| `GET /api/datasets` | List dataset manifests and per-engine index status. |
| `POST /api/indexes` | Start indexing a selected dataset into selected engines. |
| `GET /api/jobs/{job_id}` | Return indexing progress, outcome, or error. |
| `POST /api/chat` | Run a query against selected engines concurrently. |
| `GET /api/citations/{dataset_id}/{chunk_id}` | Return source metadata for a citation. |

The current MVP rebuilds the deterministic dataset index for each selected engine. Build failures
are preserved in the job record and reported to the frontend. Versioned, atomic index replacement
is a planned production-hardening improvement.
