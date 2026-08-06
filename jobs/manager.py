from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import uuid4

from indexing.service import IndexingService


class JobConflictError(RuntimeError):
    pass


@dataclass(slots=True)
class JobRecord:
    id: str
    dataset_id: str
    engine_ids: list[str]
    status: str = "queued"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    finished_at: datetime | None = None
    processed_chunks: int = 0
    result: dict[str, int] | None = None
    error: str | None = None


class JobManager:
    def __init__(self, indexing: IndexingService) -> None:
        self.indexing = indexing
        self._jobs: dict[str, JobRecord] = {}

    def start(self, dataset_id: str, engine_ids: list[str]) -> JobRecord:
        requested_engines = set(engine_ids)
        conflict = any(
            job.dataset_id == dataset_id
            and job.status in {"queued", "running"}
            and requested_engines.intersection(job.engine_ids)
            for job in self._jobs.values()
        )
        if conflict:
            raise JobConflictError(
                f"An index build is already running for dataset '{dataset_id}' "
                "and one or more selected engines."
            )
        record = JobRecord(id=str(uuid4()), dataset_id=dataset_id, engine_ids=engine_ids)
        self._jobs[record.id] = record
        asyncio.create_task(self._run(record))
        return record

    def get(self, job_id: str) -> JobRecord | None:
        return self._jobs.get(job_id)

    async def _run(self, record: JobRecord) -> None:
        record.status = "running"
        record.started_at = datetime.now(UTC)
        try:
            record.result = await self.indexing.build(
                record.dataset_id,
                record.engine_ids,
                progress=lambda count: setattr(record, "processed_chunks", count),
            )
            record.status = "succeeded"
        except Exception as error:  # Jobs must preserve failures for the API rather than disappear.
            record.status = "failed"
            record.error = str(error)
        finally:
            record.finished_at = datetime.now(UTC)
