from __future__ import annotations

from typing import Protocol

from datasets.models import Chunk
from search.models import SearchHit


class SearchEngineError(RuntimeError):
    pass


class SearchEngine(Protocol):
    id: str
    label: str

    async def health(self) -> tuple[bool, str]: ...

    async def index_exists(self, dataset_id: str) -> bool: ...

    async def delete_index(self, dataset_id: str) -> None: ...

    async def prepare_index(
        self,
        dataset_id: str,
        dimensions: int,
    ) -> None: ...

    async def index_batch(
        self,
        dataset_id: str,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> None: ...

    async def finalize_index(self, dataset_id: str) -> None: ...

    async def abort_index(self, dataset_id: str) -> None: ...

    async def lexical_search(self, dataset_id: str, query: str, limit: int) -> list[SearchHit]: ...

    async def vector_search(
        self,
        dataset_id: str,
        vector: list[float],
        limit: int,
    ) -> list[SearchHit]: ...
