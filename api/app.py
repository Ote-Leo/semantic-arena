from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.models import (
    CandidateResponse,
    ChatRequest,
    ChatResponse,
    ChunkCitationResponse,
    CitationResponse,
    DatasetResponse,
    EngineAnswerResponse,
    EngineStatusResponse,
    HealthResponse,
    IndexRequest,
    JobResponse,
    TimingsResponse,
)
from datasets.catalog import DatasetCatalog, DatasetNotFoundError
from embeddings.base import Embedder, EmbeddingError
from embeddings.openai_compatible import OpenAICompatibleEmbedder
from indexing.service import IndexingService
from jobs.manager import JobManager, JobRecord
from meilisearch.adapter import MeilisearchSearchEngine
from rag.selector import (
    DeterministicExtractiveSelector,
    EvidenceSelector,
    OpenAICompatibleEvidenceSelector,
)
from rag.service import AnswerService
from search.base import SearchEngine, SearchEngineError
from search.fusion import fuse_rankings
from search.naming import index_name
from settings import ROOT_DIR, ArenaConfig, RuntimeSettings, get_arena_config, get_runtime_settings
from solr.adapter import SolrSearchEngine

FRONTEND_DIR = ROOT_DIR / "frontend"


@dataclass(slots=True)
class ArenaServices:
    catalog: DatasetCatalog
    embedder: Embedder
    engines: dict[str, SearchEngine]
    answering: AnswerService
    indexing: IndexingService
    jobs: JobManager
    selector_mode: str


def create_app(
    runtime: RuntimeSettings | None = None,
    arena_config: ArenaConfig | None = None,
    catalog: DatasetCatalog | None = None,
    embedder: Embedder | None = None,
    engines: dict[str, SearchEngine] | None = None,
    selector: EvidenceSelector | None = None,
) -> FastAPI:
    runtime = runtime or get_runtime_settings()
    arena_config = arena_config or get_arena_config()
    catalog = catalog or DatasetCatalog(chunking=arena_config.chunking)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        client = httpx.AsyncClient(timeout=httpx.Timeout(60.0))
        active_embedder = embedder or OpenAICompatibleEmbedder(
            client,
            runtime.embedding_base_url,
            runtime.embedding_model,
            runtime.embedding_dimensions,
            runtime.embedding_api_key,
        )
        active_engines = engines or {
            "solr": SolrSearchEngine(client, runtime.solr_url, runtime.solr_configset),
            "meilisearch": MeilisearchSearchEngine(
                client, runtime.meilisearch_url, runtime.meilisearch_master_key
            ),
        }
        if selector is not None:
            active_selector = selector
            selector_mode = "custom"
        elif runtime.selector_base_url and runtime.selector_model:
            active_selector = OpenAICompatibleEvidenceSelector(
                client,
                runtime.selector_base_url,
                runtime.selector_model,
                runtime.selector_api_key,
            )
            selector_mode = "openai-compatible"
        else:
            active_selector = DeterministicExtractiveSelector()
            selector_mode = "deterministic-extractive"
        indexing = IndexingService(catalog, active_embedder, active_engines)
        app.state.services = ArenaServices(
            catalog=catalog,
            embedder=active_embedder,
            engines=active_engines,
            answering=AnswerService(active_selector, arena_config.answering.maximum_quotes),
            indexing=indexing,
            jobs=JobManager(indexing),
            selector_mode=selector_mode,
        )
        yield
        await client.aclose()

    app = FastAPI(title="Semantic Arena", version="0.1.0", lifespan=lifespan)
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")

    @app.get("/", include_in_schema=False)
    async def frontend() -> FileResponse:
        return FileResponse(
            FRONTEND_DIR / "index.html",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        services = _services(request)
        statuses = await asyncio.gather(
            *(engine.health() for engine in services.engines.values()), return_exceptions=True
        )
        engines_response: list[EngineStatusResponse] = []
        for engine, result in zip(services.engines.values(), statuses, strict=True):
            if isinstance(result, Exception):
                engines_response.append(
                    EngineStatusResponse(
                        id=engine.id, label=engine.label, available=False, detail=str(result)
                    )
                )
            else:
                available, detail = result
                engines_response.append(
                    EngineStatusResponse(
                        id=engine.id, label=engine.label, available=available, detail=detail
                    )
                )
        configured = bool(getattr(services.embedder, "configured", True))
        return HealthResponse(
            engines=engines_response,
            embedding_configured=configured,
            selector_mode=services.selector_mode,
        )

    @app.get("/api/engines", response_model=list[EngineStatusResponse])
    async def list_engines(request: Request) -> list[EngineStatusResponse]:
        services = _services(request)
        statuses = await asyncio.gather(
            *(engine.health() for engine in services.engines.values()), return_exceptions=True
        )
        responses: list[EngineStatusResponse] = []
        for engine, result in zip(services.engines.values(), statuses, strict=True):
            if isinstance(result, Exception):
                responses.append(
                    EngineStatusResponse(
                        id=engine.id, label=engine.label, available=False, detail=str(result)
                    )
                )
            else:
                available, detail = result
                responses.append(
                    EngineStatusResponse(
                        id=engine.id, label=engine.label, available=available, detail=detail
                    )
                )
        return responses

    @app.get("/api/datasets", response_model=list[DatasetResponse])
    async def list_datasets(request: Request) -> list[DatasetResponse]:
        services = _services(request)
        manifests = services.catalog.manifests()
        results: list[DatasetResponse] = []
        for manifest in manifests:
            statuses = await asyncio.gather(
                *(engine.index_exists(manifest.id) for engine in services.engines.values()),
                return_exceptions=True,
            )
            engine_indexes = {
                engine.id: bool(indexed) if not isinstance(indexed, Exception) else False
                for engine, indexed in zip(services.engines.values(), statuses, strict=True)
            }
            results.append(
                DatasetResponse(
                    id=manifest.id,
                    name=manifest.name,
                    description=manifest.description,
                    index_name=index_name(manifest.id),
                    engine_indexes=engine_indexes,
                )
            )
        return results

    @app.post("/api/indexes", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
    async def create_index(request: Request, payload: IndexRequest) -> JobResponse:
        services = _services(request)
        _require_dataset(services.catalog, payload.dataset_id)
        _require_engines(services, payload.engine_ids)
        return _job_response(services.jobs.start(payload.dataset_id, payload.engine_ids))

    @app.delete(
        "/api/indexes/{dataset_id}/{engine_id}",
        status_code=status.HTTP_204_NO_CONTENT,
    )
    async def delete_index(request: Request, dataset_id: str, engine_id: str) -> Response:
        services = _services(request)
        _require_dataset(services.catalog, dataset_id)
        engine = _require_engines(services, [engine_id])[0]
        try:
            await engine.delete_index(dataset_id)
        except SearchEngineError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.get("/api/jobs/{job_id}", response_model=JobResponse)
    async def get_job(request: Request, job_id: str) -> JobResponse:
        record = _services(request).jobs.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Indexing job was not found.")
        return _job_response(record)

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(request: Request, payload: ChatRequest) -> ChatResponse:
        services = _services(request)
        _require_dataset(services.catalog, payload.dataset_id)
        selected_engines = _require_engines(services, payload.engine_ids)
        embedding_started = time.perf_counter()
        try:
            vector = (await services.embedder.embed_many([payload.query]))[0]
        except (EmbeddingError, IndexError) as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        embedding_ms = _milliseconds(embedding_started)
        responses = await asyncio.gather(
            *(
                _answer_with_engine(
                    engine,
                    services.answering,
                    arena_config,
                    payload.dataset_id,
                    payload.query,
                    vector,
                )
                for engine in selected_engines
            )
        )
        return ChatResponse(
            dataset_id=payload.dataset_id,
            query=payload.query,
            embedding_ms=embedding_ms,
            results=responses,
        )

    @app.get("/api/citations/{dataset_id}/{chunk_id}", response_model=ChunkCitationResponse)
    async def citation(request: Request, dataset_id: str, chunk_id: str) -> ChunkCitationResponse:
        try:
            chunk = _services(request).catalog.chunk(dataset_id, chunk_id)
        except DatasetNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return ChunkCitationResponse(
            chunk_id=chunk.id,
            dataset_id=chunk.dataset_id,
            title=chunk.title,
            text=chunk.text,
            source_path=chunk.source_path,
            record_index=chunk.record_index,
            chunk_index=chunk.chunk_index,
            start_char=chunk.start_char,
            end_char=chunk.end_char,
        )

    return app


async def _answer_with_engine(
    engine: SearchEngine,
    answering: AnswerService,
    arena_config: ArenaConfig,
    dataset_id: str,
    query: str,
    vector: list[float],
) -> EngineAnswerResponse:
    started = time.perf_counter()
    try:
        lexical_started = time.perf_counter()
        lexical_task = asyncio.create_task(
            engine.lexical_search(dataset_id, query, arena_config.retrieval.lexical_limit)
        )
        vector_started = time.perf_counter()
        vector_task = asyncio.create_task(
            engine.vector_search(dataset_id, vector, arena_config.retrieval.vector_limit)
        )
        lexical_hits, vector_hits = await asyncio.gather(lexical_task, vector_task)
        lexical_ms = _milliseconds(lexical_started)
        vector_ms = _milliseconds(vector_started)
        fusion_started = time.perf_counter()
        candidates = fuse_rankings(
            lexical_hits,
            vector_hits,
            arena_config.retrieval.lexical_weight,
            arena_config.retrieval.vector_weight,
            arena_config.retrieval.rrf_k,
            arena_config.retrieval.final_limit,
        )
        fusion_ms = _milliseconds(fusion_started)
        selection_started = time.perf_counter()
        answer = await answering.answer(query, candidates)
        selection_ms = _milliseconds(selection_started)
        return EngineAnswerResponse(
            engine_id=engine.id,
            engine_label=engine.label,
            status="ok",
            message=answer.message,
            no_data=answer.no_data,
            citations=[
                CitationResponse(
                    chunk_id=citation.chunk.id,
                    quote=citation.quote,
                    title=citation.chunk.title,
                    source_path=citation.chunk.source_path,
                    record_index=citation.chunk.record_index,
                    chunk_index=citation.chunk.chunk_index,
                    start_char=citation.chunk.start_char,
                    end_char=citation.chunk.end_char,
                )
                for citation in answer.citations
            ],
            candidates=[
                CandidateResponse(
                    chunk_id=candidate.chunk.id,
                    title=candidate.chunk.title,
                    source_path=candidate.chunk.source_path,
                    excerpt=candidate.chunk.text[:500],
                    fused_score=candidate.fused_score,
                    lexical_rank=candidate.lexical_rank,
                    vector_rank=candidate.vector_rank,
                    lexical_score=candidate.lexical_score,
                    vector_score=candidate.vector_score,
                )
                for candidate in candidates
            ],
            timings=TimingsResponse(
                lexical_ms=lexical_ms,
                vector_ms=vector_ms,
                fusion_ms=fusion_ms,
                selection_ms=selection_ms,
                total_ms=_milliseconds(started),
            ),
        )
    except Exception as error:
        return EngineAnswerResponse(
            engine_id=engine.id,
            engine_label=engine.label,
            status="error",
            error=str(error),
        )


def _services(request: Request) -> ArenaServices:
    return request.app.state.services


def _require_dataset(catalog: DatasetCatalog, dataset_id: str) -> None:
    try:
        catalog.manifest(dataset_id)
    except DatasetNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


def _require_engines(services: ArenaServices, engine_ids: list[str]) -> list[SearchEngine]:
    duplicate = len(engine_ids) != len(set(engine_ids))
    unknown = sorted(set(engine_ids) - set(services.engines))
    if duplicate or unknown:
        detail = "Engine IDs must be unique and known."
        if unknown:
            detail = f"Unknown engine IDs: {', '.join(unknown)}."
        raise HTTPException(status_code=422, detail=detail)
    return [services.engines[engine_id] for engine_id in engine_ids]


def _job_response(record: JobRecord) -> JobResponse:
    return JobResponse(
        id=record.id,
        dataset_id=record.dataset_id,
        engine_ids=record.engine_ids,
        status=record.status,
        created_at=record.created_at,
        started_at=record.started_at,
        finished_at=record.finished_at,
        result=record.result,
        error=record.error,
    )


def _milliseconds(started: float) -> float:
    return round((time.perf_counter() - started) * 1_000, 2)


app = create_app()
