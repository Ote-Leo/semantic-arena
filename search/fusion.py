from __future__ import annotations

from search.models import Candidate, SearchHit


def fuse_rankings(
    lexical_hits: list[SearchHit],
    vector_hits: list[SearchHit],
    lexical_weight: float,
    vector_weight: float,
    rrf_k: int,
    limit: int,
) -> list[Candidate]:
    """Fuse ranked lists with weighted reciprocal-rank fusion."""
    merged: dict[str, dict[str, SearchHit | int | float | None]] = {}
    for rank, hit in enumerate(lexical_hits, start=1):
        row = merged.setdefault(
            hit.chunk.id,
            {
                "hit": hit,
                "lexical_rank": None,
                "vector_rank": None,
                "lexical_score": None,
                "vector_score": None,
            },
        )
        row["lexical_rank"] = rank
        row["lexical_score"] = hit.score
    for rank, hit in enumerate(vector_hits, start=1):
        row = merged.setdefault(
            hit.chunk.id,
            {
                "hit": hit,
                "lexical_rank": None,
                "vector_rank": None,
                "lexical_score": None,
                "vector_score": None,
            },
        )
        row["vector_rank"] = rank
        row["vector_score"] = hit.score

    candidates: list[Candidate] = []
    for row in merged.values():
        lexical_rank = row["lexical_rank"]
        vector_rank = row["vector_rank"]
        fused_score = 0.0
        if isinstance(lexical_rank, int):
            fused_score += lexical_weight / (rrf_k + lexical_rank)
        if isinstance(vector_rank, int):
            fused_score += vector_weight / (rrf_k + vector_rank)
        candidates.append(
            Candidate(
                chunk=row["hit"].chunk,  # type: ignore[union-attr]
                fused_score=fused_score,
                lexical_rank=lexical_rank if isinstance(lexical_rank, int) else None,
                vector_rank=vector_rank if isinstance(vector_rank, int) else None,
                lexical_score=(
                    row["lexical_score"] if isinstance(row["lexical_score"], float) else None
                ),
                vector_score=(
                    row["vector_score"] if isinstance(row["vector_score"], float) else None
                ),
            )
        )
    return sorted(candidates, key=lambda candidate: candidate.fused_score, reverse=True)[:limit]
