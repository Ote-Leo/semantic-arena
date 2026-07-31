from datasets.models import Chunk
from search.fusion import fuse_rankings
from search.models import SearchHit


def _chunk(chunk_id: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        dataset_id="demo",
        source_id=chunk_id,
        title=chunk_id,
        text=chunk_id,
        source_path="test.txt",
        record_index=0,
        chunk_index=0,
        start_char=0,
        end_char=len(chunk_id),
    )


def test_weighted_rrf_rewards_hits_present_in_both_rankings() -> None:
    first, second, third = _chunk("first"), _chunk("second"), _chunk("third")

    results = fuse_rankings(
        lexical_hits=[SearchHit(first, 10.0), SearchHit(second, 5.0)],
        vector_hits=[SearchHit(second, 0.9), SearchHit(third, 0.8)],
        lexical_weight=1.0,
        vector_weight=1.0,
        rrf_k=60,
        limit=3,
    )

    assert [result.chunk.id for result in results] == ["second", "first", "third"]
    assert results[0].lexical_rank == 2
    assert results[0].vector_rank == 1
