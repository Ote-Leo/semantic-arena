from datasets.models import Chunk
from rag.models import QuoteSelection
from rag.service import validate_selections
from search.models import Candidate


def _candidate() -> Candidate:
    chunk = Chunk(
        id="chunk-1",
        dataset_id="demo",
        source_id="source-1",
        title="Source",
        text="Metformin is used to improve blood glucose control.",
        source_path="source.md",
        record_index=0,
        chunk_index=0,
        start_char=0,
        end_char=51,
    )
    return Candidate(
        chunk=chunk,
        fused_score=1.0,
        lexical_rank=1,
        vector_rank=1,
        lexical_score=1.0,
        vector_score=1.0,
    )


def test_validation_only_accepts_exact_quote_from_retrieved_chunk() -> None:
    candidate = _candidate()

    citations = validate_selections(
        [
            QuoteSelection(
                chunk_id="unknown", quote="Metformin is used to improve blood glucose control."
            ),
            QuoteSelection(chunk_id="chunk-1", quote="Metformin improves glucose control."),
            QuoteSelection(
                chunk_id="chunk-1", quote="Metformin is used to improve blood glucose control."
            ),
        ],
        [candidate],
    )

    assert len(citations) == 1
    assert citations[0].quote == "Metformin is used to improve blood glucose control."
