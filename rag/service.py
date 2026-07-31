from __future__ import annotations

from rag.models import NO_DATA_MESSAGE, Citation, GroundedAnswer, QuoteSelection
from rag.selector import EvidenceSelector
from search.models import Candidate


class AnswerService:
    def __init__(self, selector: EvidenceSelector, maximum_quotes: int) -> None:
        self.selector = selector
        self.maximum_quotes = maximum_quotes

    async def answer(self, question: str, candidates: list[Candidate]) -> GroundedAnswer:
        selections = await self.selector.select(question, candidates, self.maximum_quotes)
        citations = validate_selections(selections, candidates)
        if not citations:
            return GroundedAnswer(message=NO_DATA_MESSAGE, citations=[], no_data=True)
        return GroundedAnswer(
            message="\n\n".join(citation.quote for citation in citations),
            citations=citations,
            no_data=False,
        )


def validate_selections(
    selections: list[QuoteSelection], candidates: list[Candidate]
) -> list[Citation]:
    """Discard all model output that cannot be proven to be retrieved source text."""
    chunks = {candidate.chunk.id: candidate.chunk for candidate in candidates}
    citations: list[Citation] = []
    seen: set[tuple[str, str]] = set()
    for selection in selections:
        chunk = chunks.get(selection.chunk_id)
        identity = (selection.chunk_id, selection.quote)
        if chunk is None or identity in seen or selection.quote not in chunk.text:
            continue
        citations.append(Citation(chunk=chunk, quote=selection.quote))
        seen.add(identity)
    return citations
