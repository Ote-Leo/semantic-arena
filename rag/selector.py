from __future__ import annotations

import json
import re
from typing import Protocol

import httpx

from rag.models import QuoteSelection, SelectionPayload
from search.models import Candidate


class SelectionError(RuntimeError):
    pass


class EvidenceSelector(Protocol):
    async def select(
        self,
        question: str,
        candidates: list[Candidate],
        maximum_quotes: int,
    ) -> list[QuoteSelection]: ...


class DeterministicExtractiveSelector:
    """Conservative fallback for local setups without an answer model."""

    async def select(
        self,
        question: str,
        candidates: list[Candidate],
        maximum_quotes: int,
    ) -> list[QuoteSelection]:
        query_terms = _terms(question)
        if not query_terms:
            return []
        selections: list[QuoteSelection] = []
        for candidate in candidates:
            best_sentence = _best_matching_sentence(candidate.chunk.text, query_terms)
            if best_sentence is None:
                continue
            selections.append(QuoteSelection(chunk_id=candidate.chunk.id, quote=best_sentence))
            if len(selections) == maximum_quotes:
                break
        return selections


class OpenAICompatibleEvidenceSelector:
    def __init__(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        model: str,
        api_key: str = "",
    ) -> None:
        self.client = client
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    async def select(
        self,
        question: str,
        candidates: list[Candidate],
        maximum_quotes: int,
    ) -> list[QuoteSelection]:
        if not candidates:
            return []
        context = "\n\n".join(
            f"[chunk_id={candidate.chunk.id}]\n{candidate.chunk.text}" for candidate in candidates
        )
        prompt = (
            "Select evidence that directly answers the question using only the supplied chunks. "
            "Do not answer from your own knowledge. Return JSON only in this exact shape: "
            '{"selections":[{"chunk_id":"...","quote":"exact text copied from that chunk"}]}. '
            "A quote must be a verbatim substring of its selected chunk. "
            "Return an empty selections array "
            "when no supplied chunk directly answers the question. Do not paraphrase.\n\n"
            f"Question: {question}\n\nEvidence:\n{context}"
        )
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json={
                    "model": self.model,
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are an evidence selector. Output JSON and "
                                "never use outside knowledge."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                },
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            payload = SelectionPayload.model_validate_json(_json_object(content))
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
            raise SelectionError(f"Evidence selection failed: {error}") from error
        return payload.selections[:maximum_quotes]


_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "be",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}
_WORD = re.compile(r"[a-z0-9][a-z0-9'-]*", re.IGNORECASE)
_SENTENCE = re.compile(r"[^.!?]+[.!?]?", re.DOTALL)


def _terms(text: str) -> set[str]:
    return {word.lower() for word in _WORD.findall(text) if word.lower() not in _STOP_WORDS}


def _best_matching_sentence(text: str, query_terms: set[str]) -> str | None:
    minimum_matches = min(2, len(query_terms))
    best: tuple[int, str] | None = None
    for match in _SENTENCE.finditer(text):
        sentence = match.group(0).strip()
        if not sentence:
            continue
        overlap = len(_terms(sentence) & query_terms)
        if overlap < minimum_matches:
            continue
        if best is None or overlap > best[0]:
            best = (overlap, sentence)
    return best[1] if best else None


def _json_object(content: str) -> str:
    """Accept a JSON object even if a provider ignored response_format fencing."""
    stripped = content.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Provider response did not contain a JSON object.")
    candidate = stripped[start : end + 1]
    json.loads(candidate)
    return candidate
