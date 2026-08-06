from __future__ import annotations

import hashlib
import re

from datasets.models import Chunk, SourceDocument

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n+")


def chunk_document(
    document: SourceDocument,
    dataset_id: str,
    max_characters: int,
    overlap_characters: int,
) -> list[Chunk]:
    """Split a document deterministically while preferring paragraph boundaries."""
    text = document.text.strip()
    if not text:
        return []

    paragraphs = _paragraphs_with_offsets(text)
    windows: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        target = min(start + max_characters, len(text))
        end = (
            target
            if target == len(text)
            else _best_boundary(paragraphs, start, target, overlap_characters)
        )
        if end <= start:
            end = target
        windows.append((start, end))
        if end == len(text):
            break
        start = max(end - overlap_characters, start + 1)

    chunks: list[Chunk] = []
    for chunk_index, (start, end) in enumerate(windows):
        chunk_text = text[start:end].strip()
        left_trim = len(text[start:end]) - len(text[start:end].lstrip())
        right_trim = len(text[start:end]) - len(text[start:end].rstrip())
        adjusted_start = start + left_trim
        adjusted_end = end - right_trim
        if not chunk_text:
            continue
        identity = f"{dataset_id}\x1f{document.id}\x1f{adjusted_start}\x1f{adjusted_end}"
        chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
        chunks.append(
            Chunk(
                id=chunk_id,
                dataset_id=dataset_id,
                source_id=document.id,
                title=document.title or document.source_path,
                text=chunk_text,
                source_path=document.source_path,
                record_index=document.record_index,
                chunk_index=chunk_index,
                start_char=adjusted_start,
                end_char=adjusted_end,
            )
        )
    return chunks


def _paragraphs_with_offsets(text: str) -> list[tuple[int, int]]:
    paragraphs: list[tuple[int, int]] = []
    start = 0
    for match in _PARAGRAPH_BREAK.finditer(text):
        if match.start() > start:
            paragraphs.append((start, match.start()))
        start = match.end()
    if start < len(text):
        paragraphs.append((start, len(text)))
    return paragraphs


def _best_boundary(
    paragraphs: list[tuple[int, int]], start: int, target: int, overlap: int
) -> int:
    earliest = target - overlap
    candidates = [end for _, end in paragraphs if max(start, earliest) < end <= target]
    if candidates:
        return candidates[-1]
    return target
