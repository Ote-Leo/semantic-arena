from __future__ import annotations

import re


def index_name(dataset_id: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", dataset_id.lower()).strip("_")
    return f"arena_{normalized}"[:80]
