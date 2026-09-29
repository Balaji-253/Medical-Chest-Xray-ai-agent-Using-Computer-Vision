import json
import re
from pathlib import Path

from app.config import EVIDENCE_FILE


def load_evidence() -> list[dict]:
    """Load local research evidence records."""
    if not EVIDENCE_FILE.exists():
        return []

    try:
        return json.loads(
            EVIDENCE_FILE.read_text(
                encoding="utf-8"
            )
        )
    except (json.JSONDecodeError, OSError):
        return []


def _tokenize(text: str) -> set[str]:
    """Convert text into normalized searchable tokens."""
    return set(
        re.findall(
            r"[a-zA-Z0-9_]+",
            text.lower(),
        )
    )


def retrieve(
    question: str,
    limit: int = 5,
) -> list[dict]:
    """
    Retrieve the most relevant local research references.

    Matching considers:
    - title
    - summary
    - topics

    This is a research-reference retrieval mechanism.
    It does not establish a medical diagnosis.
    """

    evidence = load_evidence()

    if not evidence:
        return []

    query_tokens = _tokenize(question)

    scored = []

    for item in evidence:
        title = item.get("title", "")
        summary = item.get("summary", "")

        topics = item.get(
            "topics",
            [],
        )

        if not isinstance(topics, list):
            topics = []

        topic_text = " ".join(
            str(topic)
            for topic in topics
        )

        title_tokens = _tokenize(title)
        summary_tokens = _tokenize(summary)
        topic_tokens = _tokenize(topic_text)

        # Weighted matching:
        # topics > title > summary
        topic_score = len(
            query_tokens & topic_tokens
        )

        title_score = len(
            query_tokens & title_tokens
        )

        summary_score = len(
            query_tokens & summary_tokens
        )

        score = (
            topic_score * 3
            + title_score * 2
            + summary_score
        )

        if score > 0:
            scored.append(
                (
                    score,
                    item,
                )
            )

    scored.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    return [
        item
        for _, item in scored[:limit]
    ]