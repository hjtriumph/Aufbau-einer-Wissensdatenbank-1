from langchain_core.documents import Document

from core.citations import (
    detect_refusal,
    format_context,
    parse_citation_labels,
    resolve_citations,
)


def _docs():
    return [
        Document(
            page_content="Solar gains dominate cooling load.",
            metadata={
                "doc": "a.pdf",
                "doc_title": "Paper A",
                "page_start": 3,
                "page_end": 4,
                "chunk_id": "c1",
                "heading_path": "Results",
            },
        ),
        Document(
            page_content="Outdoor temperature dominates heating load.",
            metadata={
                "doc": "b.pdf",
                "doc_title": "Paper B",
                "page_start": 7,
                "page_end": 7,
                "chunk_id": "c2",
            },
        ),
    ]


def test_format_context_labels_and_metadata():
    context, citations = format_context(_docs())
    assert "[C1] Paper A | page=3-4" in context
    assert "[C2] Paper B | page=7" in context
    assert citations[0]["label"] == "C1"
    assert citations[1]["page"] == "7"


def test_parse_and_resolve_citations_ignores_unknown_labels():
    _, citations = format_context(_docs())
    labels = parse_citation_labels("A claim [C1]. Another [C2], and a bad one [C9].")
    assert labels == ["C1", "C2", "C9"]
    cited, unknown = resolve_citations(labels, citations)
    assert [c["label"] for c in cited] == ["C1", "C2"]
    assert unknown == ["C9"]


def test_refusal_detection_multilingual():
    assert detect_refusal("根据现有资料无法回答。")
    assert detect_refusal("Based on the available documents, this question cannot be answered.")
    assert detect_refusal("Mit den vorliegenden Unterlagen kann diese Frage nicht beantwortet werden.")
    assert not detect_refusal("太阳能得热是冷却负荷的主要来源 [C1]。")
