from core.chunking import (
    fixed_page_chunks,
    parent_child_chunks,
    stable_chunk_id,
    structure_chunks,
)
from core.parsing import Element


def _elements():
    return [
        Element("heading", "2 Method", 1, level=2),
        Element("paragraph", "A" * 600, 1),
        Element("paragraph", "B" * 600, 2),
        Element("heading", "2.1 Table results", 2, level=3),
        Element("paragraph", "C" * 200, 2),
        Element("table", "| a | b |\n| --- | --- |\n| 1 | 2 |", 2),
    ]


def test_stable_chunk_id_is_deterministic():
    assert stable_chunk_id("doc.pdf", 1, "hello") == stable_chunk_id("doc.pdf", 1, " hello ")
    assert stable_chunk_id("doc.pdf", 1, "hello") != stable_chunk_id("doc.pdf", 2, "hello")


def test_structure_chunks_keep_tables_atomic_and_carry_heading_path():
    chunks = structure_chunks(_elements(), "doc.pdf", "Doc", max_chars=800)
    tables = [c for c in chunks if c.metadata["chunk_type"] == "table"]
    assert len(tables) == 1
    assert tables[0].text.startswith("2 Method > 2.1 Table results")
    assert "| 1 | 2 |" in tables[0].text
    assert all(c.metadata["heading_path"] for c in chunks if c.metadata["chunk_type"] == "text")


def test_structure_chunks_respect_max_size():
    chunks = structure_chunks(_elements(), "doc.pdf", "Doc", max_chars=800)
    # heading prefix plus body can exceed max_chars slightly, but not by 2x
    assert all(len(c.text) < 1600 for c in chunks)


def test_parent_child_links_every_child_to_a_parent():
    parents, children = parent_child_chunks(
        _elements(), "doc.pdf", "Doc", parent_size=1000, child_size=250
    )
    parent_ids = {p.metadata["chunk_id"] for p in parents}
    assert children
    assert all(c.metadata["parent_id"] in parent_ids for c in children)
    assert all(len(c.text) < 500 for c in children)


def test_fixed_chunks_are_page_local_with_overlap():
    chunks = fixed_page_chunks(
        [1, 2],
        ["word " * 500, "other " * 500],
        doc="doc.pdf",
        doc_title="Doc",
        chunk_size=200,
        chunk_overlap=30,
    )
    assert {c.metadata["page_start"] for c in chunks} == {1, 2}
    assert len({c.metadata["chunk_id"] for c in chunks}) == len(chunks)
