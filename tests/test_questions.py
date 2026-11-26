import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"

RESERVED_FEWSHOT_DOCS = (
    "Nahfeldexposition",
    "Schulen_inhalierte_Dosis",
    "hartmann_kriegel_risikobewertung",
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text or "")).strip().lower()


def _load(name: str) -> list[dict]:
    return [
        json.loads(line)
        for line in (CORPUS / name).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _page_texts() -> dict[str, dict[int, str]]:
    out: dict[str, dict[int, str]] = {}
    for path in (CORPUS / "text").glob("*.txt"):
        parts = re.split(r"\n===== PAGE (\d+) =====\n", path.read_text(encoding="utf-8"))
        out[path.name] = {int(parts[i]): parts[i + 1] for i in range(1, len(parts), 2)}
    return out


def test_question_composition_matches_preregistration():
    questions = _load("questions.jsonl")
    assert len(questions) == 30
    counts = Counter(q["type"] for q in questions)
    assert counts == {
        "numeric": 5,
        "single_hop": 5,
        "multi_hop": 6,
        "method": 6,
        "unanswerable": 8,
    }
    assert sum(1 for q in questions if q["answerable"]) == 22


def test_every_quote_appears_on_its_page():
    texts = _page_texts()
    failures = []
    for question in _load("questions.jsonl"):
        for ev in question["gold_evidence"]:
            key = Path(ev["doc"]).stem + ".txt"
            haystack = texts.get(key, {}).get(int(ev["page"]), "")
            if _normalize(ev["quote"]) not in _normalize(haystack):
                failures.append((question["id"], ev["doc"], ev["page"], ev["quote"][:60]))
    assert failures == []


def test_unanswerable_cases_have_no_evidence():
    for question in _load("questions.jsonl"):
        if question["type"] == "unanswerable":
            assert question["answerable"] is False
            assert question["gold_evidence"] == []
        else:
            assert question["answerable"] is True
            assert question["gold_evidence"]


def test_fewshot_examples_never_touch_benchmark_documents():
    for question in _load("questions.jsonl"):
        for ev in question["gold_evidence"]:
            assert not any(doc in ev["doc"] for doc in RESERVED_FEWSHOT_DOCS), question["id"]


def test_variants_share_gold_evidence_with_their_base():
    base = {q["id"]: q for q in _load("questions.jsonl")}
    variants = _load("variants.jsonl")
    assert len(variants) == 10
    assert Counter(v["language"] for v in variants) == {"en": 5, "de": 5}
    for variant in variants:
        assert variant["gold_evidence"] == base[variant["base_id"]]["gold_evidence"]
        assert variant["answerable"] is True
