from __future__ import annotations

import json
import os
import re
import threading
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from .chunking import Chunk
from .config import settings
from .textnorm import fold_german

# Collection per chunking strategy. Separate collections guarantee the arms
# never share an index, which is the whole point of the chunking ablations.
COLLECTIONS = {
    "fixed": "raglab_fixed",
    "structure": "raglab_structure",
    "parent_child": "raglab_children",
}

# Chroma 1.5.x silently loses HNSW segments on Windows non-ASCII paths when the
# default threshold (1000) is crossed mid-write. Project paths contain Chinese
# characters, so keep the threshold above the collection size.
HNSW_METADATA = {"hnsw:sync_threshold": 10000, "hnsw:batch_size": 1000}

_WORD_RE = re.compile(r"[a-z0-9]+")
_CJK_RE = re.compile(r"[㐀-鿿]+")


def tokenize(text: str) -> list[str]:
    """Latin words (umlauts folded) + CJK character bigrams.

    The corpus is English/German, so the lexical leg is only meaningful for
    English/German questions; Chinese questions are carried by the dense leg.
    CJK bigrams are still emitted so the index stays correct if a Chinese
    document is added later.
    """
    folded = fold_german((text or "").lower())
    tokens = _WORD_RE.findall(folded)
    for run in _CJK_RE.findall(folded):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


def device() -> str:
    """auto → cuda when PyTorch was built with CUDA and a device is visible."""
    requested = os.getenv("DEVICE", "auto").strip().lower()
    if requested in {"cpu", "cuda"}:
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    target = device()
    print(f"[store] embeddings on {target}: {settings.embedding_model}")
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": target},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 64 if target == "cuda" else 32,
        },
    )


@lru_cache(maxsize=3)
def get_vectorstore(strategy: str) -> Chroma:
    if strategy not in COLLECTIONS:
        raise ValueError(f"unknown chunking strategy: {strategy}")
    return Chroma(
        persist_directory=str(settings.chroma_dir),
        embedding_function=get_embeddings(),
        collection_name=COLLECTIONS[strategy],
        collection_metadata=HNSW_METADATA,
    )


def _chroma_metadata(metadata: dict) -> dict:
    out: dict = {}
    for key, value in (metadata or {}).items():
        if isinstance(value, (list, tuple)):
            out[key] = " > ".join(str(x) for x in value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
        else:
            out[key] = str(value)
    return out


def build_collection(
    strategy: str,
    chunks: Sequence[Chunk],
    reset: bool = False,
) -> Chroma:
    embeddings = get_embeddings()
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    collection = COLLECTIONS[strategy]

    vs = Chroma(
        persist_directory=str(settings.chroma_dir),
        embedding_function=embeddings,
        collection_name=collection,
        collection_metadata=HNSW_METADATA,
    )
    if reset:
        try:
            vs.delete_collection()
        except Exception:
            pass
        vs = Chroma(
            persist_directory=str(settings.chroma_dir),
            embedding_function=embeddings,
            collection_name=collection,
            collection_metadata=HNSW_METADATA,
        )

    texts = [c.text for c in chunks]
    ids = [str(c.metadata["chunk_id"]) for c in chunks]
    metadatas = [_chroma_metadata(c.metadata) for c in chunks]
    if len(set(ids)) != len(ids):
        raise ValueError(
            f"duplicate chunk ids in {strategy}: {len(ids) - len(set(ids))}; "
            "ids must be unique or Chroma will silently drop rows"
        )

    if texts:
        vectors = embeddings.embed_documents(texts)
        batch = 128
        for start in range(0, len(texts), batch):
            vs._collection.upsert(
                ids=ids[start : start + batch],
                documents=texts[start : start + batch],
                metadatas=metadatas[start : start + batch],
                embeddings=vectors[start : start + batch],
            )
    return vs


def collection_count(strategy: str) -> int:
    try:
        return int(get_vectorstore(strategy)._collection.count())
    except Exception:
        return 0


def load_all_documents(strategy: str) -> list[Document]:
    data = get_vectorstore(strategy)._collection.get(include=["documents", "metadatas"])
    return [
        Document(page_content=text or "", metadata=dict(meta or {}))
        for text, meta in zip(data.get("documents") or [], data.get("metadatas") or [])
    ]


class BM25Index:
    def __init__(self, documents: Sequence[Document]):
        from rank_bm25 import BM25Okapi

        self.documents = list(documents)
        corpus = [tokenize(d.page_content or "") for d in self.documents]
        self.ids = [str((d.metadata or {}).get("chunk_id", f"anon-{i}")) for i, d in enumerate(self.documents)]
        self._by_id = {cid: d for cid, d in zip(self.ids, self.documents)}
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def __len__(self) -> int:
        return len(self.documents)

    def search(self, query: str, top_k: int) -> list[Document]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -float(scores[i]))
        out: list[Document] = []
        for i in order:
            if scores[i] <= 0:
                break
            out.append(self.documents[i])
            if len(out) >= top_k:
                break
        return out


@lru_cache(maxsize=3)
def get_bm25_index(strategy: str) -> BM25Index:
    return BM25Index(load_all_documents(strategy))


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    weights: Sequence[float] | None = None,
    k: int = 60,
) -> list[str]:
    if not rankings:
        return []
    if weights is None:
        weights = [1.0] * len(rankings)
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    for weight, ranking in zip(weights, rankings):
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
            first_seen.setdefault(doc_id, len(first_seen))
    return sorted(scores, key=lambda doc_id: (-scores[doc_id], first_seen[doc_id]))


@lru_cache(maxsize=1)
def get_reranker():
    from sentence_transformers import CrossEncoder

    target = device()
    print(f"[store] reranker on {target}: {settings.rerank_model}")
    return CrossEncoder(settings.rerank_model, max_length=512, device=target)


def rerank(query: str, documents: Sequence[Document], top_n: int | None = None) -> list[Document]:
    docs = list(documents)
    if not docs:
        return []
    encoder = get_reranker()
    pairs = [(query, (d.page_content or "")[:2000]) for d in docs]
    scores = encoder.predict(
        pairs, batch_size=32 if device() == "cuda" else 16, show_progress_bar=False
    )
    order = sorted(range(len(docs)), key=lambda i: -float(scores[i]))
    ranked = [docs[i] for i in order]
    return ranked[:top_n] if top_n is not None else ranked


_GPU_LOCK = threading.Lock()


def dense_search(strategy: str, query: str, k: int) -> list[Document]:
    # The agent arm issues retrieval from worker threads; a single lock keeps
    # concurrent query embeddings from hitting the GPU model at the same time.
    with _GPU_LOCK:
        pairs = get_vectorstore(strategy).similarity_search_with_score(query, k=k)
    return [doc for doc, _ in pairs]


class RetrievalResult:
    def __init__(
        self,
        docs: list[Document],
        candidates: list[Document],
        trace: dict | None = None,
    ):
        self.docs = docs            # what goes into the prompt
        self.candidates = candidates  # pre-truncation pool, for E1/E2 attribution
        self.trace = trace or {}


def retrieve(
    *,
    strategy: str,
    query: str,
    mode: str,
    k: int,
    fetch_k: int,
    use_hybrid: bool = False,
    use_rerank: bool = False,
    rerank_n: int | None = None,
    rrf_k: int = 60,
) -> RetrievalResult:
    """One retrieval pass.

    mode:
      dense  - single vector leg
      hybrid - dense + BM25 -> RRF -> optional rerank -> top-k
    `candidates` is the pool before the final top-k cut, used to separate
    "never retrieved" (E1) from "retrieved but truncated" (E2).
    """
    query = str(query).strip()
    if mode == "dense":
        pool = dense_search(strategy, query, max(fetch_k, k))
        return RetrievalResult(pool[:k], pool)

    if mode != "hybrid":
        raise ValueError(f"unknown retrieval mode: {mode}")

    dense = dense_search(strategy, query, fetch_k)
    bm25 = get_bm25_index(strategy).search(query, fetch_k)
    pool_map: dict[str, Document] = {}
    dense_ids = [str((d.metadata or {}).get("chunk_id", "")) for d in dense]
    bm25_ids = [str((d.metadata or {}).get("chunk_id", "")) for d in bm25]
    for doc in [*bm25, *dense]:
        pool_map.setdefault(str((doc.metadata or {}).get("chunk_id", "")), doc)

    fused_ids = reciprocal_rank_fusion([bm25_ids, dense_ids], weights=(0.5, 0.5), k=rrf_k)
    pool = [pool_map[i] for i in fused_ids if i in pool_map]

    if use_rerank:
        ranked = rerank(query, pool[: (rerank_n or len(pool))], top_n=None)
        return RetrievalResult(ranked[:k], ranked, {"fused": fused_ids})

    return RetrievalResult(pool[:k], pool, {"fused": fused_ids})


def parents_path(strategy: str = "parent_child") -> Path:
    return settings.parents_dir / f"{COLLECTIONS[strategy]}.json"


def save_parents(parents: Sequence[Chunk], path: Path | None = None) -> Path:
    target = path or parents_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        p.metadata["chunk_id"]: {"text": p.text, "metadata": p.metadata} for p in parents
    }
    target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return target


@lru_cache(maxsize=1)
def load_parents() -> dict[str, dict]:
    path = parents_path()
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def expand_to_parents(docs: Sequence[Document], max_parents: int) -> list[Document]:
    """Map child hits to unique parent sections, keeping first-hit rank order."""
    parents = load_parents()
    seen: set[str] = set()
    out: list[Document] = []
    for doc in docs:
        meta = doc.metadata or {}
        parent_id = str(meta.get("parent_id", ""))
        if not parent_id or parent_id in seen or parent_id not in parents:
            continue
        seen.add(parent_id)
        payload = parents[parent_id]
        out.append(
            Document(
                page_content=payload["text"],
                metadata={
                    **(payload.get("metadata") or {}),
                    "parent_id": parent_id,
                    "hit_child_id": meta.get("chunk_id"),
                    "citation_snippet": doc.page_content,
                },
            )
        )
        if len(out) >= max_parents:
            break
    return out


def iter_collection_chunks(strategy: str) -> Iterable[dict]:
    data = get_vectorstore(strategy)._collection.get(include=["documents", "metadatas"])
    for cid, text, meta in zip(data["ids"], data["documents"], data["metadatas"]):
        yield {"chunk_id": cid, "text": text or "", "metadata": dict(meta or {})}
