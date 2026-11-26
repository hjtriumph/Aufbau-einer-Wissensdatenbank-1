# RAG Comparative Experiment: Vanilla LLM vs. Basic RAG vs. Optimized RAG

&gt; Operation manual (Word / PDF): `docs/RAG-Vergleichsexperiment_Betriebshandbuch_DE.docx`, `docs/RAG-Vergleichsexperiment_Betriebshandbuch_DE.pdf` — covers environment setup, index building, frontend launch, experiments, human scoring (including a practice mode), export, and troubleshooting.

On a small vertical corpus of ~20 papers, a controlled experiment answers a prerequisite question: **how much incremental gain does finely optimized RAG deliver over basic RAG, at what cost, and which tricks actually work?**

- Corpus: 20 HVAC / building energy / indoor air quality papers (~409 pages, mixed English/German)
- 16 experimental arms, 3 orthogonal dimensions + ablation attribution
- Frozen question bank of 30 questions (22 answerable + 8 unanswerable), plus 5 questions × English/German variants
- All arms share a single pipeline; differences live only in `arms/configs/*.yaml`
- Evaluation includes retrieval metrics, LLM judge, paired bootstrap, human kappa calibration, and a failure taxonomy

## Directory layout

    rag_lab/
    ├── corpus/
    │   ├── raw/                 20 PDFs (read-only)
    │   ├── text/                Per-page plain text (for question authoring and citation validation)
    │   ├── documents.json       Bibliographic metadata (title/author/language/publication info)
    │   ├── questions.jsonl      Frozen question bank of 30 questions (with gold-standard evidence)
    │   └── variants.jsonl       5 questions × equivalent English/German versions
    ├── core/                    Parsing, chunking, storage, retrieval, LLM, citations
    ├── ingest/                  Three indexes: basic / structured / parent_child
    ├── arms/
    │   ├── configs/             YAML for the 16 arms (the only source of differences)
    │   ├── schema.py            Config validation
    │   ├── pipeline.py          Shared pipeline: retrieve → fuse → rerank → assemble → generate
    │   └── agent.py             ReAct loop (arm D only)
    ├── prompts/templates.py     Frozen templates T1–T7
    ├── eval/
    │   ├── runner.py            Experiment scheduler → results/&lt;run&gt;/{raw,summary,by_type,paired}.csv
    │   ├── judge.py             T7 LLM judging (with on-disk cache)
    │   ├── metrics.py           Recall/MRR/nDCG, bootstrap, weighted kappa
    │   ├── case_loader.py       Question-bank loading and evidence→chunk-ID resolution
    │   ├── human.py             Export of 25 human-audit samples and consistency computation
    │   └── visualize.py         9 figures + single-file dashboard.html
    ├── demo/                    FastAPI + single-page frontend (arm C live Q&A + A/B/C precomputed comparison)
    ├── tests/                   Unit and integration tests
    └── results/                 Run outputs

## Environment

Verified environment: `chain_1_fixed` (Python 3.12, torch CPU, chromadb 1.5.8).

    conda activate chain_1_fixed
    pip install -r requirements.txt
    cp .env.example .env   # fill in DEEPSEEK_API_KEY

**GPU (recommended)**: by default `DEVICE=auto` — if CUDA is detected, bge-m3 and the reranker are placed on GPU. On an RTX 4070 Laptop 8GB, both models co-resident take ~5 GB; reranking 20 candidates takes ~0.7s (CPU: ~38s).

    pip install "torch==2.11.0+cu128" --index-url https://download.pytorch.org/whl/cu128
    # To force CPU, set DEVICE=cpu in .env

Default models and locked settings:

| Item | Value |
| --- | --- |
| Generation model | `deepseek-chat`, temperature 0.2 (identical across all arms) |
| Embedding | `BAAI/bge-m3` (multilingual; cross-lingual retrieval across Chinese/English/German) |
| Reranker | `BAAI/bge-reranker-v2-m3` (a component-level change required by trilingual support) |
| Fusion | RRF, BM25 : vector = 0.5 : 0.5, k=60 |
| Main-arm context | top-6; D/D' budget: 18 chunks |
| Judge | `deepseek-chat`, temperature 0 |

First run downloads `bge-m3` and `bge-reranker-v2-m3` from the HF mirror (~2 GB + 2.3 GB).

## Building the indexes

    python -m ingest.basic --reset          # fixed 1000/150 recursive chunking
    python -m ingest.structured --reset     # structure-aware chunking (blocks start at headings, tables atomic)
    python -m ingest.parent_child --reset   # 1800-char parents + 250-char children
    # or build all three at once:
    python scripts/build_all_indexes.py --reset

Ingest reports are written to `storage/ingest_report_&lt;strategy&gt;.json` (document count, chunk count, duplicate IDs). Chunk IDs are content + page + position hashes; re-ingestion is idempotent; duplicate IDs raise an error rather than letting Chroma silently drop data.

**Windows non-ASCII path pitfall**: Chroma 1.5.x on non-ASCII paths, once past the default `hnsw:sync_threshold=1000`, can silently lose HNSW segments. All collections in this project are created with `sync_threshold=10000`, `batch_size=1000`.

## Running experiments

    # Smoke test: 1 answerable + 1 unanswerable × A/B/C/D × 1 repeat
    python -m eval.runner --smoke

    # Phase 1: 4 main arms + D'
    python -m eval.runner --arms A "A'" B C "D'" --workers 8

    # Full 16 arms (30 questions × 3 repeats = 1,440 rows)
    python -m eval.runner --workers 8

    # English/German variants (run separately, reported per language)
    python -m eval.runner --include-variants --workers 8

Output directory `results/YYYYMMDD-HHMMSS/`:

| File | Contents |
| --- | --- |
| `raw.jsonl` | One row per case: answer, context, retrieved chunk IDs, candidate pool, tokens, latency, judge scores, `error_class` |
| `summary.csv` | One row per arm: completeness/relevance/faithfulness/citation precision/refusal/latency/cost/retrieval metrics |
| `by_type.csv` | Arm × question type |
| `by_language.csv` | Arm × language (Chinese main report; English/German reported separately) |
| `paired.csv` | Paired bootstrap of each arm vs. baseline (mean difference, 95% CI, p-value) |
| `errors.csv` | Wrong answers and failure categories |
| `run_meta.json` | Models, temperatures, question-bank hash, full arm configs (pre-registration evidence) |

Retrieval is deterministic: each (retrieval config, question) pair runs exactly once; the 3 repeats share the result, but every repeat still counts measured retrieval latency. Arms differing only in downstream config (C / C-P1 / C-P2) share the same retrieval pass, avoiding redundant reranking on CPU.

### Failure taxonomy (E1–E6)

`E1` gold chunk not retrieved → `E2` retrieved but cut off by top-k → `E3` evidence in context but answer wrong/incomplete → `E4` citation does not support the claim → `E5` refusal error → `E6` engineering error. `errors.csv` is distributed across these; each iteration fixes only the largest layer.

## Full results (16 arms, GPU)

Command: `python -m eval.runner --repeats 3 --workers 8 --out-dir results/full_gpu`
(30 questions × 3 repeats × 16 arms = 1,440 rows; RTX 4070 Laptop 8GB, fp32, total ~11 minutes)

### Main analysis arms (dimension 1)

| Arm | Completeness [95% CI] | Relevance | Faithfulness | Citation precision | Correct refusal | Wrong refusal | Latency p50 / p95 | Cost/question | Recall@6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A Vanilla LLM | 0.288 [0.182, 0.386] | 0.682 | — | — | 0% | 1.5% | 1.8s / 5.6s | ¥0.0021 | — |
| A' Structured LLM | 0.220 [0.114, 0.341] | 0.591 | — | — | 0% | 10.6% | 3.9s / 6.0s | ¥0.0044 | — |
| B Basic RAG | 0.765 [0.644, 0.871] | 0.955 | 0.944 | — | 100% | 4.5% | 1.4s / 3.6s | ¥0.0050 | 0.574 |
| C Optimized RAG | 0.758 [0.629, 0.879] | 0.939 | 0.964 | 0.968 | 100% | 7.6% | 3.7s / 5.9s | ¥0.0105 | 0.700 |
| D' Equal context top-18 | 0.894 [0.803, 0.962] | 1.000 | 0.966 | — | 100% | 0% | 1.5s / 5.9s | ¥0.0118 | 0.574 |

### Prompt arms (dimension 2)

| Arm | Completeness [95% CI] | Wrong refusal | Latency p50 | vs. baseline |
| --- | --- | --- | --- | --- |
| B-P1 (zero-shot CoT) | 0.780 [0.659, 0.886] | 6.1% | 2.3s | +0.015, p=0.571 |
| B-P2 (3-shot) | 0.735 [0.614, 0.849] | 4.5% | 1.3s | −0.030, p=0.312 |
| C-P1 (zero-shot CoT) | 0.765 [0.659, 0.871] | 9.1% | 4.3s | +0.008, p=0.908 |
| C-P2 (3-shot) | 0.720 [0.576, 0.856] | 18.2% | 3.5s | −0.038, p=0.415 |

### Architecture-paradigm arms (dimension 3)

| Arm | Completeness [95% CI] | Recall@6 | Evidence coverage | Wrong refusal | Latency p50 |
| --- | --- | --- | --- | --- | --- |
| D ReAct (≤4 rounds / 18 chunks) | 0.712 [0.614, 0.811] | 0.478 | 0.534 | 0% | 3.7s |
| D' Equal context (one-shot 18 chunks) | 0.894 [0.803, 0.962] | 0.574 | 0.740 | 0% | 1.5s |

D' → D: −0.182 [−0.265, −0.099], p &lt; 0.001.

### Ablation arms (attribution)

| Arm | Change | Completeness [95% CI] | Recall@6 | Wrong refusal | vs. baseline |
| --- | --- | --- | --- | --- | --- |
| C-1 | Structure-aware chunking | 0.727 [0.606, 0.841] | 0.674 | 6.1% | −0.038, p=0.438 |
| C-2 | + hybrid | 0.674 [0.515, 0.811] | 0.452 | 13.6% | −0.091, p=0.016 |
| C-3 | + rerank | 0.758 [0.636, 0.871] | 0.574 | 4.5% | −0.008, p=0.748 |
| C-4 | Five-part prompt discipline | 0.742 [0.614, 0.864] | 0.574 | 9.1% | −0.023, p=0.358 |
| C-q | Step-back 3+3 | 0.841 [0.750, 0.932] | 0.722 | 3.0% | +0.083, p=0.020 |
| C-c | Parent–child chunking | 0.599 [0.439, 0.750] | 0.595 | 21.2% | −0.159, p=0.004 |

### Conclusions

1. **Retrieval is the main effect**: closed-book A/A' completeness is 0.22–0.29, and all 8 unanswerable questions are answered anyway (correct-refusal rate 0%); with retrieval attached, 0.76–0.89 with 100% correct refusals. B→A is −0.477 [−0.614, −0.326]; B→A' is −0.545 [−0.689, −0.402]; p &lt; 0.001.
2. **The full optimization stack does not convert into answer quality**: C vs. B is −0.008 [−0.136, 0.121], p=0.905; but retrieval-level Recall@6 improves 0.574→0.700 and nDCG@10 0.512→0.591 — the gain stops at the retrieval layer.
3. **Ablation finds exactly one trick that works**: Step-back is the only significant positive effect (+0.083 [0.008, 0.197], p=0.020), and simultaneously pushes wrong refusals down to 3.0%. Hybrid alone is significantly worse (−0.091, p=0.016: Chinese questions get no lexical matches on the BM25 leg, which still consumes fusion slots); parent–child chunking is significantly worse (−0.159, p=0.004: 3 parent chunks under-cover, wrong-refusal rate 21.2%); structure-aware chunking, reranking, and prompt discipline are all non-significant.
4. **The agentic route is worse on this corpus**: D vs. D' is −0.182 [−0.265, −0.099], p &lt; 0.001. The cumulative recall of multi-round autonomous retrieval (evidence coverage 0.534) is actually lower than one-shot 18 chunks (0.740); the advantage lies not in "multi-round decision-making" but in "seeing more." D' is the highest-completeness arm of the entire experiment (0.894).
5. **Prompting tricks are non-significant overall**: CoT and 3-shot do not beat the baseline on either B or C; C-P2 even pushes wrong refusals to 18.2%.
6. **Cost and deployment implications**: on GPU, C's p50 is 3.7s (B: 1.4s), C-q 5.4s; all arms finish within 8s. Cost per question is ¥0.005–0.012. If the target is a single-building / small-institute corpus, **B + Step-back (or directly D' one-shot 18 chunks)** is more cost-effective than the full optimization stack.

Caveats: the judge and the judged model are both DeepSeek; completeness has not yet undergone the 25-sample human audit and kappa calibration, so conclusions should be treated as directional evidence. Sample: 30 questions, single run. Latency measured on RTX 4070 Laptop 8GB + fp32 and does not generalize to server GPUs. `results/phase1/` retains an earlier pure-CPU run of 5 arms (C p50 27.5s) for reference; the two latency figures are not directly comparable.

Cross-lingual validation (`results/variants-smoke3/`): one multi-hop variant each in English and German ran arm C, answering in English and German respectively; but both questions retrieved only the single paper by Ni et al., and once all 6 context slots were filled by that one paper the system refused — the same failure mechanism as Chinese q013, a limitation of multi-hop context budget rather than language.

Figures and single-file dashboard: `results/full_gpu/fig1_ladder.png`, `dashboard.html`.

## Evaluation and visualization

    python -m eval.visualize --run results/&lt;run&gt;

Generates 9 figures and `dashboard.html` (single file with embedded charts, summary tables, and real answers to the same questions from three arms).

Human calibration:

    # 1) Export 25 blind-scoring samples (reference answers / key points pre-filled)
    python -m eval.human export --run results/full_gpu --n 25

    # 2a) Web scoring (recommended; UI switchable among Chinese/English/German)
    python -m uvicorn eval.human_app:APP --port 8001
    # open http://127.0.0.1:8001
    # Blind scoring: arm names and judge scores hidden; saved per case; kappa and per-arm results shown only after all 25 are done

    # 2b) Or fill in the CSV directly
    # results/full_gpu/human_check.csv: human_completeness / human_relevance / human_refused

    # 3) Command-line verification (web scoring also writes back to this CSV)
    python -m eval.human score --csv results/full_gpu/human_check.csv

Outputs exact agreement, ±0.25 agreement rate, and quadratic-weighted kappa for the completeness / relevance / refusal groups. `human_check_key.csv` holds the row_id → arm/question mapping; do not open it while scoring.

**Content language**: the scoring page's top-right corner switches the interface among Chinese/English/German; questions, reference answers, key points, and the answers being judged switch accordingly. English/German content is pre-generated machine translation (`results/full_gpu/human_translations.json`); the page states explicitly "scoring is based on the Chinese original" and can expand the Chinese original for side-by-side comparison; entries missing a translation fall back to Chinese with a notice. To generate / backfill translations:

    python scripts/translate_content.py --run results/full_gpu

**Relevance convention**: the T7 judge occasionally returns 0.5; per the manual's 0/1 definition, reports and human calibration normalize `≥0.5 → 1`; raw values remain in `raw.jsonl`.

## Demo

    python -m uvicorn demo.app:APP --port 8000
    # or python -m demo.app

- Top-right corner switches the UI language: 中文 / English / Deutsch; labels, hints, sample questions, and buttons switch accordingly.
- Live Q&A exposes only arm C: structure-aware chunking + hybrid + rerank + citations/refusals; **the answer language follows the question language**, independent of the UI language.
- The second tab reads precomputed A/B/C answers from the most recent `raw.jsonl` for a zero-latency comparison; it always includes one unanswerable question.
- Precomputed comparison answers come from the Chinese question bank (zh); switching the UI to English/German shows the corresponding machine-translated versions, labeled "machine translation; original in Chinese"; cited source snippets remain in their original language.

## Tests

    python -m pytest tests -q

Coverage: all 16 arm configs and the "differ by exactly one variable" constraint; question-type quotas in the bank; all 71 gold-standard citations validated page by page; zero overlap between few-shot examples and the question bank; chunk-ID stability; parent–child mapping; citation parsing; bootstrap/kappa; refusal detection.

## Important design boundaries

- **BM25 is essentially ineffective for Chinese questions**: the corpus is English/German, so Chinese queries have no matches on the lexical leg; hybrid-retrieval gains under a Chinese corpus / Chinese questions come mainly from reranking. This is a known limitation of the pre-registered design; English/German variants are reported separately to observe hybrid's true effect.
- **A/A' are not authorized to refuse**: their refusals are spontaneous behaviors; arm C variants have "authorized refusal." Interpret refusal metrics grouped by arm.
- **C-c retrieval metrics are computed at the parent-chunk level**: child chunks are mapped back to parents before matching gold evidence; context evidence coverage is also reported. Chunk sizes differ, so raw Recall values are not directly comparable across strategies.
- **Judge and judged model share a source**: DeepSeek self-judging may inflate scores; the 25-sample human audit plus kappa calibration is mandatory before completeness can enter conclusions.
- **Latency is reported as median and p95 only**: to avoid individual hung requests contaminating the mean.