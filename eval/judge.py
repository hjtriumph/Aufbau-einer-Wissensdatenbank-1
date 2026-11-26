"""T7 LLM judge: completeness / relevance / faithfulness / citation precision / refusal."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from core.citations import detect_refusal, parse_citation_labels
from core.config import settings
from core.llm import complete

from .case_loader import EvalCase

JUDGE_VERSION = "t7-v2"

JUDGE_SYSTEM = """你是严谨、公正的评测员。给定：问题、参考答案、关键要点、被评答案，以及可选的检索上下文与引用清单。

逐项评分：
1. 完整性 completeness：对照关键要点，被评答案覆盖了多少。全部覆盖=1，部分覆盖=0.5，几乎未覆盖=0。
2. 相关性 relevance：答案是否切题。切题=1，不切题=0。
3. 忠实度 faithfulness：把答案拆成事实性声明，逐条核对是否被检索上下文支持，
   给出支持比例（0~1）。没有上下文或答案拒答时，该项为 null。
4. 引用精度 citation_precision：逐个 [C#] 检查引用片段是否真的支持它所附着的声明，
   给出被支持引用占全部引用的比例（0~1）。该臂没有引用义务时为 null。
5. 拒答 refusal：被评答案是否属于拒答。只有当整条回答没有给出实质答案、以"无法回答/资料不足"为主时才算拒答；
   答案给出了实质结论、只是在【不确定项】里提到缺失或限制，不算拒答。

规则：
- 严格但公正；不因措辞差异扣分。
- 不奖励无依据的额外信息。
- 数值必须与参考答案一致才算覆盖。
- 参考答案与关键要点是中文，被评答案可能是中文/英文/德文，语义等价即可。
- 只输出 JSON，不要输出其他文字。"""

JUDGE_USER = """问题：
{question}

参考答案：
{reference_answer}

关键要点：
{key_points}

可答题：{answerable}
是否有检索上下文：{has_context}
是否有引用义务：{citation_required}

被评答案：
{answer}

检索上下文（截断）：
{context}

引用清单：
{citations}

只输出以下 JSON：
{{
  "completeness": 0.0,
  "relevance": 0.0,
  "faithfulness": null,
  "citation_precision": null,
  "refused": false,
  "reason": "一句话理由"
}}"""


def _cache_path(payload: str) -> Path:
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return settings.cache_dir / "judge" / f"{digest}.json"


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"judge returned no JSON: {text[:200]}")
    return json.loads(cleaned[start : end + 1])


def _binary_score(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return 1.0 if float(value) >= 0.5 else 0.0
    except (TypeError, ValueError):
        return None


def _coerce_score(value: Any, allowed: tuple[float, ...]) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return min(allowed, key=lambda x: abs(x - number))


def _citations_text(citations: list[dict[str, Any]]) -> str:
    if not citations:
        return "（无）"
    return "\n".join(
        f"[{c.get('label')}] {c.get('doc_title')} p.{c.get('page')} :: {str(c.get('snippet', ''))[:260]}"
        for c in citations
    )


def judge_row(case: EvalCase, row: dict[str, Any], citation_required: bool) -> dict[str, Any]:
    answer = str(row.get("answer", ""))
    context = str(row.get("context", ""))
    citations = row.get("citations", []) or []
    refused_det = detect_refusal(answer)

    if not answer.strip():
        return {
            "completeness": None if not case.answerable else 0.0,
            "relevance": 0.0 if case.answerable else None,
            "faithfulness": None,
            "citation_precision": None,
            "refused": True,
            "refused_det": True,
            "reason": "empty answer",
            "judge_error": "",
        }

    cited_labels = set(parse_citation_labels(answer))
    cited_citations = [c for c in citations if c.get("label") in cited_labels] if citation_required else []
    has_citation_obligation = bool(citation_required and citations)
    no_citation_given = has_citation_obligation and not cited_labels

    user = JUDGE_USER.format(
        question=case.question,
        reference_answer=case.reference_answer,
        key_points="\n".join(f"- {k}" for k in case.key_points),
        answerable=case.answerable,
        has_context=bool(context),
        citation_required=has_citation_obligation,
        answer=answer[:6000],
        context=context[:12000] if context else "（无）",
        citations=_citations_text(cited_citations) if citation_required else "（本臂不作引用要求）",
    )
    payload = (
        JUDGE_SYSTEM
        + user
        + settings.judge_model
        + str(settings.judge_temperature)
        + JUDGE_VERSION
    )

    cache_file = _cache_path(payload)
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    try:
        text, _usage, _latency = complete(
            [("system", JUDGE_SYSTEM), ("human", user)], judge=True
        )
        parsed = _extract_json(text)
    except Exception as exc:
        return {
            "completeness": None if not case.answerable else 0.0,
            "relevance": None,
            "faithfulness": None,
            "citation_precision": None,
            "refused": refused_det,
            "refused_det": refused_det,
            "reason": "",
            "judge_error": f"{type(exc).__name__}: {exc}",
        }

    result = {
        "completeness": (
            _coerce_score(parsed.get("completeness"), (0.0, 0.5, 1.0))
            if case.answerable
            else None
        ),
        "relevance": _binary_score(parsed.get("relevance")) if case.answerable else None,
        "faithfulness": (
            None
            if not context or refused_det
            else _coerce_score(parsed.get("faithfulness"), (0.0, 0.25, 0.5, 0.75, 1.0))
        ),
        "citation_precision": (
            None
            if not has_citation_obligation or refused_det
            else (
                0.0
                if no_citation_given
                else _coerce_score(parsed.get("citation_precision"), (0.0, 0.25, 0.5, 0.75, 1.0))
            )
        ),
        "refused": bool(parsed.get("refused", refused_det)),
        "refused_det": refused_det,
        "reason": str(parsed.get("reason", ""))[:400],
        "judge_error": "",
    }

    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        tmp.replace(cache_file)
    except Exception:
        pass
    return result
