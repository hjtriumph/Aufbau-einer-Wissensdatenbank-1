from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ARM_ORDER = (
    "A",
    "A'",
    "B",
    "C",
    "B-P1",
    "B-P2",
    "C-P1",
    "C-P2",
    "D",
    "D'",
    "C-1",
    "C-2",
    "C-3",
    "C-4",
    "C-q",
    "C-c",
)

DIMENSIONS = ("main", "prompt", "paradigm", "ablation")
STRATEGIES = ("fixed", "structure", "parent_child")


@dataclass
class RetrievalConfig:
    mode: str = "dense"  # none | dense | hybrid
    strategy: str = "fixed"
    k: int = 6
    fetch_k: int = 20
    hybrid: bool = False
    rerank: bool = False
    rerank_n: int = 20
    max_parents: int = 0


@dataclass
class QueryConfig:
    transform: str = "none"  # none | stepback
    stepback_k: int = 3


@dataclass
class PromptConfig:
    template: str = "T1"  # T1 | T2 | T5
    cot: bool = False
    fewshot: int = 0


@dataclass
class OutputConfig:
    citations: bool = False
    refusal: bool = False


@dataclass
class AgentConfig:
    enabled: bool = False
    max_steps: int = 4
    max_chunks: int = 18
    per_step_k: int = 6


@dataclass
class ArmConfig:
    id: str
    name: str
    dimension: str
    baseline: str | None
    answer: str = ""
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    query: QueryConfig = field(default_factory=QueryConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

    @property
    def uses_context(self) -> bool:
        return self.retrieval.mode != "none"

    @property
    def retrieval_label(self) -> str:
        if self.retrieval.mode == "none":
            return "无检索"
        parts = [f"{self.retrieval.strategy}"]
        if self.retrieval.hybrid:
            parts.append("hybrid")
        if self.retrieval.rerank:
            parts.append("rerank")
        parts.append(f"top-{self.retrieval.k}")
        return " + ".join(parts)


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key) or {}
    if not isinstance(value, dict):
        raise ValueError(f"arm {raw.get('id')}: section {key} must be a mapping")
    return value


def parse_arm(raw: dict[str, Any]) -> ArmConfig:
    arm_id = str(raw.get("id", "")).strip()
    if not arm_id:
        raise ValueError("arm is missing id")

    dimension = str(raw.get("dimension", ""))
    if dimension not in DIMENSIONS:
        raise ValueError(f"arm {arm_id}: bad dimension {dimension}")

    retrieval = RetrievalConfig(**_section(raw, "retrieval"))
    if retrieval.mode not in {"none", "dense", "hybrid"}:
        raise ValueError(f"arm {arm_id}: bad retrieval.mode {retrieval.mode}")
    if retrieval.strategy not in STRATEGIES:
        raise ValueError(f"arm {arm_id}: bad retrieval.strategy {retrieval.strategy}")

    arm = ArmConfig(
        id=arm_id,
        name=str(raw.get("name", arm_id)),
        dimension=dimension,
        baseline=raw.get("baseline"),
        answer=str(raw.get("answer", "")),
        retrieval=retrieval,
        query=QueryConfig(**_section(raw, "query")),
        prompt=PromptConfig(**_section(raw, "prompt")),
        output=OutputConfig(**_section(raw, "output")),
        agent=AgentConfig(**_section(raw, "agent")),
    )

    if arm.query.transform not in {"none", "stepback"}:
        raise ValueError(f"arm {arm_id}: bad query.transform")
    if arm.prompt.template not in {"T1", "T2", "T5"}:
        raise ValueError(f"arm {arm_id}: bad prompt.template")
    if arm.retrieval.mode == "none" and arm.prompt.fewshot:
        raise ValueError(f"arm {arm_id}: few-shot without retrieval is not a defined arm")
    if arm.retrieval.mode == "none" and arm.output.citations:
        raise ValueError(f"arm {arm_id}: citations without context is not a defined arm")
    if arm.agent.enabled and arm.query.transform != "none":
        raise ValueError(f"arm {arm_id}: agent and query transform are separate arms")
    if arm.agent.enabled and arm.prompt.template != "T5":
        raise ValueError(f"arm {arm_id}: agent arms must use the T5 template")
    if arm.prompt.template == "T5" and not arm.agent.enabled:
        raise ValueError(f"arm {arm_id}: T5 is reserved for agent arms")
    return arm


def load_arm(path: str | Path) -> ArmConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: arm config must be a mapping")
    return parse_arm(raw)


def load_arms(config_dir: str | Path) -> dict[str, ArmConfig]:
    arms: dict[str, ArmConfig] = {}
    for path in sorted(Path(config_dir).glob("*.yaml")):
        arm = load_arm(path)
        if arm.id in arms:
            raise ValueError(f"duplicate arm id: {arm.id}")
        arms[arm.id] = arm

    missing = [arm_id for arm_id in ARM_ORDER if arm_id not in arms]
    if missing:
        raise ValueError(f"missing arm configs: {missing}")
    return {arm_id: arms[arm_id] for arm_id in ARM_ORDER}
