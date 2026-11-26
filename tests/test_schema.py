from pathlib import Path

import pytest
import yaml

from arms.schema import ARM_ORDER, load_arms, parse_arm

CONFIG_DIR = Path(__file__).resolve().parents[1] / "arms" / "configs"


def test_loads_all_sixteen_arms():
    arms = load_arms(CONFIG_DIR)
    assert list(arms) == list(ARM_ORDER)
    assert len(arms) == 16


def test_arm_variants_differ_only_in_the_intended_component():
    arms = load_arms(CONFIG_DIR)
    # C-2 is B plus hybrid retrieval
    assert arms["C-2"].retrieval.strategy == arms["B"].retrieval.strategy
    assert arms["C-2"].retrieval.hybrid is True
    assert arms["C-2"].retrieval.rerank is False
    assert arms["C-2"].prompt.template == arms["B"].prompt.template
    # C-3 is B plus rerank
    assert arms["C-3"].retrieval.hybrid is False
    assert arms["C-3"].retrieval.rerank is True
    # C-4 is B plus structured prompt discipline, same retrieval as B
    assert arms["C-4"].retrieval.__dict__ == arms["B"].retrieval.__dict__
    assert arms["C-4"].prompt.template == "T2"
    assert arms["C-4"].output.citations is True


def test_agent_arm_uses_t5_and_budget():
    arms = load_arms(CONFIG_DIR)
    assert arms["D"].agent.enabled is True
    assert arms["D"].prompt.template == "T5"
    assert arms["D"].agent.max_steps == 4
    assert arms["D"].agent.max_chunks == 18


def test_rejects_citations_without_context():
    raw = yaml.safe_load((CONFIG_DIR / "A.yaml").read_text(encoding="utf-8"))
    raw["output"] = {"citations": True, "refusal": False}
    with pytest.raises(ValueError):
        parse_arm(raw)


def test_rejects_bad_dimension():
    raw = yaml.safe_load((CONFIG_DIR / "B.yaml").read_text(encoding="utf-8"))
    raw["dimension"] = "nonsense"
    with pytest.raises(ValueError):
        parse_arm(raw)
