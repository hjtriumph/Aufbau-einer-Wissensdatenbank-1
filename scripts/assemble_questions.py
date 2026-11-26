"""Assemble the frozen 30-question benchmark from the drafted candidates.

Run once, then the benchmark is frozen: corpus/questions.jsonl.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"


def load_drafts() -> dict[str, dict]:
    drafts: dict[str, dict] = {}
    for path in sorted(CORPUS.glob("draft_q*.json")):
        for item in json.loads(path.read_text(encoding="utf-8")):
            drafts[item["id"]] = item
    return drafts


def clean(item: dict, qid: str, qtype: str) -> dict:
    return {
        "id": qid,
        "question": item["question"],
        "type": qtype,
        "reference_answer": item["reference_answer"],
        "key_points": item["key_points"],
        "gold_evidence": item["gold_evidence"],
        "answerable": True,
        "language": "zh",
        "notes": item.get("notes", ""),
    }


DATA2BEM = {
    "question": "Data2BEM 在剑桥大学办公楼案例中，相对专业实践把总建模时间降低了多少？具体耗时从多少降到多少？",
    "type": "numeric",
    "reference_answer": "Data2BEM 相对专业实践把总建模时间降低超过 90%：从专业实践的 8–32 小时降至约 48 分钟，且几乎不需要人工干预。",
    "key_points": ["降低超过90%", "从8–32小时", "降至约48分钟", "最少人工输入"],
    "gold_evidence": [
        {
            "doc": "Automated building energy modeling for energy.pdf",
            "page": 2,
            "quote": "reduced total modeling time by over 90% (48 min versus 8–32 h)",
        }
    ],
    "answerable": True,
    "language": "zh",
    "notes": "时间收益只在摘要给出，正文分阶段耗时容易混淆。",
}

MOE_FEATURES = {
    "question": "An Explainable Transformer-based MoE 热负荷预测框架使用哪些输入特征？三个专家分别对应哪类负荷状态？",
    "type": "single_hop",
    "reference_answer": "输入特征为室外温度、历史热负荷时间序列，以及月份与小时的正弦/余弦四个时间编码。K-means 依据历史能耗把建筑聚为低、中、高三种负荷状态，每种状态由一个轻量 Informer 编码器专家独立训练。",
    "key_points": ["室外温度", "历史热负荷序列", "月份与小时正余弦编码", "低中高三种负荷状态", "每个状态一个Informer专家"],
    "gold_evidence": [
        {
            "doc": "An_Explainable_Transformer-based_Mixture_of_Expert.pdf",
            "page": 6,
            "quote": "We use the outside temperature (T), a historical time series of the heat load",
        },
        {
            "doc": "An_Explainable_Transformer-based_Mixture_of_Expert.pdf",
            "page": 6,
            "quote": "The three experts (low, medium, and high)",
        },
        {
            "doc": "An_Explainable_Transformer-based_Mixture_of_Expert.pdf",
            "page": 2,
            "quote": "clustered into heat load regimes (low, medium, high) using K-means",
        },
    ],
    "answerable": True,
    "language": "zh",
    "notes": "特征清单与专家分工分处方法节两段，需完整召回。",
}

OCCUPANCY_MULTI = {
    "question": "两篇研究都评估了 occupancy 特征，但结论侧重点不同：Neubauer 等（Relationship between feature importance…）指出 occupancy 的 RFFI 被严重低估，而酒店研究（Influence of occupancy…）指出时间特征已编码入住模式。请解释这两个结论分别说明了什么，以及它们为何并不矛盾。",
    "type": "multi_hop",
    "reference_answer": "前者说明 RFFI 基于不纯度下降、偏好高基数特征；occupancy 只有 0/1 两个取值（低基数），其 RFFI 均值仅约 0.04，因此应改用 PFI 衡量。后者说明加入 occupancy 的收益取决于模型是否已含小时、星期、年周等时间特征：含时间特征时收益很小，去掉时间特征后收益更大（如 Hotel 1 客房翼 NRMSE 降约 3.8%），且高 PFI 不必然带来精度提升。两者不矛盾：一个讲重要性度量方法对低基数特征的偏置，一个讲 occupancy 信息与时间特征的冗余，二者共同说明 occupancy 的价值是条件性的。",
    "key_points": ["RFFI偏好高基数特征", "occupancy仅0/1取值", "低基数导致RFFI低估", "时间特征编码入住模式", "高PFI不必然提升精度"],
    "gold_evidence": [
        {
            "doc": "Relationship betweenfeature importanceandbuildingcharacteristics for.pdf",
            "page": 9,
            "quote": "occupancy has only two values (low cardinality)",
        },
        {
            "doc": "Relationship betweenfeature importanceandbuildingcharacteristics for.pdf",
            "page": 5,
            "quote": "the magnitude of the feature attributions, while the PFI is based on",
        },
        {
            "doc": "Influence of occupancy data on heating and.pdf",
            "page": 6,
            "quote": "time features already encode many occupancy-related patterns",
        },
        {
            "doc": "Influence of occupancy data on heating and.pdf",
            "page": 5,
            "quote": "ranging from 3.0 % to 14.7 %",
        },
    ],
    "answerable": True,
    "language": "zh",
    "notes": "两文的“重要性”口径不同（度量方法 vs 预测收益），容易混为一谈。",
}

EXPLAIN_MULTI = {
    "question": "两篇供热负荷预测研究都强调可解释性，但可解释路径不同：Neubauer 等的多步预测研究如何用 SHAP 与注意力解释模型？Babakhani 等的 MoE 研究如何解释专家分工？",
    "type": "multi_hop",
    "reference_answer": "Neubauer 等用 Deep SHAP 把预测递归分解到各输入特征与滞后小时，并用注意力图说明最近时间步最重要；Babakhani 等用门控权重量化三个专家对最终预测的贡献，并在专家内部检查注意力模式与特征重要性来解释时间关注点和输入影响。前者解释“输入特征与时间步的贡献”，后者解释“专家分工与门控分配”。",
    "key_points": ["Deep SHAP分解到特征与滞后小时", "注意力图显示最近时间步重要", "门控权重量化专家贡献", "专家内部检查注意力与特征重要性", "两者解释对象不同"],
    "gold_evidence": [
        {
            "doc": "Explainable multi-step heating load forecasting Using SHAP values and.pdf",
            "page": 2,
            "quote": "By mapping temporal attention, it was possible to demonstrate the importance of the most recent time steps",
        },
        {
            "doc": "Explainable multi-step heating load forecasting Using SHAP values and.pdf",
            "page": 8,
            "quote": "recursively propagating the Deep LIFT multipliers",
        },
        {
            "doc": "An_Explainable_Transformer-based_Mixture_of_Expert.pdf",
            "page": 2,
            "quote": "Explainability is provided not only at the MoE level",
        },
        {
            "doc": "An_Explainable_Transformer-based_Mixture_of_Expert.pdf",
            "page": 6,
            "quote": "The weights provide explainability by quantifying each",
        },
    ],
    "answerable": True,
    "language": "zh",
    "notes": "两文可解释对象的层级不同，容易笼统答成“都用 SHAP”。",
}

UNANSWERABLE = [
    ("q023", "桥梁抗震设计中，钢筋混凝土桥墩的位移延性系数通常取多少？依据哪本规范？", "结构抗震"),
    ("q024", "退役光伏组件的回收经济性如何？每吨组件的净回收收益是多少？", "光伏回收"),
    ("q025", "数据中心液冷系统的典型 PUE 是多少？与风冷相比能降低多少？", "数据中心"),
    ("q026", "氢燃料电池公交车的加氢站建设成本与运营成本分别是多少？", "氢能交通"),
    ("q027", "城市污水处理厂曝气系统的单位电耗通常是多少 kWh/m³？", "给排水"),
    ("q028", "电梯再生制动能量回收系统的节能率一般是多少？", "电梯"),
    ("q029", "制冷剂 R290 的全球变暖潜值（GWP）和安全等级分别是多少？", "制冷剂标准"),
    ("q030", "欧盟 2024 年修订的建筑能效指令（EPBD）要求成员国在何时完成公共建筑存量改造？", "建筑政策"),
]


def main() -> None:
    drafts = load_drafts()
    order = [
        ("d1-01", "numeric"),
        ("d2-01", "numeric"),
        ("d3-01", "numeric"),
        ("d4-01", "numeric"),
        ("d1-02", "single_hop"),
        ("d2-02", "single_hop"),
        ("d3-02", "single_hop"),
        ("d4-02", "single_hop"),
        ("d1-03", "multi_hop"),
        ("d2-03", "multi_hop"),
        ("d3-03", "multi_hop"),
        ("d4-03", "multi_hop"),
        ("d1-05", "method"),
        ("d2-04", "method"),
        ("d3-04", "method"),
        ("d3-05", "method"),
        ("d4-05", "method"),
        ("d4-06", "method"),
    ]
    base = []
    for item_id, qtype in order:
        item = clean(drafts[item_id], "", qtype)
        if item_id == "d2-02":
            item["gold_evidence"][0]["quote"] = (
                "This is due to the storage and release of energy in the"
            )
        base.append(item)

    final = []
    final += base[0:4] + [DATA2BEM]
    final += base[4:8] + [MOE_FEATURES]
    final += base[8:12] + [OCCUPANCY_MULTI, EXPLAIN_MULTI]
    final += base[12:18]

    for index, item in enumerate(final, start=1):
        item["id"] = f"q{index:03d}"

    for qid, question, domain in UNANSWERABLE:
        final.append(
            {
                "id": qid,
                "question": question,
                "type": "unanswerable",
                "reference_answer": (
                    f"该问题属于{domain}领域，超出本语料库（暖通/建筑能源/室内空气）的范围，"
                    "语料中没有相关依据；正确行为是明确拒答，而不是凭常识作答。"
                ),
                "key_points": [],
                "gold_evidence": [],
                "answerable": False,
                "language": "zh",
                "notes": f"不可答题：{domain}，语料中无对应内容。",
            }
        )

    path = CORPUS / "questions.jsonl"
    path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in final) + "\n",
        encoding="utf-8",
    )
    print(f"[questions] wrote {len(final)} -> {path}")
    print(Counter(item["type"] for item in final))


if __name__ == "__main__":
    main()
