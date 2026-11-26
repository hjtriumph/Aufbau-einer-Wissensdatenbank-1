"""T1–T7 prompt templates (frozen for the pre-registered experiment).

Language rule: answers follow the language of the question. The benchmark is
authored in Chinese, with English/German variants evaluated separately.
"""

from __future__ import annotations

T1_SYSTEM = (
    "你是一个暖通工程知识库助手。请根据提供的上下文回答问题。"
    "回答语言必须与提问语言一致。"
)

# Closed-book variant for Arm A: mentioning "context" with no context supplied
# makes the model refuse on answerable questions, which would confound the
# retrieval effect with a prompt artefact.
T1_CLOSED_BOOK_SYSTEM = (
    "你是一个暖通工程知识库助手。请直接回答以下问题。"
    "回答语言必须与提问语言一致。"
)

T1_USER = """根据以下内容回答问题。

上下文：
{context}

问题：{question}

回答："""

T1_CLOSED_BOOK_USER = """请回答以下问题。

问题：{question}

回答："""

T2_SYSTEM = """你是暖通工程知识库助手。仅依据给定上下文回答，遵守以下规则：
1. 每个事实性声明必须标注来源，格式 [C#]。
2. 上下文不足以回答时，明确回答"根据现有资料无法回答"，禁止推测。
3. 涉及数值时必须原文引用，不得约算。
4. 回答语言与提问语言一致（支持中文/德语/英语提问）。

按以下结构输出：
【结论】一句话直接回答
【依据】支撑结论的原文要点，逐条带 [C#] 引用
【分析】证据与结论之间的推理过程
【不确定项】上下文中缺失或矛盾的部分
【来源】引用的论文标题 + 页码"""

T2_USER = """上下文：
{context}

问题：{question}

回答："""

T2_NO_CONTEXT_SYSTEM = """你是暖通工程知识库助手。请仅凭自身知识回答问题。
按以下结构输出：
【结论】一句话直接回答
【依据】支撑结论的要点
【分析】推理过程
【不确定项】你不确定或资料中可能缺失的部分
【来源】如引用具体研究，请注明论文标题

回答语言必须与提问语言一致。"""

COT_INSTRUCTION = "在给出最终答案前，先逐步分析给定证据，列出推理过程，再下结论。"

STEPBACK_SYSTEM = """你是问题抽象助手。给定一个具体问题，生成一个更上位、更抽象的问题，
用于检索支撑该问题背后的通用原理。只输出抽象后的问题本身。

示例：
具体问题：ED-LSTM 研究中热需求预测的输入特征有哪些？
抽象问题：深度学习做建筑热负荷预测时，输入特征一般如何选择？"""

STEPBACK_USER = "具体问题：{question}\n\n抽象问题："

REACT_SYSTEM = """你可以使用工具 search(query) 检索知识库。检索结果会以 [C#] 编号返回。
按以下规则循环：
- 如果还需要信息，输出一行：行动：search(查询词)
- 如果现有证据足够回答，输出：最终答案：<带 [C#] 引用的完整回答>
最多检索 4 次。回答语言与提问语言一致。最终答案必须只使用检索到的证据；
证据不足时明确说明"根据现有资料无法回答"。"""

REACT_STEP_USER = """问题：{question}

已检索信息：
{observations}

请决定下一步。只输出以下两种之一：
行动：search(查询词)
最终答案：<带 [C#] 引用的完整回答>"""

REACT_FORCE_FINAL = """问题：{question}

已检索信息：
{observations}

检索预算已用完，请立即给出最终答案，必须带 [C#] 引用；
若证据不足，明确回答"根据现有资料无法回答"。"""

# ---------------------------------------------------------------------------
# T4 few-shot examples.
#
# Deliberately drawn from three short German reports (Nahfeldexposition,
# Schulen_inhalierte_Dosis, hartmann_kriegel_risikobewertung). No benchmark
# question may target these documents; tests/test_leakage.py enforces that the
# example questions never appear in corpus/questions.jsonl.
# ---------------------------------------------------------------------------

_SIMPLE_EXAMPLES = [
    {
        "question": "在混合通风的室内，距离感染者 1.5 m 处吸入双倍气溶胶颗粒的概率是多少？",
        "context": """[C1] Nahfeldexposition respiratorischer Partikel anhand statistischer Auswertung um eine emittierende Person im Innenraum | page=1
Im Vergleich zur idealen Durchmischung zeigt sich bei Mischlüftung in einem Abstand von 1,5 m eine 25 prozentige Wahrscheinlichkeit, die doppelte Menge an Aerosolpartikeln einzuatmen als im Fernfeld.""",
        "answer": "【结论】约 25%。\n\n【依据】[C1] 指出，在混合通风条件下，距排放源 1.5 m 处吸入两倍于远场气溶胶颗粒量的概率为 25%。",
    },
    {
        "question": "在教室里，吸入量超过 RKI 密切接触者定义的人一定在感染者附近吗？",
        "context": """[C1] Vergleich der inhalierten Dosis von virenbeladenen Aerosolpartikeln in Klassenzimmern | page=1
Es zeigt sich, dass die Position der Personen, die an einem Schultag mehr potentiell virenbeladene Aerosolpartikel einatmen als enge Kontaktperson nach RKI-Definition, nicht zwingend im unmittelbaren Umfeld der infektiösen Person liegt.""",
        "answer": "【结论】不一定。\n\n【依据】[C1] 表明，在一个教学日内吸入量超过 RKI 密切接触者定义的人员，其位置不一定紧邻感染者。",
    },
    {
        "question": "Hartmann 与 Kriegel 如何用 CO2 浓度划分四种通风状况？",
        "context": """[C1] Beispielhafte Risikobewertung für verschiedene Alltagssituationen | page=2
• keine Lüftung – Fenster geschlossen
• mäßig gelüftet – mittlere CO2-Konzentration etwa 2000 ppm (15m³/(h·Per))
• gut gelüftet –mittlere CO2-Konzentration etwa 1200 ppm (30m³/(h·Per))
• sehr gut gelüftet - mittlere CO2-Konzentration etwa 1000 ppm (40m³/(h·Per))""",
        "answer": "【结论】四种状况分别对应不通风、约 2000 ppm、约 1200 ppm、约 1000 ppm 的平均 CO2 浓度。\n\n【依据】[C1] 列出：不通风（关窗）、适度通风约 2000 ppm（15 m³/(h·人)）、良好通风约 1200 ppm（30 m³/(h·人)）、很好通风约 1000 ppm（40 m³/(h·人)）。",
    },
]

_STRUCTURED_EXAMPLES = [
    {
        "question": "在混合通风的室内，距离感染者 1.5 m 处吸入双倍气溶胶颗粒的概率是多少？",
        "context": _SIMPLE_EXAMPLES[0]["context"],
        "answer": """【结论】约 25%。
【依据】
- [C1] 在混合通风、距排放源 1.5 m 处，吸入量为远场两倍的概率为 25%。
- [C1] 同一距离上，暴露超过假设值 5 倍的风险为 5%。
【分析】该概率来自对多组 CFD 模拟的空间统计，反映了理想混合假设与实际局部浓度之间的偏差；距离越近，偏离越大。
【不确定项】原文未给出 25% 对应的具体房间尺寸与换气次数。
【来源】Nahfeldexposition respiratorischer Partikel anhand statistischer Auswertung，第 1 页""",
    },
    {
        "question": "在教室里，吸入量超过 RKI 密切接触者定义的人一定在感染者附近吗？",
        "context": _SIMPLE_EXAMPLES[1]["context"],
        "answer": """【结论】不一定，高吸入量人员可能不在感染者紧邻位置。
【依据】
- [C1] 吸入量超过 RKI 密切接触者定义的人员，其位置并不必然位于感染者周边。
- [C1] 由于感染者在疫情暴发后（如果能确定）才被知晓，口罩与保持距离等措施应继续执行。
【分析】局部气流组织决定了气溶胶的空间分布，因此"靠近感染者"不等于"吸入量最高"；这削弱了仅靠距离判断风险的可靠性。
【不确定项】原文比较的是特定教室几何与通风方式，结论不能直接外推到所有房间。
【来源】Vergleich der inhalierten Dosis von virenbeladenen Aerosolpartikeln in Klassenzimmern，第 1 页""",
    },
    {
        "question": "Hartmann 与 Kriegel 如何用 CO2 浓度划分四种通风状况？",
        "context": _SIMPLE_EXAMPLES[2]["context"],
        "answer": """【结论】四种状况按平均 CO2 浓度划分：不通风、约 2000 ppm、约 1200 ppm、约 1000 ppm。
【依据】
- [C1] 不通风：窗户关闭。
- [C1] 适度通风：平均 CO2 约 2000 ppm，对应 15 m³/(h·人)。
- [C1] 良好通风：约 1200 ppm，对应 30 m³/(h·人)。
- [C1] 很好通风：约 1000 ppm，对应 40 m³/(h·人)。
【分析】作者用 CO2 浓度作为每人新风量的代理指标，从而把通风强度与感染风险计算联系起来。
【不确定项】所给 CO2 值仅适用于无回风运行的房间。
【来源】Beispielhafte Risikobewertung für verschiedene Alltagssituationen，第 2 页""",
    },
]

LANGUAGE_DIRECTIVE = {
    "en": "Answer language: English. Do not answer in Chinese.",
    "de": "Antwortsprache: Deutsch. Bitte nicht auf Chinesisch antworten.",
}


def language_directive(language: str) -> str:
    # Chinese questions already default to Chinese; only non-Chinese questions
    # need an explicit override, so the Chinese benchmark prompts stay frozen.
    return LANGUAGE_DIRECTIVE.get(language, "")


SIMPLE_EXAMPLES = _SIMPLE_EXAMPLES
STRUCTURED_EXAMPLES = _STRUCTURED_EXAMPLES


def simple_example_messages() -> list[tuple[str, str]]:
    return [(ex["question"], ex["answer"]) for ex in SIMPLE_EXAMPLES]


def structured_example_messages() -> list[tuple[str, str]]:
    return [(ex["question"], ex["answer"]) for ex in STRUCTURED_EXAMPLES]
