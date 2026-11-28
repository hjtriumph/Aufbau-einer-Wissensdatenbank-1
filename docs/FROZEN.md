# 预注册冻结清单

冻结日期：2026-11-28

本文件与 `frozen_manifest.json` 记录实验开始前冻结的题库、提示词、臂配置与评测代码哈希。
此后任何修改都必须记录理由与影响范围，并重跑全量基准。

## 题库

- 总题数：30
- 题型分布：{'numeric': 5, 'single_hop': 5, 'multi_hop': 6, 'method': 6, 'unanswerable': 8}
- 可答题：22，不可答题：8
- 金标准证据共 72 条，全部逐条校验为对应页的原文子串。
- few-shot 示例仅取自三篇德语短报告（Nahfeldexposition、Schulen_inhalierte_Dosis、
  hartmann_kriegel_risikobewertung），30 题中没有任何一题引用这三篇文档。

## 语言变体

- 5 道题 × 英/德等价版本 = 10 条，与中文主库共享金标准证据。
- 变体单独运行、按语言单独报告，不混入 30 题主指标。

## 文件哈希

| 文件 | sha256（前 16 位） | 字节 |
| --- | --- | --- |
| `corpus/questions.jsonl` | `cd93e77d35745244` | 41791 |
| `corpus/variants.jsonl` | `62a2312be6d4a22b` | 15890 |
| `corpus/documents.json` | `2e2b2be818822523` | 7084 |
| `prompts/templates.py` | `fe31083078013278` | 9757 |
| `arms/configs/A.yaml` | `93ee806bc502334c` | 352 |
| `arms/configs/A_prime.yaml` | `7acee775c18bed73` | 393 |
| `arms/configs/B.yaml` | `398cc06bd8687b0d` | 345 |
| `arms/configs/C.yaml` | `d15504ce9c32eaff` | 402 |
| `arms/configs/B-P1.yaml` | `fac8135e437bfc0b` | 353 |
| `arms/configs/B-P2.yaml` | `9c7d7cc01db0ad9d` | 374 |
| `arms/configs/C-P1.yaml` | `001a463bbcacdad2` | 385 |
| `arms/configs/C-P2.yaml` | `f1fadf7dbfe5f872` | 395 |
| `arms/configs/D.yaml` | `42db7a72b91d39e6` | 424 |
| `arms/configs/D_prime.yaml` | `102062405c98dc71` | 368 |
| `arms/configs/C-1.yaml` | `c27bb6a38e93874c` | 352 |
| `arms/configs/C-2.yaml` | `f3e509a129bd4a24` | 373 |
| `arms/configs/C-3.yaml` | `315e7ab8c4d6cad6` | 379 |
| `arms/configs/C-4.yaml` | `d1baaf0484c77619` | 357 |
| `arms/configs/C-q.yaml` | `f14d83f9fb61e808` | 424 |
| `arms/configs/C-c.yaml` | `aac035a43f2ecc23` | 435 |
| `arms/pipeline.py` | `af5ec066fc68697d` | 9572 |
| `arms/agent.py` | `7313edfd992e9c2d` | 6488 |
| `core/chunking.py` | `66c95b0d5412d09a` | 9986 |
| `core/parsing.py` | `29a1872209e8387f` | 13358 |
| `core/store.py` | `25107ebab8d59a68` | 12557 |
| `core/textnorm.py` | `4f377ffd87a0b49f` | 1721 |
| `core/citations.py` | `0baf007f5c69ffd2` | 4114 |
| `core/llm.py` | `36d052411e31b36c` | 1986 |
| `ingest/common.py` | `9a88484bd5eca0cc` | 2509 |
| `ingest/basic.py` | `85f2db5c05a0996a` | 1379 |
| `ingest/structured.py` | `e007177d7bf138a9` | 1603 |
| `ingest/parent_child.py` | `cd381a4b81f7403f` | 1968 |
| `eval/case_loader.py` | `0364ffd99bd1997a` | 5737 |
| `eval/human.py` | `7f9ccae84b1c1698` | 6647 |
| `eval/human_app.py` | `7a5bfb8bfa2a9930` | 9168 |
| `eval/human.html` | `095476d81600b7a4` | 28971 |
| `scripts/translate_content.py` | `190b7e28ac9446aa` | 7841 |
| `demo/app.py` | `6e4f6d2338b029e6` | 7115 |
| `demo/index.html` | `60563496608a3efa` | 19848 |
| `eval/judge.py` | `7494adc262249ef7` | 7402 |
| `eval/runner.py` | `f3b318645828d36b` | 29161 |
| `eval/metrics.py` | `091d83d1296b5e8d` | 5346 |

## 与手册的已记录偏差

1. ** D 臂步数语义**：D 最多 4 个 LLM 轮次，其中最多 3 次检索（每次 6 块），达到 18 块预算后
   强制进入作答；手册原文为“≤4 步、累计 ≤18 块”，未规定每步块数。此处按“步=检索行动、预算优先”
   实现，并在结果中记录每步检索量与累计块数。
2. **固定分块解析器**：手册写 PyPDFLoader；实现使用 pypdf 逐页抽取文本 + 递归切分，与 PyPDFLoader
   同源（PyPDFLoader 底层即 pypdf），保持页级元数据。
3. **结构感知解析器**：使用自研 PyMuPDF 解析（标题层级、表格原子、双栏阅读顺序、页眉页脚清洗），
   未引入第三方版面模型。
4. **重排模型**：三语需求使用 `bge-reranker-v2-m3`；未使用 `bge-reranker-base` 回退。
5. **BM25 分词**：拉丁词 + 德语变音折叠 + CJK 二元组；中文提问对英/德文语料的词法路基本无匹配，
   混合检索在中文主指标上的增益主要来自 rerank。该限制在 README 中明示，英/德变体单独报告。
6. **检索复用**：同一（检索配置，问题）只跑一次，3 次重复共用检索结果但均计入实测检索延迟；
   这只影响运行时间，不影响答案与指标。
7. **不可答题**：8 道为语料范围外的其他领域问题，正确行为是明确拒答；A/A' 的拒答属于自发拒答。
8. **拒答判定口径**：judge 看整条回答是否以拒答为主；五段式答案在【不确定项】里提到缺失/限制不算拒答。
   该口径在首轮冒烟后修正，并对第一阶段已生成的 450 条答案重判（未重新生成答案）。
9. **非中文语言指令**：英/德提问在用户消息末尾追加显式语言指令（Answer language: English /
   Antwortsprache: Deutsch）。中文主库的提示词逐字未变，因此第一阶段中文结果无需重跑。
10. **设备选择**：`DEVICE=auto` 时检测到 CUDA 即把嵌入与重排放在 GPU。全量 16 臂结果在
    RTX 4070 Laptop 8GB（fp32）上产生（results/full_gpu）；results/phase1 是早期 5 臂纯 CPU
    运行，仅作延迟对照，二者的延迟数字不可直接比较。答案质量与设备无关。
11. **相关性口径**：judge 在 76/1056 条可答行上返回了 0.5。按手册的 0/1 定义，报告与人工校准
    统一按 ≥0.5 → 1 归一；raw.jsonl 保留原始 judge 值。
12. **人工评分 UI**：新增 eval/human_app.py 与 eval/human.html；盲评（不传臂名/judge 分数）、
    逐条自动保存、全部完成后才显示一致性与 kappa，并同步回写 human_check.csv。
13. **内容翻译**：人工评分页与预制对比的英/德内容为 DeepSeek 预翻译（缓存于 results/<run>/
    *_translations.json），仅作阅读辅助；人工评分与结论以中文原文为准，翻译缺失时回退中文。

## 统计口径

- 每题 3 次重复先取均值，再在 22 道可答题上做配对 bootstrap（2000 次，95% CI）。
- 主维度基线 B；提示词维度分别以 B、C 为基线；范式维度以 D' 为基线；消融以 B 或 C 为基线。
- 延迟只报中位数与 p95；成本按输入 ¥2/M、输出 ¥8/M 估算，不含缓存折扣。
