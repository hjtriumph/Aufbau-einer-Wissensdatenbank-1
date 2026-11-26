# RAG 对比实验：纯 LLM vs 基础 RAG vs 优化 RAG

> 操作手册（Word / PDF）：`docs/操作手册.docx`、`docs/操作手册.pdf`，覆盖环境、入库、前端启动、实验、人工打分（含练手模式）、导出与排错。

在一个约 20 篇论文的小型垂直语料库上，用受控实验回答一个前置问题：**精细优化的 RAG 相对基础 RAG 到底带来多少增量，代价是什么，哪些技巧真正有效？**

- 语料：20 篇暖通 / 建筑能源 / 室内空气论文（约 409 页，英/德混合）
- 16 个实验臂，3 个正交维度 + 消融归因
- 30 题冻结题库（22 可答 + 8 不可答），另有 5 题 × 英/德变体
- 所有臂共用一条流水线，差异只走 `arms/configs/*.yaml`
- 评测包含检索指标、LLM judge、配对 bootstrap、人工 kappa 校准与失败分类学

## 目录

```text
rag_lab/
├── corpus/
│   ├── raw/                 20 篇 PDF（只读）
│   ├── text/                逐页纯文本（出题与引文校验用）
│   ├── documents.json       文献元数据（标题/作者/语言/出版信息）
│   ├── questions.jsonl      30 题冻结题库（含金标准证据）
│   └── variants.jsonl       5 题 × 英/德等价版本
├── core/                    解析、分块、存储、检索、LLM、引用
├── ingest/                  三套索引：basic / structured / parent_child
├── arms/
│   ├── configs/             16 个臂的 YAML（唯一差异来源）
│   ├── schema.py            配置校验
│   ├── pipeline.py          共享流水线：检索→融合→重排→拼装→生成
│   └── agent.py             ReAct 循环（仅 D 臂）
├── prompts/templates.py     T1–T7 模板（冻结）
├── eval/
│   ├── runner.py            实验调度 → results/<run>/{raw,summary,by_type,paired}.csv
│   ├── judge.py             T7 LLM 评判（带磁盘缓存）
│   ├── metrics.py           Recall/MRR/nDCG、bootstrap、加权 kappa
│   ├── case_loader.py       题库加载与证据→块 ID 解析
│   ├── human.py             25 条人工抽查导出与一致性计算
│   └── visualize.py         9 张图 + 单文件 dashboard.html
├── demo/                    FastAPI + 单页前端（C 实时问答 + A/B/C 预制对比）
├── tests/                   单元与集成测试
└── results/                 运行输出
```

## 环境

已验证环境：`chain_1_fixed`（Python 3.12, torch CPU, chromadb 1.5.8）。

```bash
conda activate chain_1_fixed
pip install -r requirements.txt
cp .env.example .env   # 填入 DEEPSEEK_API_KEY
```

**GPU（推荐）**：默认 `DEVICE=auto`，检测到 CUDA 就把 bge-m3 与重排模型放到 GPU。RTX 4070 Laptop 8GB 实测两者同时驻留约 5 GB，20 候选重排约 0.7s（CPU 约 38s）。

```bash
pip install "torch==2.11.0+cu128" --index-url https://download.pytorch.org/whl/cu128
# 强制 CPU 时在 .env 设 DEVICE=cpu
```

默认模型与锁死项：

| 项 | 取值 |
| --- | --- |
| 生成模型 | `deepseek-chat`，temperature 0.2（全臂一致） |
| 嵌入 | `BAAI/bge-m3`（多语言，中/英/德跨语言检索） |
| 重排 | `BAAI/bge-reranker-v2-m3`（三语必需的组件级改动） |
| 融合 | RRF，BM25 : 向量 = 0.5 : 0.5，k=60 |
| 主臂上下文 | top-6；D/D' 预算 18 块 |
| 评判 | `deepseek-chat`，temperature 0 |

`bge-m3` 与 `bge-reranker-v2-m3` 首次运行会从 HF 镜像下载（约 2 GB + 2.3 GB）。

## 建库

```bash
python -m ingest.basic --reset          # 固定 1000/150 递归分块
python -m ingest.structured --reset     # 结构感知分块（标题起块、表格原子）
python -m ingest.parent_child --reset   # 1800 父块 + 250 子块
# 或一次建三套：
python scripts/build_all_indexes.py --reset
```

入库报告写入 `storage/ingest_report_<strategy>.json`（文档数、块数、重复 ID）。块 ID 由内容 + 页 + 位置哈希生成，重复入库幂等；若出现重复 ID 会直接报错，而不是让 Chroma 静默丢数据。

**Windows 中文路径坑**：Chroma 1.5.x 在非 ASCII 路径且越过默认 `hnsw:sync_threshold=1000` 时可能静默丢失 HNSW 段。本项目所有 collection 以 `sync_threshold=10000`、`batch_size=1000` 创建。

## 运行实验

```bash
# 冒烟：1 可答 + 1 不可答 × A/B/C/D × 1 次
python -m eval.runner --smoke

# 第一阶段：4 主臂 + D'
python -m eval.runner --arms A "A'" B C "D'" --workers 8

# 全量 16 臂（30 题 × 3 次 = 1,440 行）
python -m eval.runner --workers 8

# 英/德变体（单独运行，按语言单独报告）
python -m eval.runner --include-variants --workers 8
```

输出目录 `results/YYYYMMDD-HHMMSS/`：

| 文件 | 内容 |
| --- | --- |
| `raw.jsonl` | 每行一条：答案、上下文、检索块 ID、候选池、token、延迟、judge 分数、`error_class` |
| `summary.csv` | 每臂一行：完整性/相关性/忠实度/引用精度/拒答/延迟/成本/检索指标 |
| `by_type.csv` | 臂 × 题型 |
| `by_language.csv` | 臂 × 语言（中文主报告，英/德单独成组） |
| `paired.csv` | 各臂 vs 基线的配对 bootstrap（均值差、95% CI、p 值） |
| `errors.csv` | 错题与失败类别 |
| `run_meta.json` | 模型、温度、题库哈希、全部臂配置（预注册证据） |

检索是确定性的：每个（检索配置，问题）只跑一次，3 次重复共用结果，但每条重复都计入实测检索延迟。差异配置（C / C-P1 / C-P2）共享同一次检索，避免在 CPU 上重复跑重排。

### 失败分类（E1–E6）

`E1` 金标块未被召回 → `E2` 召回了但被 top-k 截断 → `E3` 证据进了上下文但答案错/漏 → `E4` 引用不支持声明 → `E5` 拒答失误 → `E6` 工程错误。`errors.csv` 按此分布，下一轮只修占比最大的那一层。

## 全量结果（16 臂，GPU）

命令：`python -m eval.runner --repeats 3 --workers 8 --out-dir results/full_gpu`
（30 题 × 3 次 × 16 臂 = 1,440 行；RTX 4070 Laptop 8GB，fp32，总耗时 11 分钟）

### 主分析臂（维度一）

| 臂 | 完整性 [95% CI] | 相关性 | 忠实度 | 引用精度 | 正确拒答 | 错误拒答 | 延迟 p50 / p95 | 成本/题 | Recall@6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A 纯 LLM | 0.288 [0.182, 0.386] | 0.682 | — | — | 0% | 1.5% | 1.8s / 5.6s | ¥0.0021 | — |
| A' 结构化 LLM | 0.220 [0.114, 0.341] | 0.591 | — | — | 0% | 10.6% | 3.9s / 6.0s | ¥0.0044 | — |
| B 基础 RAG | 0.765 [0.644, 0.871] | 0.955 | 0.944 | — | 100% | 4.5% | 1.4s / 3.6s | ¥0.0050 | 0.574 |
| C 优化 RAG | 0.758 [0.629, 0.879] | 0.939 | 0.964 | 0.968 | 100% | 7.6% | 3.7s / 5.9s | ¥0.0105 | 0.700 |
| D' 等上下文 top-18 | 0.894 [0.803, 0.962] | 1.000 | 0.966 | — | 100% | 0% | 1.5s / 5.9s | ¥0.0118 | 0.574 |

### 提示词臂（维度二）

| 臂 | 完整性 [95% CI] | 错误拒答 | 延迟 p50 | 相对基线 |
| --- | --- | --- | --- | --- |
| B-P1（zero-shot CoT） | 0.780 [0.659, 0.886] | 6.1% | 2.3s | +0.015，p=0.571 |
| B-P2（3-shot） | 0.735 [0.614, 0.849] | 4.5% | 1.3s | −0.030，p=0.312 |
| C-P1（zero-shot CoT） | 0.765 [0.659, 0.871] | 9.1% | 4.3s | +0.008，p=0.908 |
| C-P2（3-shot） | 0.720 [0.576, 0.856] | 18.2% | 3.5s | −0.038，p=0.415 |

### 架构范式臂（维度三）

| 臂 | 完整性 [95% CI] | Recall@6 | 证据覆盖率 | 错误拒答 | 延迟 p50 |
| --- | --- | --- | --- | --- | --- |
| D ReAct（≤4 轮 / 18 块） | 0.712 [0.614, 0.811] | 0.478 | 0.534 | 0% | 3.7s |
| D' 等上下文（一次性 18 块） | 0.894 [0.803, 0.962] | 0.574 | 0.740 | 0% | 1.5s |

D' → D：−0.182 [−0.265, −0.099]，p < 0.001。

### 消融臂（归因）

| 臂 | 改动 | 完整性 [95% CI] | Recall@6 | 错误拒答 | 相对基线 |
| --- | --- | --- | --- | --- | --- |
| C-1 | 结构感知分块 | 0.727 [0.606, 0.841] | 0.674 | 6.1% | −0.038，p=0.438 |
| C-2 | + hybrid | 0.674 [0.515, 0.811] | 0.452 | 13.6% | −0.091，p=0.016 |
| C-3 | + rerank | 0.758 [0.636, 0.871] | 0.574 | 4.5% | −0.008，p=0.748 |
| C-4 | 五段式提示纪律 | 0.742 [0.614, 0.864] | 0.574 | 9.1% | −0.023，p=0.358 |
| C-q | Step-back 3+3 | 0.841 [0.750, 0.932] | 0.722 | 3.0% | +0.083，p=0.020 |
| C-c | 父子切割 | 0.599 [0.439, 0.750] | 0.595 | 21.2% | −0.159，p=0.004 |

### 结论

1. **检索是主效应**：闭卷 A/A' 完整性 0.22–0.29，8 道不可答题全部硬答（正确拒答 0%）；接入检索后 0.76–0.89，正确拒答 100%。B→A 为 −0.477 [−0.614, −0.326]，B→A' 为 −0.545 [−0.689, −0.402]，p < 0.001。
2. **完整优化栈没有转化为答案质量**：C 相对 B 为 −0.008 [−0.136, 0.121]，p=0.905；但检索层 Recall@6 0.574→0.700、nDCG@10 0.512→0.591，增益停在检索层。
3. **消融只找到一个有效技巧**：Step-back 是唯一显著正收益（+0.083 [0.008, 0.197]，p=0.020），同时把错误拒答压到 3.0%。hybrid 单独使用显著变差（−0.091，p=0.016，中文提问下 BM25 词法腿无匹配，还挤占了融合名额）；父子切割显著变差（−0.159，p=0.004，3 个父块覆盖不足、错误拒答 21.2%）；结构分块、重排、提示词纪律均不显著。
4. **Agentic 路线在该语料上更差**：D 相对 D' 为 −0.182 [−0.265, −0.099]，p < 0.001。多轮自主检索的累计召回（证据覆盖率 0.534）反而低于一次性 18 块（0.740）；优势不在“多轮决策”，而在“看得更多”。D' 是全实验完整性最高的臂（0.894）。
5. **提示词技巧整体不显著**：CoT 与 3-shot 在 B/C 上都没有超过基线；C-P2 还把错误拒答推到 18.2%。
6. **代价与部署含义**：GPU 上 C 的 p50 为 3.7s（B 1.4s），C-q 5.4s，全部臂都在 8s 内；单题成本 ¥0.005–0.012。若目标是单栋建筑/小型研究所语料库，**B + Step-back（或直接 D' 的 18 块一次性上下文）** 比完整优化栈更划算。

注意事项：judge 与被评模型同为 DeepSeek，完整性尚未经过 25 条人工抽查与 kappa 校准，结论应视为方向性证据；样本 30 题、单次运行；延迟为 RTX 4070 Laptop 8GB + fp32 的结果，不等同于服务器 GPU。`results/phase1/` 保留了早期 5 臂纯 CPU 运行的对照（C p50 27.5s），两者延迟不可直接比较。

跨语言验证（`results/variants-smoke3/`）：英/德各一道多跳变体跑 C 臂，回答分别用英语/德语；但两题都只召回 Ni 等一篇论文，6 个上下文槽被单篇占满后拒答——与中文 q013 同一失败机制，限制在多跳上下文预算而非语言。

图表与单文件看板：`results/full_gpu/fig1_ladder.png`、`dashboard.html`。

## 评测与可视化

```bash
python -m eval.visualize --run results/<run>
```

生成 9 张图与 `dashboard.html`（单文件，内嵌图表、汇总表、三臂同题真实回答）。

人工校准：

```bash
# 1) 导出 25 条盲评样本（参考答案/关键要点已填入）
python -m eval.human export --run results/full_gpu --n 25

# 2a) 网页打分（推荐，界面可切换中/英/德）
python -m uvicorn eval.human_app:APP --port 8001
# 打开 http://127.0.0.1:8001
# 盲评：不显示臂名与 judge 分数；逐条自动保存；25 条打完才显示 kappa 与分臂结果

# 2b) 或直接填 CSV
# results/full_gpu/human_check.csv 的 human_completeness / human_relevance / human_refused

# 3) 命令行复核（网页打分也会同步写回这份 CSV）
python -m eval.human score --csv results/full_gpu/human_check.csv
```

输出完整性/相关性/拒答三组 exact agreement、±0.25 一致率、二次加权 kappa。`human_check_key.csv` 保存 row_id → 臂/题号的映射，打分时不要打开。

**内容语言**：打分页右上角切换中/英/德时，题目、参考答案、关键要点与被评答案也会切换。英/德内容是预生成的机器翻译（`results/full_gpu/human_translations.json`），页面明确标注"评分以中文原文为准"，并可展开中文原文对照；翻译缺失的条目会回退中文并提示。生成/补译：

```bash
python scripts/translate_content.py --run results/full_gpu
```

**相关性口径**：T7 judge 偶尔返回 0.5；按手册的 0/1 定义，报告与人工校准统一按 `≥0.5 → 1` 归一，原始值保留在 `raw.jsonl`。

## Demo

```bash
python -m uvicorn demo.app:APP --port 8000
# 或 python -m demo.app
```

- 页面右上角可切换界面语言：中文 / English / Deutsch；标签、提示、示例问题与按钮同步切换。
- 实时只开放 C 臂：结构分块 + hybrid + rerank + 引用/拒答；**回答语言跟随提问语言**，与界面语言相互独立。
- 第二个标签页从最近一次 `raw.jsonl` 读取 A/B/C 预缓存回答，零延迟对比；固定包含一道不可答题。
- 预制对比回答来自中文题库（zh）；切到英/德界面时会显示对应的机器翻译版本，并标注"机器翻译、原文为中文"；源文引用片段保持原语言。

## 测试

```bash
python -m pytest tests -q
```

覆盖：16 臂配置与"只差一个变量"约束、题库题型配额、71 条金标准引文逐条对页校验、few-shot 与题库零重叠、块 ID 稳定性、父子映射、引用解析、bootstrap/kappa、拒答判定。

## 重要设计边界

- **BM25 对中文提问基本失效**：语料为英/德文，中文查询在词法路上没有匹配，混合检索的增益在中国语料/中文提问下主要来自 rerank。这是预注册设计的已知限制；英/德变体单独报告，用于观察 hybrid 的真实作用。
- **A/A' 不授权拒答**：它们的拒答是自发行为；C 系是"被授权的拒答"，解读拒答指标时按臂分组。
- **C-c 的检索指标在父块层计算**：子块映射回父块后再对金标证据，同时报告上下文证据覆盖率；块大小不同，Recall 原始值不可与其他策略直接横比。
- **judge 与被评模型同源**：DeepSeek 自评可能偏高，必须用 25 条人工抽查 + kappa 校准后才能把完整性写进结论。
- **延迟只报中位数与 p95**：避免个别挂起请求污染均值。
