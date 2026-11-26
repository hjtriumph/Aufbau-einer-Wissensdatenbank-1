# 手册 → 代码映射

本文件把《RAG 对比实验完整实现方案手册》逐章落到代码，便于答辩时指出"这一条在哪里实现"。

| 手册章节 | 实现位置 |
| --- | --- |
| 三.1 语料与元数据 | `corpus/raw/`、`corpus/documents.json`（标题/作者/语言/DOI） |
| 三.2 固定递归分块 | `ingest/basic.py`、`core/chunking.py::fixed_page_chunks` |
| 三.2 结构感知分块 | `ingest/structured.py`、`core/parsing.py`、`core/chunking.py::structure_chunks` |
| 三.2 父子切割 | `ingest/parent_child.py`、`core/chunking.py::parent_child_chunks` |
| 三.3 稳定块 ID / 幂等入库 | `core/chunking.py::stable_chunk_id`、`core/store.py::build_collection` |
| 三.3 中文路径 HNSW 坑 | `core/store.py::HNSW_METADATA` |
| 三.4 三语支持 | `core/store.py::tokenize`（德语变音折叠）、`RERANK_MODEL=bge-reranker-v2-m3`、T2 语言条款 |
| 四 题库与冻结 | `corpus/questions.jsonl`、`corpus/variants.jsonl`、`scripts/freeze.py`、`docs/FROZEN.md` |
| 五.1 四个主臂 | `arms/configs/{A,A_prime,B,C}.yaml`、`arms/pipeline.py` |
| 五.2 提示词臂 | `arms/configs/{B-P1,B-P2,C-P1,C-P2}.yaml`、`prompts/templates.py`（T3/T4） |
| 五.3 范式臂 | `arms/configs/{D,D_prime}.yaml`、`arms/agent.py` |
| 五.4 消融臂 | `arms/configs/C-{1,2,3,4,q,c}.yaml` |
| 六 T1–T7 | `prompts/templates.py`（T1–T6）、`eval/judge.py`（T7） |
| 七.1 生成质量与行为指标 | `eval/judge.py`、`eval/runner.py::summarize_arm` |
| 七.2 检索指标 | `eval/runner.py`（Recall/MRR/nDCG/证据覆盖率）、`eval/metrics.py` |
| 七.3 统计与校准 | `eval/metrics.py::paired_bootstrap_diff`、`eval/human.py` |
| 七.5 回归门禁 | `tests/`、`results/<run>/run_meta.json` 的配置快照 |
| 七.6 失败分类 E1–E6 | `eval/runner.py::classify_error`、`results/<run>/errors.csv` |
| 七.7 迭代闭环 | 同一 runner + 题库冻结；`FROZEN.md` 记录每次变更理由 |
| 八 可视化交付 | `eval/visualize.py` → 9 张图 + `dashboard.html` |
| 九 前端演示 | `demo/app.py`、`demo/index.html` |
| 十 工程结构 | `rag_lab/` 目录 |

## 16 臂速查

| 臂 | 检索 | 分块 | 提示 | 输出 | 维度 |
| --- | --- | --- | --- | --- | --- |
| A | 无 | — | T1 | 直接回答 | 主 |
| A' | 无 | — | T2 无上下文 | 五段式 | 主 |
| B | 向量 top-6 | 固定 | T1 | 直接回答 | 主 |
| C | hybrid + rerank top-6 | 结构 | T2 | 引用 + 拒答 | 主 |
| B-P1 | 同 B | 固定 | T1 + CoT | 直接回答 | 提示词 |
| B-P2 | 同 B | 固定 | T1 + 3-shot | 直接回答 | 提示词 |
| C-P1 | 同 C | 结构 | T2 + CoT | 引用 + 拒答 | 提示词 |
| C-P2 | 同 C | 结构 | T2 + 3-shot | 引用 + 拒答 | 提示词 |
| D | 多轮 ≤4 轮 / 18 块 | 固定 | T5 ReAct | 引用 | 范式 |
| D' | 向量 top-18 | 固定 | T1 | 直接回答 | 范式 |
| C-1 | 同 B | 结构 | T1 | 直接回答 | 消融 |
| C-2 | 向量 + BM25 | 固定 | T1 | 直接回答 | 消融 |
| C-3 | 向量 + rerank | 固定 | T1 | 直接回答 | 消融 |
| C-4 | 同 B | 固定 | T2 | 引用 + 拒答 | 消融 |
| C-q | 同 C + Step-back 3+3 | 结构 | T2 | 引用 + 拒答 | 消融 |
| C-c | 子块 hybrid + rerank → 父块 | 父子 | T2 | 引用 + 拒答 | 消融 |

## 指标定义（实现口径）

- **完整性**：judge 对照参考答案与 key_points，给 0 / 0.5 / 1；每题 3 次重复先取均值。
- **相关性**：judge 给 0 / 1。
- **忠实度**：judge 将答案拆成声明，逐条核对检索上下文，给支持比例；无上下文臂为 N/A。
- **引用精度**：只核对答案里实际出现的 `[C#]`；未给引用但有引用义务记 0。
- **正确拒答率**：8 道不可答题中判为拒答的比例。
- **错误拒答率**：22 道可答题中判为拒答的比例。
- **检索指标**：金标准证据解析到当前索引的块 ID 后计算 Recall@3/6、MRR@10、nDCG@10；C-c 在父块层计算。
- **证据覆盖率**：进上下文的块对金标准证据的覆盖比例。
- **成本**：输入 ¥2/M + 输出 ¥8/M，不含缓存折扣。
- **延迟**：整条链路（检索 + 生成）的中位数与 p95；3 次重复均计入实测检索耗时。
