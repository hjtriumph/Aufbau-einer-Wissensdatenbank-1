---
title: "RAG 对比实验 · 操作手册"
subtitle: "入库 / 启动 / 实验运行 / 人工抽查打分 / 导出与统计"
author: "暖通工程知识库项目"
date: "2026-11-26"
lang: zh-CN
toc-title: "目录"
---

# 0. 这份手册解决什么

本手册面向实际操作，覆盖：

- 从 20 篇 PDF 重建三套向量索引；
- 启动问答 Demo 与人工打分网页；
- 预检、冒烟、第一阶段与全量 16 臂实验；
- 人工抽查打分（含"先练手、不污染正式结果"的做法）；
- 导出评分、计算 kappa、生成图表与 dashboard；
- 中/英/德内容翻译缓存的生成与更新；
- 常见故障排查与命令速查。

实验设计、16 臂定义、预注册与已知限制见 `README.md` 与 `docs/FROZEN.md`，本手册不重复。

**默认工作目录**：

```text
D:\mir\招聘准备\柏林工业大学论文项目-全新\rag_lab
```

以下命令均在该目录下执行。Windows 推荐使用 Git Bash（Claude Code 内置终端）；PowerShell 的差异在对应位置单独标注。

# 1. 五分钟快速开始

| 步骤 | 命令 | 结果 |
| --- | --- | --- |
| 1. 激活环境 | `conda activate chain_1_fixed` | Python 3.12 + 依赖 |
| 2. 启动问答 | `python -m uvicorn demo.app:APP --port 8000` | http://127.0.0.1:8000 |
| 3. 启动打分 | `python -m uvicorn eval.human_app:APP --port 8001` | http://127.0.0.1:8001 |
| 4. 重新入库（可选） | `python scripts/build_all_indexes.py --reset` | 三套索引与报告 |

当前 `results/full_gpu` 已有一次完整的 16 臂运行，索引也已建好；如果只是演示和打分，**不需要重新入库或重跑实验**，直接执行第 2、3 步即可。

# 2. 环境与目录

## 2.1 环境

```bash
conda activate chain_1_fixed
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

预期（当前机器）：

```text
2.11.0+cu128 True
```

- `.env` 必须有有效的 `DEEPSEEK_API_KEY`；没有则所有生成、judge、翻译都会失败。
- `DEVICE=auto`：检测到 CUDA 就把 bge-m3 与重排模型放到 GPU；强制 CPU 改为 `DEVICE=cpu`。
- 依赖清单：`requirements.txt`；GPU 版 torch 安装命令见 README。
- 首次运行会加载模型：bge-m3 约 2.3 GB 显存，重排模型约 2.3 GB，合计约 5 GB；8 GB 显存可同时驻留。

## 2.2 目录

| 路径 | 内容 |
| --- | --- |
| `corpus/raw/` | 20 篇原始 PDF（只读，勿改） |
| `corpus/questions.jsonl` | 30 题冻结题库（含金标准证据） |
| `corpus/variants.jsonl` | 5 题 × 英/德变体 |
| `corpus/documents.json` | 文献标题/作者/语言/DOI |
| `core/` `ingest/` `arms/` `eval/` | 代码 |
| `prompts/templates.py` | T1–T7 提示词（冻结） |
| `storage/chroma/` | Chroma 向量库 |
| `storage/parsed/` | PDF 解析缓存 |
| `storage/parents/` | 父块 JSON |
| `storage/cache/judge/` | judge 结果缓存 |
| `results/full_gpu/` | 当前完整 16 臂运行 |
| `demo/` | 问答 Demo |
| `eval/human_app.py` + `eval/human.html` | 人工打分网页 |
| `docs/` | 预注册清单、本手册 |

## 2.3 从零恢复环境（换电脑 / 重装）

如果 `chain_1_fixed` 不存在或依赖损坏，按下面步骤恢复。已有可用环境时跳过。

1. 安装 Anaconda 或 Miniconda（Windows 64 位）。
2. 创建并激活环境（环境名可自取，下文以 `chain_1_fixed` 为例）：

   ```bash
   conda create -n chain_1_fixed python=3.12 -y
   conda activate chain_1_fixed
   cd "D:/mir/招聘准备/柏林工业大学论文项目-全新/rag_lab"
   pip install -r requirements.txt
   ```

3. 安装 GPU 版 PyTorch（推荐；纯 CPU 可跳过，并在 `.env` 设 `DEVICE=cpu`）：

   ```bash
   pip install "torch==2.11.0+cu128" --index-url https://download.pytorch.org/whl/cu128
   ```

4. 配置 API 与设备：

   ```bash
   cp .env.example .env
   # 编辑 .env，填入 DEEPSEEK_API_KEY；DEVICE=auto
   ```

5. 首次运行会自动下载模型：`BAAI/bge-m3`（约 2 GB）与 `BAAI/bge-reranker-v2-m3`（约 2.3 GB）。国内镜像已通过环境变量 `HF_ENDPOINT=https://hf-mirror.com` 配置；如未生效，可在运行前设置：

   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   # PowerShell: $env:HF_ENDPOINT = "https://hf-mirror.com"
   ```

6. 验证：

   ```bash
   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   python scripts/build_all_indexes.py --reset
   python scripts/preflight.py
   ```

# 3. 入库（建三套索引）

## 3.1 一键建三套

```bash
python scripts/build_all_indexes.py --reset
```

`--reset` 会删除并重建对应 collection。**这是破坏性操作**：已有索引会被清空后重算，但不会影响 `results/` 里的历史结果。

## 3.2 分步建

```bash
python -m ingest.basic --reset          # 固定 1000/150 递归分块
python -m ingest.structured --reset     # 结构感知分块（标题起块、表格原子）
python -m ingest.parent_child --reset   # 1800 父块 + 250 子块
```

只重建其中一套时，用对应的单条命令。

## 3.3 预期结果（当前语料）

| 索引 | collection | 数量 | 报告 |
| --- | --- | --- | --- |
| 固定分块 | `raglab_fixed` | 1,730 块 | `storage/ingest_report_fixed.json` |
| 结构感知 | `raglab_structure` | 1,066 块 | `storage/ingest_report_structure.json` |
| 父子切割 | `raglab_children` | 6,765 子块 + 978 父块 | `storage/ingest_report_parent_child.json` |

每份报告应满足：

- `document_count = 20`
- `chunk_count = expected_chunk_count`
- `duplicate_ids = 0`

## 3.4 入库后验证

```bash
python scripts/preflight.py
```

预期输出要点：

```text
collection fixed: 1730 units
collection structure: 1066 units
collection parent_child: 6765 units
parents: 978
unresolved evidence: none
dense top-20 recall on gold (fixed): 22/22
dense top-20 recall on gold (structure): 22/22
preflight OK
```

如果 `resolved evidence` 出现未解析项，或召回不是 22/22，先检查 `corpus/questions.jsonl` 的引文是否仍能在 `corpus/text/` 对应页找到；题库已冻结，不应随意改动。

## 3.5 重建后注意

- 题库、提示词、分块参数改变后，必须重建对应索引并重跑实验。
- Chroma 在 Windows 中文路径下已用 `hnsw:sync_threshold=10000` 规避静默丢段；不要手动改这个参数。
- 入库耗时：GPU 上数分钟；纯 CPU 会慢一个数量级，建议先用 `DEVICE=cpu` 验证再决定。

# 4. 启动前端页面（Demo 与人工打分）

项目有两个独立的网页服务，可以同时开：

| 页面 | 端口 | 启动命令 | 是否需要 GPU |
| --- | --- | --- | --- |
| 问答 Demo | 8000 | `python -m uvicorn demo.app:APP --port 8000` | 首次提问时加载嵌入/重排模型 |
| 人工打分 | 8001 | `python -m uvicorn eval.human_app:APP --port 8001` | 不需要 |

两个页面都支持右上角切换 **中文 / English / Deutsch**。

## 4.1 启动问答 Demo（8000）

### 启动与打开

```bash
conda activate chain_1_fixed
cd "D:/mir/招聘准备/柏林工业大学论文项目-全新/rag_lab"
python -m uvicorn demo.app:APP --port 8000
```

看到 `Uvicorn running on http://127.0.0.1:8000` 后，浏览器打开：

```text
http://127.0.0.1:8000
```

终端窗口要保持运行；关闭终端即停止服务。停止服务也可在终端按 `Ctrl+C`。

### 页面结构

| 区域 | 内容 | 操作 |
| --- | --- | --- |
| 顶部 | 标题 + 语言切换（中文 / English / Deutsch） | 点语言按钮切换界面与内容 |
| 标签页 | 实时问答（C 臂） / 三臂对比（预制） | 点击切换功能 |
| 输入区 | 多行输入框 + “提问”按钮 | `Ctrl+Enter` 或点按钮发送 |
| 示例问题 | 随界面语言变化的 5 个问题 | 点击直接填入并提问 |
| 回答区 | 五段式答案 + `[C#]` 引用卡片 | 阅读、核对引用页码与片段 |
| 状态徽标 | 臂名、响应时间、检索缓存、tokens/成本、拒答 | 判断是否命中缓存、是否拒答 |

### 实时问答操作

1. 在输入框用中文、英语或德语提问；`Ctrl+Enter` 或点"提问"发送。
2. 首次提问会加载模型，约 **40–45 秒**；同一问题再次提问命中检索缓存，约 **3 秒**。
3. 回答按五段式输出；每条事实性声明带 `[C#]` 引用，下方是引用卡片（论文标题 + 页码 + 原文片段）。
4. 证据不足时会拒答（`拒答` 徽标）；不可答题可以直接用来演示这一点。
5. 点示例问题会直接填入并发送；示例问题随界面语言变化（中/英/德混合）。

### 三臂对比操作

1. 点第二个标签页"三臂对比（预制）"。
2. 页面读取最近一次运行的 `raw.jsonl`，挑数值题、多跳题、不可答题各一道，三栏并列展示 A（纯 LLM）/ B（基础 RAG）/ C（优化 RAG）的真实回答。
3. 这是预缓存结果：零延迟、零额外 API 调用；可以用来讲"闭卷硬答 vs 检索后拒答"的对比。
4. 英/德界面显示机器翻译版本，并标注原文为中文；某组没有翻译时会标"原文（中文）"。

### 界面语言 vs 回答语言

- 界面语言切换：标签、提示、示例问题、对比区列名与内容。
- **实时回答语言跟随提问语言**：界面切到 Deutsch 但输入中文，回答仍是中文；点德语示例问题，回答是德语。
- 这是有意设计，符合"回答语言与提问语言一致"的实验规则。

### 演示脚本建议（3 分钟）

1. 打开实时问答，点"降低热泵供水温度对运行成本有什么影响？"——展示五段式 + 引用卡片 + 响应时间。
2. 切到"三臂对比"，展示同一道不可答题：A 硬答、B/C 拒答。
3. 切到 Deutsch，点德语示例，展示界面与回答语言同时切换。
4. 回到实时问答，问一个语料外问题（如"桥梁抗震的位移延性系数是多少？"），展示拒答。

## 4.2 启动人工打分页（8001）

### 启动与打开

```bash
conda activate chain_1_fixed
cd "D:/mir/招聘准备/柏林工业大学论文项目-全新/rag_lab"
python -m uvicorn eval.human_app:APP --port 8001
```

浏览器打开 http://127.0.0.1:8001。

服务启动时会打印读取的运行目录，例如：

```text
[human] run dir: D:\mir\...\rag_lab\results\full_gpu
```

页面顶部会显示运行名与进度；当前应为 `full_gpu`、`0 / 25`。

### 页面结构

| 区域 | 内容 | 操作 |
| --- | --- | --- |
| 顶部 | 运行名、进度条、盲评标记、“查看校准结果” | 25/25 后按钮才可用 |
| 左侧 | 1–25 的格子 | 点击跳转；绿色 = 已评分 |
| 卡片顶部 | 当前条号、已评分/未评分 | — |
| 内容区 | 问题、参考答案与关键要点、被评答案 | 阅读后打分 |
| 评分区 | 完整性 0/0.5/1、相关性 0/1、拒答 0/1 | 快捷键 `1/2/3` 选完整性 |
| 底部 | 上一条、下一条、清除本条、保存状态 | `← →` 导航；自动保存 |


### 操作步骤

1. 确认顶部运行名是 `full_gpu`（不是 `_practice`）。
2. 逐条阅读：问题 → 参考答案与关键要点 → 被评答案。
3. 选择完整性 `1 / 0.5 / 0`（快捷键 `1 / 2 / 3`）、相关性 `0/1`、是否拒答 `0/1`。
4. 备注可写判断依据；`Ctrl+Enter` 立即保存。
5. 左侧格子变绿 = 该条已评分；点击格子可跳转；`← →` 切换条目。
6. 25/25 后点"查看校准结果"：显示三组 exact agreement、±0.25 一致率、加权 kappa，以及按臂/按题型汇总。
7. 结果页右上角"下载评分 CSV"导出。

### 盲评与自动保存

- 页面和接口都不返回臂名与 judge 分数；提前请求结果接口会返回 409。
- 每次选择或备注输入都会自动保存到 `results/full_gpu/human_scores.json`，并同步回写 `human_check.csv`。
- 中途关闭浏览器不影响已保存分数；重新打开会从第一条未评分继续。
- 不要把 `human_check_key.csv` 打开放在旁边，否则盲评失效。

### 分数是怎么算的

- 完整性：1 = 要点全部覆盖且数字正确；0.5 = 覆盖约一半/有实质遗漏；0 = 基本没答到/答错。
- 相关性：1 = 正面回答；0 = 答非所问。
- 拒答：1 = 整条是拒答/无实质答案；0 = 有实质结论（【不确定项】里的缺失说明不算拒答）。
- kappa 是人工分与 judge 分的二次加权一致性，不是"答案正确率"。

## 4.3 端口、后台运行与停止

### 停止

在运行服务的终端按 `Ctrl+C`。

### 查端口占用

Git Bash：

```bash
netstat -ano | findstr :8000
```

PowerShell：

```powershell
Get-NetTCPConnection -LocalPort 8000 | Select-Object OwningProcess
```

拿到 PID 后可结束进程：

```bash
taskkill //PID <PID> //F
```

### 换端口

```bash
python -m uvicorn demo.app:APP --port 8003
python -m uvicorn eval.human_app:APP --port 8004
```

### 后台运行（可选）

Git Bash：

```bash
python -m uvicorn demo.app:APP --port 8000 > storage/demo.log 2>&1 &
```

关闭终端会结束进程；需要长期运行请用 Windows 任务计划或 `start` 新开窗口。

### 让局域网其他设备访问（谨慎）

```bash
python -m uvicorn demo.app:APP --host 0.0.0.0 --port 8000
```

需要放行 Windows 防火墙端口；只在可信网络内使用，Demo 没有鉴权。

## 4.4 前端自检（不用浏览器）

```bash
# Demo 元数据
curl http://127.0.0.1:8000/api/meta

# 对比数据（当前运行、3 组案例）
curl "http://127.0.0.1:8000/api/comparison?lang=zh"

# 打分页元数据（进度与运行名）
curl http://127.0.0.1:8001/api/human/meta

# 打分页条目（应不含 judge/arm 字段）
curl http://127.0.0.1:8001/api/human/items
```

如果 `curl` 不通，先确认服务是否启动、端口是否一致、是否被其他进程占用。

# 5. 跑实验

## 5.1 预检与冒烟

```bash
python scripts/preflight.py
python -m eval.runner --smoke
```

冒烟是 1 道可答题 + 1 道不可答题 × A/B/C/D × 1 次，用来确认 API、检索、重排、judge 全链路可用。冒烟会花钱（约几分钱），但比全量失败便宜。

## 5.2 第一阶段（5 臂）

```bash
python -m eval.runner --arms A "A'" B C "D'" --repeats 3 --workers 8 --out-dir results/phase1
```

- 5 臂 × 30 题 × 3 次 = 450 行。
- `results/phase1/` 是早期纯 CPU 对照；如需新的 GPU 版本，换一个新的输出目录。

## 5.3 全量 16 臂

```bash
python -m eval.runner --repeats 3 --workers 8 --out-dir results/full_gpu
```

- 16 臂 × 30 题 × 3 次 = 1,440 行。
- 当前机器（RTX 4070 Laptop 8GB）实测约 11 分钟；生成成本约 ¥11.23（judge 另计，未写入 `usage`）。
- 输出：`raw.jsonl`、`summary.csv`、`by_type.csv`、`by_language.csv`、`paired.csv`、`errors.csv`、`report.md`、`run_meta.json`。

### 断点续跑

用**同一个** `--out-dir` 再执行一次即可：

```bash
python -m eval.runner --repeats 3 --workers 8 --out-dir results/full_gpu
```

已完成的 `(arm, case, repeat)` 会跳过；只补未完成行。换硬件（CPU ↔ GPU）时**不要**续跑同一个目录，延迟口径会混。

### 子集与变体

```bash
# 只跑指定臂
python -m eval.runner --arms C C-q --workers 4 --out-dir results/check_c

# 只跑前 5 题
python -m eval.runner --limit 5 --out-dir results/smoke5

# 英/德变体（单独报告，不混入 30 题主指标）
python -m eval.runner --include-variants --workers 8 --out-dir results/variants
```

### 重判（只换 judge，不重新生成）

```bash
python -m eval.runner --rejudge --out-dir results/full_gpu --workers 8
```

适用于修改了 judge 提示词或拒答口径、但不想重新调用生成模型的场景。会重写 `raw.jsonl` 的 judge 字段和汇总表。

## 5.4 图表与 dashboard

```bash
python -m eval.visualize --run results/full_gpu
```

生成 9 张图 + 单文件 `dashboard.html`，直接双击用浏览器打开。图表包括阶梯主图、提示词、范式、消融、热力图、质量—延迟—成本、拒答双指标、引用精度、配对 CI。

`results/phase1/` 的 CPU 结果仍保留，作为延迟对照；由于是 CPU，C 臂 p50 约 27 秒，不能和 GPU 结果直接比较。

## 5.5 GPU / CPU 口径

- 延迟指标依赖硬件。同一份结果目录里不要混 CPU 与 GPU 行。
- 换机器后，建议新建 `results/<run>_<device>` 目录重跑。
- `DEVICE=cpu` 可强制 CPU；8 GB 显存不够时也可先用 CPU 验证流程。

# 6. 人工抽查打分

## 6.1 正式打分流程

1. 启动打分页：

   ```bash
   python -m uvicorn eval.human_app:APP --port 8001
   ```

2. 打开 http://127.0.0.1:8001，确认顶部运行名为 `full_gpu`、进度 `0 / 25`。
3. 逐条阅读题目、参考答案与关键要点、被评答案，选择：
   - 完整性：`1 / 0.5 / 0`（快捷键 `1 / 2 / 3`）；
   - 相关性：`1 / 0`；
   - 是否拒答：`1 / 0`。
4. 备注可写“数字对但漏了适用条件”“引用编号有误”等；`Ctrl+Enter` 立即保存。
5. 左侧格子变绿表示该条已评分；`← →` 切换条目。
6. 25/25 后点击"查看校准结果"，显示完整性/相关性/拒答三组：
   - exact agreement；
   - ±0.25 一致率；
   - 二次加权 kappa；
   - 按臂、按题型的描述性汇总（n 很小，只看方向）。
7. 打分自动保存到 `results/full_gpu/human_scores.json`，并同步回写 `results/full_gpu/human_check.csv`。

## 6.2 练手模式（不污染正式结果）

核心：**复制一份运行目录，用 `HUMAN_RUN_DIR` 指向副本，换一个端口**。副本内的 `human_scores.json` 和 `human_check.csv` 与正式目录完全隔离。

Git Bash：

```bash
mkdir -p results/_practice
cp results/full_gpu/human_check.csv results/_practice/
cp results/full_gpu/human_check_key.csv results/_practice/
cp results/full_gpu/human_translations.json results/_practice/
HUMAN_RUN_DIR="$(pwd)/results/_practice" python -m uvicorn eval.human_app:APP --port 8002
```

PowerShell：

```powershell
New-Item -ItemType Directory -Force results\_practice | Out-Null
Copy-Item results\full_gpu\human_check.csv results\_practice\
Copy-Item results\full_gpu\human_check_key.csv results\_practice\
Copy-Item results\full_gpu\human_translations.json results\_practice\
$env:HUMAN_RUN_DIR = (Resolve-Path results\_practice).Path
python -m uvicorn eval.human_app:APP --port 8002
```

打开 http://127.0.0.1:8002，确认顶部运行名为 `_practice`。练手结束后：

```bash
# 停止 8002 服务后
rm -rf results/_practice
```

`_practice` 目录删除即彻底丢弃练手分数；正式 `results/full_gpu/` 不会被修改。

注意：

- 如果正式打分页已占用 8001，练手页请用 8002/8003；
- 复制 `human_translations.json` 才能看到英/德内容；不复制则英/德界面会回退中文并提示；
- 练手时不要打开 `human_check_key.csv`，否则盲评失效。

## 6.3 评分标准

| 指标 | 取值 | 判定 |
| --- | --- | --- |
| 完整性 | 1 | 参考答案的要点全部覆盖，数字正确 |
| 完整性 | 0.5 | 覆盖约一半，或有实质遗漏/小错 |
| 完整性 | 0 | 基本没答到、答错、空答 |
| 相关性 | 1 / 0 | 正面回答该问题 / 答非所问 |
| 拒答 | 1 / 0 | 整条是拒答、无实质答案 / 有实质结论 |

五段式答案在【不确定项】里写“缺失/无法确定”，但正文有结论，`human_refused = 0`。

## 6.4 完成后的 kappa

页面直接显示；也可用命令行复核：

```bash
python -m eval.human score --csv results/full_gpu/human_check.csv
```

输出示例：

```json
{
  "completeness": {"n": 25, "exact_agreement": 0.72, "within_025": 0.96, "weighted_kappa": 0.68},
  "relevance": {"n": 25, "exact_agreement": 0.92, "within_025": 1.0, "weighted_kappa": 0.71},
  "refusal": {"n": 25, "exact_agreement": 0.96, "within_025": 1.0, "weighted_kappa": 0.83}
}
```

| kappa | 解释 | 使用建议 |
| --- | --- | --- |
| ≥ 0.8 | 几乎完全一致 | judge 分数可直接用于结论 |
| 0.6–0.8 | 高度一致 | 可用于结论，报告注明 kappa |
| 0.4–0.6 | 中等 | 只作相对排序，不写绝对分数 |
| < 0.4 | 一致性差 | 不用 judge 完整性下结论；只报人工子集或改 judge 后重判 |

n=25 时置信区间较宽，报告写“kappa=0.68（n=25）”，不要写成精确到小数点后两位的强结论。

## 6.5 导出

- 网页结果页右上角"下载评分 CSV"；
- 或直接取 `results/full_gpu/human_check.csv`（UTF-8 BOM，Excel 可直接打开）；
- 原始逐条分数在 `results/full_gpu/human_scores.json`；
- 盲评映射在 `results/full_gpu/human_check_key.csv`（row_id → 臂/题号），**打分完成后**再打开。

## 6.6 重置与恢复

- **重打某一条**：页面点"清除本条"，再重新选择。
- **重打全部**：停止 8001 服务，删除 `results/full_gpu/human_scores.json`，再重新导出空白评分表（同一 seed 会得到同一批 25 条）：

  ```bash
  python -m eval.human export --run results/full_gpu --n 25 --seed 42
  ```

  这会清空所有人工分与备注，**不可撤销**；需要保留就先备份 `human_check.csv` 和 `human_scores.json`。
- **重新抽 25 条**：换 `--seed`（如 `--seed 43`），并删除 `human_translations.json` 后重新翻译，否则 row_id 与旧翻译错位。

# 7. 翻译缓存（英/德内容）

固定内容的英/德版本由脚本预生成：

```bash
python scripts/translate_content.py --run results/full_gpu
```

- 人工评分：`results/full_gpu/human_translations.json`（25 条 × en/de）；
- 对比回答：`results/full_gpu/comparison_translations.json`（3 组 × en/de）；
- 脚本会校验输出语言，整段中文会被判失败并重试；超长答案按段落分块；可断点续跑；
- 新增/修改题目、重新导出评分样本、或换实验运行后，**删除对应翻译文件再重跑**，避免 row_id / case_id 错位；
- 翻译缺失时，页面自动回退中文并显示提示，不会把中文当外语展示。

# 8. 常见问题

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `ModuleNotFoundError: langchain_openai` | 用了 base 环境 | `conda activate chain_1_fixed` |
| `cuda build: None` / 仍用 CPU | 装了 CPU 版 torch | 按 README 安装 `torch==2.11.0+cu128`，或设 `DEVICE=cpu` 接受 CPU |
| CUDA out of memory | 显存被其他程序占用 | 关闭占显存的程序；或 `DEVICE=cpu` 重跑（新建输出目录） |
| 端口被占用 | 8000/8001 已有服务 | 换端口 `--port 8003`；`netstat -ano | findstr :8000` 查进程 |
| 打分页"查看校准结果"灰色 | 未打满 25 条 | 继续打分；后端也会对提前请求返回 409 |
| 英/德界面内容仍是中文 | 翻译缓存缺失 | 运行 `scripts/translate_content.py`，重启打分页 |
| 练手分数出现在正式结果里 | 没设 `HUMAN_RUN_DIR` 或端口搞混 | 确认页面顶部运行名是 `_practice`；删除 `_practice` 重来 |
| 入库块数与预期不符 | 解析/分块参数改动或旧缓存 | 看 `storage/ingest_report_*.json`；必要时删 `storage/parsed/` 后 `--reset` 重建 |
| judge 结果想重算 | 改了 judge 提示词 | `python -m eval.runner --rejudge --out-dir results/full_gpu`；judge 缓存会自动命中相同输入 |
| API 401/超时 | key 无效或网络 | 检查 `.env` 的 `DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL` |
| 控制台中文/德文乱码 | 终端编码是 GBK | 项目 CLI 已自动切 UTF-8；自定义脚本可设 `PYTHONIOENCODING=utf-8` |
| 首次演示很慢（40 秒+） | 模型加载 | 正常；之后同题命中检索缓存约 3 秒 |
| `human_check.csv` 把 Excel 弄乱 | 直接保存成了非 UTF-8 | 另存为"CSV UTF-8"；或直接用网页打分，不经过 Excel |

# 9. 命令速查

| 目的 | 命令 |
| --- | --- |
| 激活环境 | `conda activate chain_1_fixed` |
| 建三套索引 | `python scripts/build_all_indexes.py --reset` |
| 预检 | `python scripts/preflight.py` |
| 冒烟 | `python -m eval.runner --smoke` |
| 全量 16 臂 | `python -m eval.runner --repeats 3 --workers 8 --out-dir results/full_gpu` |
| 重判 | `python -m eval.runner --rejudge --out-dir results/full_gpu` |
| 出图/看板 | `python -m eval.visualize --run results/full_gpu` |
| 问答 Demo | `python -m uvicorn demo.app:APP --port 8000` |
| 人工打分 | `python -m uvicorn eval.human_app:APP --port 8001` |
| 练手打分 | `HUMAN_RUN_DIR="$(pwd)/results/_practice" python -m uvicorn eval.human_app:APP --port 8002` |
| 导出样本 | `python -m eval.human export --run results/full_gpu --n 25 --seed 42` |
| 计算 kappa | `python -m eval.human score --csv results/full_gpu/human_check.csv` |
| 生成翻译 | `python scripts/translate_content.py --run results/full_gpu` |
| 测试 | `python -m pytest tests -q` |
| 冻结清单 | `python scripts/freeze.py` |

# 10. 典型操作场景

### 场景 A：第一次跑通（约 30–40 分钟）

1. `conda activate chain_1_fixed`
2. `cp .env.example .env`，填 `DEEPSEEK_API_KEY`
3. `python scripts/build_all_indexes.py --reset`（GPU 上数分钟）
4. `python scripts/preflight.py`（应 22/22）
5. `python -m eval.runner --smoke`（2 题 × 4 臂）
6. `python -m eval.runner --arms A "A'" B C "D'" --out-dir results/phase1_gpu`
7. `python -m eval.visualize --run results/phase1_gpu`
8. `python -m uvicorn demo.app:APP --port 8000`，打开验证

### 场景 B：只做演示（约 2 分钟）

1. `python -m uvicorn demo.app:APP --port 8000`
2. 打开 http://127.0.0.1:8000
3. 按 4.1 的"演示脚本建议"走一遍
4. 需要对比区时确认 `results/full_gpu/raw.jsonl` 存在

### 场景 C：正式人工打分（约 45–75 分钟）

1. 确认 `results/full_gpu/human_check.csv` 存在且 0/25
2. `python -m uvicorn eval.human_app:APP --port 8001`
3. 打开 8001，确认运行名 `full_gpu`
4. 按 6.1 完成 25 条
5. 查看 kappa，下载 CSV
6. 把 kappa 写进报告，替换"尚未校准"的表述

### 场景 D：练手打分（约 10 分钟，不影响正式结果）

1. 复制到 `results/_practice`（见 6.2）
2. `HUMAN_RUN_DIR=... python -m uvicorn eval.human_app:APP --port 8002`
3. 打开 8002，确认运行名 `_practice`
4. 随便打几条，验证界面与保存
5. 停止服务，`rm -rf results/_practice`

### 场景 E：改题/改提示词后的回归

1. 修改 `corpus/questions.jsonl` 或 `prompts/templates.py`
2. 若影响分块/题库证据：重建索引 `python scripts/build_all_indexes.py --reset`
3. `python scripts/preflight.py`（证据解析必须 22/22）
4. 新建输出目录重跑全量（**不要**覆盖旧结果）：
   `python -m eval.runner --out-dir results/regression_YYYYMMDD`
5. `python -m eval.visualize --run results/regression_YYYYMMDD`
6. 与旧结果对比 `paired.csv`；质量下降且 CI 不含零 → 判为回归
7. 更新 `docs/FROZEN.md` 并 `python scripts/freeze.py`

### 场景 F：换硬件复跑（CPU ↔ GPU）

1. 新建目录：`results/full_gpu2` / `results/full_cpu`
2. 设 `DEVICE=auto` 或 `DEVICE=cpu`
3. `python -m eval.runner --out-dir results/full_gpu2`
4. 不要把不同硬件的结果续跑到同一目录；延迟不可比

# 附录 A 当前状态

- 语料：20 篇 PDF，409 页，英/德混合。
- 索引：固定 1,730 块；结构 1,066 块；父子 6,765 子块 + 978 父块。
- 题库：30 题（22 可答 + 8 不可答），72 条金标准引文；英/德变体 10 条。
- 完整运行：`results/full_gpu`，16 臂 × 30 题 × 3 次 = 1,440 行，GPU 约 11 分钟。
- 主臂完整性：A 0.288、A' 0.220、B 0.765、C 0.758、D' 0.894（22 道可答题）。
- 关键结论：检索是主效应；C 相对 B 不显著（−0.008，p=0.905）；Step-back 是唯一显著正收益（+0.083，p=0.020）；D 显著差于 D'（−0.182，p<0.001）。
- 人工打分：`results/full_gpu/human_check.csv` 当前 0/25，`human_scores.json` 尚未生成。

# 附录 B 指标口径（操作相关）

- 完整性：judge 对照参考答案与关键要点给 0 / 0.5 / 1；每题 3 次重复先取均值。
- 相关性：0 / 1；judge 偶发 0.5 时按 ≥0.5 → 1 归一并保留原始值。
- 忠实度：答案声明被检索上下文支持的比例；无上下文臂为空。
- 引用精度：只核对答案中实际出现的 `[C#]`；有引用义务但未给引用记 0。
- 正确/错误拒答：8 道不可答题被判拒答 / 22 道可答题被判拒答。
- 延迟：整条链路的中位数与 p95；GPU 与 CPU 不可混比。
- 成本：输入 ¥2/M + 输出 ¥8/M，不含缓存折扣，也不含 judge 调用。

# 附录 C 安全与分享

- `rag_lab/.env` 含真实 API key，**对外打包/上传前必须删除或替换为 `.env.example`**。
- `storage/` 与 `results/` 含中间数据，可按需裁剪；`corpus/raw/` 为原始论文，注意版权与分享范围。
- 对外演示前，建议用练手目录确认打分流程，避免误改正式 `human_scores.json`。

# 附录 D 输出文件说明

### D.1 实验运行目录

| 文件 | 说明 |
| --- | --- |
| `raw.jsonl` | 每行一条：臂、题号、答案、上下文、检索块 ID、候选池、token、延迟、judge 分数、`error_class` |
| `summary.csv` | 每臂一行：完整性/相关性/忠实度/引用精度/拒答/延迟/成本/检索指标 |
| `by_type.csv` | 臂 × 题型 |
| `by_language.csv` | 臂 × 语言（中文主报告，英/德单独成组） |
| `paired.csv` | 各臂 vs 基线的配对 bootstrap（均值差、95% CI、p 值） |
| `errors.csv` | 错题与失败类别（E1–E6） |
| `report.md` | 简版 Markdown 报告 |
| `run_meta.json` | 模型、温度、题库哈希、全部臂配置（预注册证据） |
| `fig*.png` | 9 张图（按阶段生成） |
| `dashboard.html` | 单文件看板，双击打开 |

### D.2 人工打分目录

| 文件 | 说明 |
| --- | --- |
| `human_check.csv` | 25 条盲评表（含参考答案、关键要点、judge 分、人工分） |
| `human_check_key.csv` | row_id → 臂/题号映射；打分时不要打开 |
| `human_scores.json` | 网页打分的原始记录（自动保存） |
| `human_translations.json` | 人工评分条目的英/德机器翻译缓存 |
| `comparison_translations.json` | 三臂对比回答的英/德机器翻译缓存 |

### D.3 常见字段

| 字段 | 含义 |
| --- | --- |
| `arm` | 实验臂 ID（A、B、C、C-q…） |
| `case_id` | 题号（q001…q030） |
| `repeat` | 第几次重复（0/1/2） |
| `completeness` | judge 完整性（0/0.5/1） |
| `relevance` | judge 相关性原始值（0/0.5/1；报告按 ≥0.5→1） |
| `faithfulness` | judge 忠实度（0–1，无上下文为空） |
| `citation_precision` | judge 引用精度（有引用义务的臂） |
| `refused` | 是否拒答 |
| `retrieved_ids` | 最终进入上下文的块/父块 ID |
| `candidate_ids` | 截断前的候选池，用于 E1/E2 归因 |
| `error_class` | E1 未召回 / E2 截断 / E3 生成 / E4 引用 / E5 拒答 / E6 工程 |

