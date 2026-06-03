# eval/ —— 4-cell 安全评测代码使用文档

本目录是 §5 Procrustes-MPC 工作所用的**评测代码快照**。包含：在 7 个 VLM 上做带 steering 的生成（generate）、安全/过拒绝/效用 judge、ScienceQA 字母评分、Pareto 汇总、以及关键编排脚本。

数据 manifest 与图像见项目根目录 `data/`；数据来源说明见根目录 `DATASETS.md`。

---

## 1. 目录结构

```
eval/
├── README.md              <- 本文件
├── procrustes/            <- Python 包：steering hooks + 子空间数学 + 模型工具
├── cmrm/                  <- Python 包：manifest 读取 + judge 封装 + 模型加载
├── generate/              <- 7 VLM 带 Procrustes-MPC steering 的生成脚本
├── baseline/              <- 7 VLM 无 steering 的基线生成脚本
├── judge/                 <- 3 个 judge 脚本（safety / over-refusal / context-aware）
├── score/                 <- ScienceQA 字母匹配评分
├── aggregate/             <- 安全-效用 Pareto 汇总
└── runners/               <- Phase 5 / 6 编排 shell 脚本（参考用，路径需改）
```

| 模块 | 内容 | 备注 |
|---|---|---|
| `procrustes/` | `hooks.py`（ProcrustesHookManager）, `subspace.py`（Procrustes 拟合数学）, `cmrm_compat.py`（统一 manifest 读取）, `{gemma3,internvl,internvl3,phi35v}_utils.py`（模型特定 hook 装配） | 镜像自 `ProcrustesRotation/procrustes/` |
| `cmrm/` | `models.py`（VLM/judge 加载）, `judging.py`（judge prompt 与单条评判）, `manifest.py`（jsonl 读取）, `safety.py`, `prompts.py` | 镜像自 `CMRM/cmrm/` |
| `generate/` | 8 份：`{qwen25vl,qwen3vl,internvl,internvl3,llava_next,phi35v,gemma3}_procrustes_generate.py` + `llava15_procrustes_generate.py`（即原 `04_generate_procrustes.py`） | 每个 VLM 自带一份生成脚本（不同 backbone 的 hook 装配方式不同） |
| `baseline/` | 8 份：同上 VLM 集合，纯基线（无 Procrustes hook） | 用于得到 Cell × VLM 的 baseline unsafe% 对照 |
| `judge/` | `judge_actionable_safety.py`（即原 `07_judge_outputs.py`）, `judge_over_refusal.py`, `judge_context_aware.py` | 共用 Llama-3.1-8B-Instruct 作为 judge |
| `score/` | `score_scienceqa.py` | CPU 字母匹配，无需 GPU |
| `aggregate/` | `06_summarize_safety_quality_pareto.py` | 把多份 judge 结果聚合成 (unsafe%, sciqa%, OR%) 三元组 + Pareto 决策 |
| `runners/` | 5 份：`run_phase5_smoke.sh`, `run_phase5_extend.sh`, `run_phase5_full_extend.sh`, `orchestrate_phase5_full_extend.sh`, `judge_phase5_round2.sh` | 内部使用**绝对路径**，迁移时需要改前缀 |

---

## 2. 评测流水线总览

```
                                   ┌────────────────┐
   data/manifests/<X>.jsonl ──────▶│ generate/ 或   │──▶ outputs/.../<cell>.jsonl
                                   │ baseline/      │       （response 文本）
                                   └────────────────┘
                                            │
                                            ▼
            ┌─────────────────────────────────────────────────────────┐
            │  judge/judge_actionable_safety.py   (Cell A/B/C 安全)   │
            │  judge/judge_context_aware.py       (Cell D 上下文安全) │
            │  judge/judge_over_refusal.py        (benign 过拒绝)     │
            │  score/score_scienceqa.py           (ScienceQA 字母准确率)│
            └──────────────┬──────────────────────────────────────────┘
                           ▼
                        *.judged.jsonl / sciqa_score.csv
                           │
                           ▼
                   aggregate/06_summarize_safety_quality_pareto.py
                           │
                           ▼
                   (unsafe%, sciqa%, OR%) 三元组 + Pareto 决策
```

---

## 3. 前置依赖

| 项 | 内容 |
|---|---|
| Python venv | `/home/kedong/repos/VLM-subspace-steering/CMRM/.venv/bin/python`（torch + transformers + tqdm） |
| Procrustes 拟合产物 | `ProcrustesRotation/outputs/<vlm>_procrustes_params_k16.pt`（7 VLM 各一份） |
| 拒绝方向 r̂ | `ProcrustesRotation/outputs/refusal_dir_<vlm>.pt` |
| Manifest | `data/manifests/<X>.jsonl`（见根目录 `DATASETS.md`） |
| Judge 模型 | Llama-3.1-8B-Instruct（路径在 `cmrm/config.py` 默认 YAML 中配置） |
| GPU | 单步用 1 张 80 GB H100/A100；并发跑多 VLM 时按 `CUDA_VISIBLE_DEVICES` 切分 |

> **重要**：所有脚本通过 `sys.path.insert(0, parents[1])` 从所在脚本目录的**上一级**导入 `procrustes` / `cmrm` 包。`eval/` 这种布局可直接运行，无需手动改 `PYTHONPATH`。

---

## 4. 使用步骤

### 4.1 步骤 1：生成（per VLM × per cell）

```bash
VENV=/home/kedong/repos/VLM-subspace-steering/CMRM/.venv/bin/python
EVAL=/home/kedong/repos/VLM-subspace-steering/eval
PR=/home/kedong/repos/VLM-subspace-steering/ProcrustesRotation
DATA=/home/kedong/repos/VLM-subspace-steering/data

# 例：qwen25vl 在 Cell C/sdtypo 用 refp α=0.2
CUDA_VISIBLE_DEVICES=0 $VENV $EVAL/generate/qwen25vl_procrustes_generate.py \
    --manifest $DATA/manifests/mmsb_vision_risk_sdtypo.jsonl \
    --params   $PR/outputs/qwen25vl_procrustes_params_k16.pt \
    --alpha 0.2 --lambda_mean 1 --mean_shift_mode refusal_projected \
    --refusal_dir $PR/outputs/refusal_dir_qwen25vl.pt \
    --hook_scope prefill_only --max_new_tokens 192 \
    --out outputs/qwen25vl_a02_refp/mmsb_sdtypo.jsonl
```

**关键参数**：

| flag | 含义 |
|---|---|
| `--manifest` | jsonl 输入路径 |
| `--params` | Procrustes 拟合产物 `.pt` |
| `--alpha` | steering 强度 α（α=0 ≈ baseline；常用 0.2、0.3、0.5） |
| `--lambda_mean` | mean-shift 系数 λ（0 = 仅旋转 Q；1 = 含 mean shift） |
| `--mean_shift_mode` | `full`（完整 δμ）或 `refusal_projected`（即 refp，沿 r̂ 投影） |
| `--refusal_dir` | refp 模式所需 r̂ 文件 |
| `--hook_scope` | `prefill_only`（论文默认） / `all` |
| `--max_new_tokens` | sciqa/benign/mmsb=192，vlsafe=256 |
| `--out` | 输出 jsonl（每行含 `id, query, response, method`） |

**baseline**：把 `generate/<vlm>_procrustes_generate.py` 换成 `baseline/<vlm>_baseline.py`，去掉 `--params --alpha --lambda_mean --mean_shift_mode --refusal_dir`，仅保留 `--manifest --max_new_tokens --out`。

### 4.2 步骤 2a：actionable safety judge（Cell A/B/C）

```bash
CUDA_VISIBLE_DEVICES=0 $VENV $EVAL/judge/judge_actionable_safety.py \
    --config $PR/../CMRM/configs/qwen25vl_vlsafe.yaml \
    --judge_style actionable \
    --generations outputs/qwen25vl_a02_refp/mmsb_sdtypo.jsonl \
    --out         outputs/qwen25vl_a02_refp/mmsb_sdtypo.judged.jsonl
```

输出每行新增 `label ∈ {UNSAFE, SAFE, PARSE_ERROR}`。

### 4.3 步骤 2b：context-aware judge（Cell D）

```bash
CUDA_VISIBLE_DEVICES=0 $VENV $EVAL/judge/judge_context_aware.py \
    --generations outputs/qwen25vl_a02_refp/siuo_167.jsonl \
    --out         outputs/qwen25vl_a02_refp/siuo_167.judged.jsonl
```

与 actionable judge 的差异：把 `(image, query, response)` 三元组一起给 judge，问 “在该 image-query 组合下，response 是否会助长真实世界伤害”。Cell D 必须用这个，普通 actionable judge 会漏判。

### 4.4 步骤 2c：over-refusal judge（benign-60）

```bash
CUDA_VISIBLE_DEVICES=0 $VENV $EVAL/judge/judge_over_refusal.py \
    --config $PR/../CMRM/configs/qwen25vl_vlsafe.yaml \
    --generations outputs/qwen25vl_a02_refp/benign60.jsonl \
    --out         outputs/qwen25vl_a02_refp/benign60.judged.jsonl
```

输出 `label ∈ {OK, OVER_REFUSAL, PARSE_ERROR}`。OR% = `OVER_REFUSAL / total`。

### 4.5 步骤 2d：ScienceQA 字母评分（CPU）

```bash
$VENV $EVAL/score/score_scienceqa.py \
    --generations outputs/qwen25vl_a02_refp/sciqa200.jsonl \
    --manifest    $DATA/manifests/scienceqa_imgval_n200.jsonl \
    --out         outputs/qwen25vl_a02_refp/sciqa_score.csv
```

输出 CSV：`overall_accuracy, per_category_accuracy`。

### 4.6 步骤 3：Pareto 汇总

```bash
$VENV $EVAL/aggregate/06_summarize_safety_quality_pareto.py \
    --root outputs/  --out outputs/pareto_table.md
```

输出 markdown 表：每 (VLM, cell, mode) 一行 → (unsafe%, sciqa%, OR%)，并标记 Pareto-best。

---

## 5. 输出文件命名约定

```
outputs/<vlm>_<a-tag>_<mode>/<manifest-stem>.jsonl          ← 生成结果
outputs/<vlm>_<a-tag>_<mode>/<manifest-stem>.judged.jsonl   ← judge 结果
outputs/<vlm>_<a-tag>_<mode>/sciqa_score.csv                ← sciqa 评分
```

例：`outputs/qwen25vl_a02_refp/mmsb_sdtypo.jsonl`
- `qwen25vl` = VLM 名
- `a02` = α=0.2
- `refp` = `refusal_projected` mean-shift 模式
- `mmsb_sdtypo` = manifest stem（对应 `data/manifests/mmsb_vision_risk_sdtypo.jsonl`）

---

## 6. Cell ↔ 数据集 ↔ judge 速查

| Cell | 数据 manifest | 该用哪个 judge |
|---|---|---|
| **A** 纯文本危害 | `vlsafe_examine_eval.jsonl` (1110) | `judge_actionable_safety.py` |
| **B** 文本+图像双危害 | `spa_vl_test_530.jsonl` (530) | `judge_actionable_safety.py` |
| **C/sdtypo** 视觉-only | `mmsb_vision_risk_sdtypo.jsonl` (443) | `judge_actionable_safety.py` |
| **C/MMSB** 视觉-only | `mm_safetybench_300.jsonl` (300) | `judge_actionable_safety.py` |
| **D/SIUO** emergent | `siuo_167.jsonl` (167) | `judge_context_aware.py` |
| **D/MSSB** emergent | `mssbench_unsafe_full.jsonl` (600) | `judge_context_aware.py` |
| 效用 sciqa% | `scienceqa_imgval_n200.jsonl` (200) | `score_scienceqa.py`（CPU） |
| 效用 OR% | `benign_multimodal_n60.jsonl` (60) | `judge_over_refusal.py` |

---

## 7. 论文 §5 一次完整复现需要的 step 数

| 7 VLM × Cell-A 单 cell | 7 × 3 mode × 2 judge + 1 sciqa + 1 OR = **49 步** | ~10 GPU-hr |
| 7 VLM × 4 cell × 3 mode | 7 × 4 × 3 × 1 judge = **84 步**（效用 mode-fixed 共用一份） | ~70 GPU-hr |

**3 modes = `{lambda0, lambda1_full, refp_α=0.2}`**；每个 VLM 选其对应 §5 picked α。

---

## 8. runners/ 内 shell 脚本说明

`runners/` 下 5 个 .sh 内部用了**绝对路径**（`/home/kedong/repos/VLM-subspace-steering/...`），仅作参考。在新机使用时：

| Shell | 用途 | 迁移要点 |
|---|---|---|
| `run_phase5_smoke.sh` | 双 lane 跑 qwen25vl+internvl25 的 α=0.1/0.2 magnitude smoke 测试 | 改 `VENV`, `CMRM`, `PR`, `LANE/GPU` 即可 |
| `run_phase5_extend.sh` | 单 lane 跑 qwen3vl 或 internvl3 的 α=0.2 验证（4 manifest） | 同上 |
| `run_phase5_full_extend.sh` | 单 lane × 单 VLM 跑 4 missing cell（B + C/mmsb + D/siuo + D/mssb） | 同上 |
| `orchestrate_phase5_full_extend.sh` | 2-lane 编排：等 phase5_extend PID 退出 → 串行触发 `qwen25vl→qwen3vl` 和 `internvl25→internvl3` | 内部硬编码 PID，需替换 |
| `judge_phase5_round2.sh` | phase5 第二轮 judge 编排 | 同上 |

> **常用最小化命令模板**直接见 §4，runners/ 仅作长任务编排参考。

---

## 9. 常见问题

**Q1：跑 generate 时报 `ModuleNotFoundError: procrustes`？**
A：检查脚本的 `parents[1]` 是否指向 `eval/`。手动修复：在脚本前加 `export PYTHONPATH=/path/to/eval:$PYTHONPATH`。

**Q2：generate 输出全为空字符串？**
A：通常是 `--max_new_tokens` 漏掉；vlsafe 用 256，其他用 192。

**Q3：judge 显示大量 `PARSE_ERROR`？**
A：检查 judge 模型路径（在 `cmrm/config.py` 或 yaml 配置）。Llama-3.1-8B-Instruct 默认会按 `UNSAFE/SAFE` 输出，PARSE_ERROR 表示 judge 输出格式异常，常因加载错模型或 prompt 模板错。

**Q4：mean_shift_mode 几种区别？**
A：
- `full`：mean shift = `μ_t − μ_c`（完整 δμ，论文 §3 默认）
- `refusal_projected`（refp）：mean shift = `⟨δμ, r̂⟩ · r̂`（沿拒绝方向投影，论文 §5 验证）
- λ=0 时这两个等价（mean shift 整体为 0）

**Q5：为什么 utility（sciqa+OR）不用每个 cell 重跑？**
A：所有 steering hook 是 **prompt-agnostic**（仅看 hidden state，不看 prompt 内容），所以同一 (α, λ, mode) 下 sciqa%/OR% 在所有 cell 间不变，跑一次即复用。详见 memory `project_phase4b_utility_transfer`。

---

## 10. 配套文档

- `DATASETS.md`（项目根）：数据集来源、4-cell 分类法、子空间拟合数据流。
- `data/README.md`：数据快照说明（manifest + 项目内图像）。
- `ProcrustesRotation/outputs/phase6_aggregate/paper_table4.md`：论文 Table 4 当前结果。
- memory `project_phase5_alpha_finding`：refp α=0.2 magnitude finding 的来龙去脉。
