# data/ —— 项目所用数据快照

本目录是 §5 Procrustes-MPC 工作中**实际使用的数据**的一份快照，便于离线归档与论文复现。结构：

```
data/
├── manifests/   # 9 个 jsonl manifest，全部 ~2 MB
└── images/      # 项目内生成的 5 个图像目录，共 ~510 MB
```

**完整字段含义、用途与子空间拟合数据流见项目根目录 `DATASETS.md`。** 本 README 仅说明拷贝快照本身。

---

## 1. manifests/ —— 9 份 jsonl

| 文件 | 行数 | 用途 (cell) |
|---|---|---|
| `vlsafe_alignment_anchor_seed42.jsonl` | 600 | **子空间拟合**（取前 200 条做 Procrustes Q 的 anchor） |
| `vlsafe_examine_eval.jsonl` | 1110 | Cell A（纯文本危害） |
| `spa_vl_test_530.jsonl` | 530 | Cell B（文本+图像双危害） |
| `mmsb_vision_risk_sdtypo.jsonl` | 443 | Cell C/sdtypo（视觉-only） |
| `mm_safetybench_300.jsonl` | 300 | Cell C/MMSB（视觉-only） |
| `siuo_167.jsonl` | 167 | Cell D/SIUO（emergent） |
| `mssbench_unsafe_full.jsonl` | 600 | Cell D/MSSB（emergent） |
| `scienceqa_imgval_n200.jsonl` | 200 | 效用：sciqa% |
| `benign_multimodal_n60.jsonl` | 60 | 效用：over-refusal% |

每条记录的 schema：
```json
{"id": "...", "query": "...", "image_path": "...绝对路径...", "metadata": {...}}
```

**注意：`image_path` 全部为绝对路径，未重写。** 一是当前路径在本机仍指向原位置不会失效；二是要保留可溯源的原始 split 标识。如需把数据迁移到其他机器使用，请同步用 `sed` 把绝对前缀替换为目标机器的前缀。

---

## 2. images/ —— 5 个项目内图像目录（共 ~510 MB）

| 子目录 | 体积 | 镜像源（项目内） | 被引用于 |
|---|---|---|---|
| `siuo/` | 199 MB | `ProcrustesRotation/outputs/siuo_images/` | `siuo_167.jsonl` |
| `spa_vl_test/` | 21 MB | `ProcrustesRotation/outputs/spa_vl_test_images/` | `spa_vl_test_530.jsonl` |
| `mmsb_vision_risk/` | 93 MB | `ProcrustesRotation/outputs/mmsb_vision_risk_images/` | `mmsb_vision_risk_sdtypo.jsonl` |
| `mm_safetybench/` | 183 MB | `ProcrustesRotation/outputs/mm_safetybench_images/` | `mm_safetybench_300.jsonl` |
| `scienceqa/` | 14 MB | `CMRM/data/scienceqa_images/` | `scienceqa_imgval_n200.jsonl` |

> 这些目录中的图像由项目脚本生成（SD-TYPO 渲染、SPA-VL 解压、SIUO 解压、ScienceQA 抽样等），**因此完整纳入项目快照**。

---

## 3. ⚠ 未拷贝的外部图像

以下 manifest 引用的图像位于**外部 HF mirror**，体积过大或非项目生成，未纳入快照：

| 来源 | 路径前缀 | 被引用于 |
|---|---|---|
| COCO train2017 | `/hub/huggingface/datasets/coco/train2017/` | `vlsafe_alignment_anchor_seed42.jsonl`, `vlsafe_examine_eval.jsonl`, `benign_multimodal_n60.jsonl` |
| MSSBench | `/hub/huggingface/datasets/MSSBench/` | `mssbench_unsafe_full.jsonl`（300 张唯一图） |

**在新机复现时需要**：
- 通过 `huggingface-cli download` 重新获取这两个数据集，落到相同前缀路径下；
- 或者全量 `sed` 重写 manifest 内的 `image_path`。

---

## 4. 子空间拟合用的数据

- **Procrustes Q**：仅消耗 `manifests/vlsafe_alignment_anchor_seed42.jsonl` 的前 200 条（图像走 COCO 路径，见上文 §3）。
- **拒绝方向 r̂**：**不读 jsonl**，使用 `ProcrustesRotation/scripts/fit_refusal_direction.py` 内硬编码的 30 条 harmful + 30 条 benign **纯文本** probes。无外部图像。

---

## 5. 快照时间

- 创建：2026-05-28
- 用途：§5 全 7 VLM × 4-cell 评测的官方数据快照（即 paper Table 4 的输入）。
