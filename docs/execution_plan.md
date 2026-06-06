# 执行计划书（HAM_in_DL 课程项目）

## 1. 项目要求解读（来自 2026_Project_Requirements.md）

### 1.1 任务目标

- 基于 ISIC 2018 Task 3 / HAM10000 风格数据集完成皮肤病灶 7 分类（MEL/NV/BCC/AKIEC/BKL/DF/VASC）。
- 完成端到端深度学习流程：数据准备 → 训练（baseline + 迁移学习）→ 评估（多指标 + 混淆矩阵）→ 可解释性（Grad-CAM）→ 错误分析 → 报告输出。
- 严格区分 train/val/test：train 用于训练；val 用于模型选择和调参；test 仅用于最终汇报。

### 1.2 必做项（硬性）

- 至少 1 个 baseline 深度学习模型（例如简单 CNN）。
- 至少 1 个 transfer learning 模型（例如 ResNet/DenseNet/EfficientNet/MobileNet 等）。
- 使用 PyTorch 实现训练、验证、测试、模型保存。
- 报告指标：accuracy、precision、recall、F1-score、Macro-F1、confusion matrix。
- 对比不同模型或训练策略，并讨论：
  - class imbalance
  - overfitting
  - data augmentation
  - generalization
- 提供可解释性：Grad-CAM（或同等可视化方法）。
- 分析成功/失败预测案例（错误分析）。
- 提交：report（PDF）、源代码、（可选）权重、README、AI dialogue records。

### 1.3 交付与评分口径

- 总分 30（占课程总评 30%）
  - 研究报告（含实验分析 + 可解释性）：20
  - 代码实现、可复现性、AI 对话记录：10

## 2. 仓库工程方案与约定

### 2.1 目录分层

- `src/ham_in_dl/`：核心库（可被多个脚本复用的“稳定接口”）
- `scripts/`：命令行入口（只负责“读取配置 + 组装 pipeline + 保存产物”）
- `configs/*.yaml`：实验配置（路径/模型/超参/输出目录的单一事实来源）
- `data/`：本地数据（git 忽略大文件，仅保留占位）
- `outputs/`：图表/预测/Grad-CAM/表格等产物（用于报告）
- `checkpoints/`：模型权重（用于复现实验与最终提交）
- `report/`：最终论文与参考文献
- `docs/`：计划、实验记录模板等

### 2.2 数据落盘规范

- `data/raw/_downloads/`：课程下发的原始压缩包（例如 `HAM10000_TestSet.zip`）。
- `data/raw/train/`、`data/raw/val/`、`data/raw/test/`：脚本直接读取的图片目录（与 `configs/*.yaml` 对齐）。
- `data/raw/metadata.csv`：统一的标签表（建议包含 train/val/test 的 label；若 test 标签另有 csv，则在构建 metadata 时合并）。
- `data/splits/`：保存可复现实验的 split 清单（例如 `train.csv`、`val.csv`、`test.csv`，列：`image_id,image_path,label`）。

### 2.3 统一接口原则（防止“脚本各写一套”）

- “读取配置、组装对象”放在 `scripts/`。
- “做具体事情”放在 `src/ham_in_dl/`：
  - 读取 metadata / 生成 items
  - 构建 transforms / dataset / dataloader
  - 构建 model / loss / optimizer
  - fit / evaluate / gradcam / error analysis
- 所有脚本产生的关键产物（checkpoint、预测 csv、图表 png、表格 csv）都必须可重复生成，并在 README 或 docs 中给出命令。

## 3. 整体数据流向（端到端）

```mermaid
flowchart TD
  A[课程下发数据 zip/_downloads] --> B[解压并整理到 data/raw/{train,val,test}]
  B --> C[metadata.csv 统一标签表]
  C --> D[make_splits.py 生成/校验 data/splits/{train,val,test}.csv]
  D --> E[Dataset/Dataloader: items + transforms]
  E --> F[train_baseline.py / train_transfer.py: fit]
  F --> G[checkpoints/*.pt]
  G --> H[evaluate.py: 推理 + 指标]
  H --> I[outputs/predictions/*.csv]
  H --> J[outputs/tables/*.csv]
  H --> K[outputs/figures/*.png]
  I --> L[analyze_errors.py: 成功/失败样例]
  G --> M[run_gradcam.py: Grad-CAM]
  M --> N[outputs/gradcam/*.png]
  J --> O[report: 结果表格]
  K --> O
  N --> O
  L --> O
```

## 4. 脚本填充清单（逐个脚本：做什么、读什么、写什么、接口怎么定）

本节的目标是把 `scripts/*.py` 从 “TODO skeleton” 填充为可复现的 pipeline，同时把核心逻辑沉到 `src/ham_in_dl/`，保证后续扩展不变成泥球。

### 4.1 scripts/make_splits.py（生成或校验 split）

#### 责任边界

- 读取 `configs/*.yaml` 里的 `data.*` 路径。
- 校验 `data/raw/{train,val,test}` 目录是否存在、是否为空。
- 校验/生成 `data/raw/metadata.csv`（若课程已给出则只校验列；若 test 标签为单独文件则合并）。
- 输出 `data/splits/train.csv`、`data/splits/val.csv`、`data/splits/test.csv`（用于训练/评估“只看 split 文件”而不依赖目录枚举的顺序）。

#### 需要新增/补齐的核心接口（建议放到 src/ham_in_dl/data/split.py）

- `def build_metadata(train_dir: str, val_dir: str, test_dir: str, test_gt_csv: str | None) -> "pd.DataFrame": ...`
- `def write_splits(metadata_df, out_dir: str) -> None: ...`
- `def verify_splits(out_dir: str, expected_classes: list[str]) -> None: ...`

#### I/O 约定

- 输入：
  - `data/raw/train/`、`data/raw/val/`、`data/raw/test/`
  - `data/raw/metadata.csv`（或从原始 csv 转换得到）
- 输出：
  - `data/splits/train.csv`、`data/splits/val.csv`、`data/splits/test.csv`
- CSV 列（统一口径）：
  - `image_id`：例如 `ISIC_0035920`
  - `image_path`：相对路径或绝对路径（二选一，建议相对路径便于移动）
  - `label`：字符串（MEL/NV/…）或 int（0..6），二选一；建议存字符串便于人工排查

#### 验收标准

- 能在无 GPU 环境下运行完并输出 3 个 split 文件。
- split 文件中的 `image_path` 都能在本地找到，且 label 只出现在 7 类集合内。

### 4.2 scripts/train_baseline.py（训练 baseline CNN）

#### 责任边界

- 用配置构建：seed、transforms、dataset/dataloader、model、loss、optimizer、训练循环、checkpoint、日志。
- 写出：
  - `checkpoints/best_*.pt`（建议按模型名命名）
  - 每个 epoch 的训练/验证指标（建议 csv + 控制台打印）
  - 训练曲线图（loss/acc）到 `outputs/figures/`

#### 需要新增/补齐的核心接口

建议将训练逻辑集中在 `src/ham_in_dl/training/`：

- `trainer.py`
  - `def train_one_epoch(model, loader, optimizer, loss_fn, device) -> dict: ...`
  - `def validate_one_epoch(model, loader, loss_fn, device) -> dict: ...`
  - `def fit(model, train_loader, val_loader, optimizer, loss_fn, device, epochs: int, early_stopping_patience: int, checkpoint_path: str) -> dict: ...`
- `metrics.py`
  - `def accuracy_from_logits(logits, y) -> float: ...`
  - `def update_running_metrics(running: dict, batch_metrics: dict, batch_size: int) -> dict: ...`

建议将数据装配集中在 `src/ham_in_dl/data/`：

- `def load_split_csv(path: str) -> list[tuple[str, int]]: ...`
- `def build_dataloaders(config: dict) -> tuple[DataLoader, DataLoader, DataLoader]: ...`

#### I/O 约定

- 输入：
  - `configs/baseline_cnn.yaml`
  - `data/splits/*.csv`（优先；若不存在则给出明确报错提示“先运行 make_splits”）
- 输出：
  - checkpoint：`checkpoints/best_baseline_cnn.pt`（建议）
  - logs：`outputs/tables/baseline_cnn_train_log.csv`
  - figures：`outputs/figures/baseline_cnn_learning_curve.png`

#### 验收标准

- 能至少跑通 1 个 epoch 并保存 checkpoint。
- 训练/验证指标可重复生成（同 seed 下波动在合理范围）。

### 4.3 scripts/train_transfer.py（训练迁移学习模型，例如 ResNet18）

#### 责任边界

- 结构与 `train_baseline.py` 一致，仅模型构建不同（`model_factory.build_model`）。
- 支持常见迁移学习策略（通过 config 控制）：
  - `pretrained: true/false`
  - `freeze_backbone: true/false`
  - 可选：分层学习率（backbone vs head）

#### 需要新增/补齐的核心接口

- `models/transfer.py`：已提供 `build_resnet18`，需确保与 config 字段一致。
- `training/`：复用 `fit`，避免两套训练循环分叉。

#### I/O 约定

- 输入：`configs/resnet18.yaml`（或新增 config）
- 输出：
  - `checkpoints/best_resnet18.pt`
  - `outputs/tables/resnet18_train_log.csv`
  - `outputs/figures/resnet18_learning_curve.png`

#### 验收标准

- pretrained / freeze_backbone 不同组合能运行（至少在少量 batch 上 smoke test）。

### 4.4 scripts/evaluate.py（评估与导出预测）

#### 责任边界

- 读取 checkpoint，选择 split（train/val/test）做推理。
- 输出可用于报告的结果：
  - 指标表（accuracy/precision/recall/F1/macro-F1）
  - confusion matrix（图 + 数值表）
  - 逐样本预测清单（用于错误分析与可解释性挑样本）

#### 需要新增/补齐的核心接口（建议放到 src/ham_in_dl/evaluation/evaluate.py）

- `def predict(model, loader, device) -> tuple[y_true, y_pred, y_prob, image_paths]: ...`
- `def compute_metrics(y_true, y_pred, class_names: list[str]) -> dict: ...`
- `def confusion_matrix_table(y_true, y_pred, class_names) -> "pd.DataFrame": ...`
- `def save_predictions_csv(path: str, image_path, y_true, y_pred, y_prob) -> None: ...`

#### I/O 约定

- 输入：
  - `--config configs/resnet18.yaml`
  - `--checkpoint checkpoints/best_resnet18.pt`
  - `--split {train,val,test}`
- 输出（建议命名规则：`{model}_{split}_*`）：
  - `outputs/predictions/resnet18_test_predictions.csv`
  - `outputs/tables/resnet18_test_metrics.csv`
  - `outputs/tables/resnet18_test_confusion_matrix.csv`
  - `outputs/figures/resnet18_test_confusion_matrix.png`

#### 验收标准

- 在 val/test 上跑完并写出上述四类产物。

### 4.5 scripts/analyze_errors.py（错误分析：成功/失败案例）

#### 责任边界

- 读取 `outputs/predictions/*_predictions.csv`。
- 选择代表性样本：
  - Top-K 高置信正确
  - Top-K 高置信错误
  - Top-K 低置信（模型不确定）
  - 每类若干错误样本（保证类别覆盖）
- 输出报告素材：
  - 样本清单（csv）
  - 可选：拼图可视化到 `outputs/figures/`（若实现）

#### 需要新增/补齐的核心接口（建议放到 src/ham_in_dl/evaluation/error_analysis.py）

- `def select_success_and_failure_cases(pred_df, k: int, per_class: bool = True) -> dict[str, "pd.DataFrame"]: ...`
- `def export_case_csv(cases: dict, out_dir: str, prefix: str) -> None: ...`

#### I/O 约定

- 输入：`--predictions outputs/predictions/resnet18_test_predictions.csv`
- 输出：
  - `outputs/tables/resnet18_test_error_cases_success.csv`
  - `outputs/tables/resnet18_test_error_cases_failure.csv`
  - 可选：`outputs/figures/resnet18_test_failure_grid.png`

#### 验收标准

- 能输出可被报告引用的案例清单，且每条记录能定位到本地图片。

### 4.6 scripts/run_gradcam.py（Grad-CAM）

#### 责任边界

- 对单张图片生成 Grad-CAM 热力图，并叠加到原图。
- 支持选择目标层（默认用 resnet 最后一个 block 或最后一层卷积）。
- 输出可直接放进报告的图。

#### 需要新增/补齐的核心接口（建议放到 src/ham_in_dl/interpretation/gradcam.py）

- `def generate_gradcam(model, image_tensor, target_layer, target_class: int | None) -> "np.ndarray": ...`
- `def overlay_heatmap(image_rgb, heatmap) -> "np.ndarray": ...`
- `def save_gradcam_figure(out_path: str, image_rgb, overlay_rgb) -> None: ...`

#### I/O 约定

- 输入：
  - `--config configs/resnet18.yaml`
  - `--checkpoint checkpoints/best_resnet18.pt`
  - `--image-path data/raw/test/ISIC_xxx.jpg`
- 输出：
  - `outputs/gradcam/resnet18_ISIC_xxx.png`

#### 验收标准

- 能对至少 1 张图片产生 overlay 图并保存。

## 5. 任务切片（可并行、可验收）

### Task 1：数据落盘与 split（Data Lead）

- 目的：统一 data/raw 与 metadata/splits 口径，保障后续 pipeline 不会“读取错数据集”。
- 改动范围：`scripts/make_splits.py`、`src/ham_in_dl/data/split.py`、（可选）`src/ham_in_dl/utils/io.py`
- 验收标准：生成 `data/splits/*.csv`，路径可用，标签闭包为 7 类。
- 风险：课程下发的标签文件列名不一致；对策：写适配层并在 docs 记录映射规则。

### Task 2：Dataloader 与增强（Data Lead / Training Lead）

- 目的：实现从 split csv 到 PyTorch DataLoader 的稳定接口。
- 改动范围：`src/ham_in_dl/data/dataset.py`、`src/ham_in_dl/data/transforms.py`
- 验收标准：能在小批量样本上迭代输出 tensor 和 label，尺寸与类别范围正确。
- 风险：PIL 读取失败/图片损坏；对策：在构建 items 时做可选校验并记录坏样本。

### Task 3：训练循环与 checkpoint（Training Lead）

- 目的：实现 `fit`（含 early stopping / best checkpoint / 日志）。
- 改动范围：`src/ham_in_dl/training/trainer.py`、`checkpoint.py`、`metrics.py`、`scripts/train_*.py`
- 验收标准：baseline/resnet18 均能训练并保存权重；日志/曲线产物落盘。
- 风险：GPU/CPU 环境差异；对策：默认 device 自动选择，支持 `--device` 覆盖（如需要）。

### Task 4：评估与可复现产物（Evaluation Lead）

- 目的：指标、混淆矩阵、预测清单标准化输出（为报告服务）。
- 改动范围：`src/ham_in_dl/evaluation/*`、`scripts/evaluate.py`
- 验收标准：能输出 metrics/confusion_matrix/predictions 三件套。
- 风险：precision/recall/F1 的 macro/micro 口径混淆；对策：在输出表中明确 `average=macro`。

### Task 5：Grad-CAM 与案例分析（Interpretation Lead）

- 目的：生成可解释性图和成功/失败案例（报告硬性要求）。
- 改动范围：`src/ham_in_dl/interpretation/*`、`scripts/run_gradcam.py`、`scripts/analyze_errors.py`
- 验收标准：至少产出若干张可用 Grad-CAM 图和一份错误案例清单。
- 风险：目标层选择不当导致热力图无意义；对策：提供可配置 target layer 并记录选择理由。

### Task 6：报告与可复现性（Report Lead）

- 目的：将 outputs 中的产物组织成 report 需要的图表、表格、实验对比结论。
- 改动范围：`report/`、`docs/experiment_log_template.md`、`README.md`
- 验收标准：报告中所有数值/图表均能通过脚本从 checkpoint/outputs 复现。
- 风险：实验记录缺失导致结论不可追溯；对策：每次实验填实验日志模板并保存 config 副本。

## 6. 最小可复现闭环（建议作为全组统一验收）

按顺序能跑通以下命令即视为“工程闭环打通”：

1. `python scripts/make_splits.py --config configs/baseline_cnn.yaml`
2. `python scripts/train_baseline.py --config configs/baseline_cnn.yaml`
3. `python scripts/train_transfer.py --config configs/resnet18.yaml`
4. `python scripts/evaluate.py --config configs/resnet18.yaml --checkpoint checkpoints/best_resnet18.pt --split val`
5. `python scripts/evaluate.py --config configs/resnet18.yaml --checkpoint checkpoints/best_resnet18.pt --split test`
6. `python scripts/analyze_errors.py --predictions outputs/predictions/resnet18_test_predictions.csv`
7. `python scripts/run_gradcam.py --config configs/resnet18.yaml --checkpoint checkpoints/best_resnet18.pt --image-path data/raw/test/example.jpg`

