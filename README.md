# HAM_in_DL

Skin Lesion Classification Using Deep Learning — HAM10000 / ISIC 2018 Task 3 group project scaffold.

本仓库是课程项目"皮肤病灶图像 7 分类"的完整实现，使用 PyTorch 完成了: 数据预处理、lesion-level 划分、Baseline CNN 训练、ResNet18 / ConvNeXt-Tiny 迁移学习训练、验证评估、测试评估、Grad-CAM 可解释性可视化和错误案例分析。

## 1. Project goals

本项目需要完成：

- 至少一个 baseline deep learning model，例如简单 CNN。
- 至少一个 transfer learning model，例如 ResNet、DenseNet、EfficientNet、MobileNet。
- 使用 PyTorch 完成训练、验证、测试和模型保存。
- 报告 accuracy、precision、recall、F1-score、Macro-F1 和 confusion matrix。
- 对比不同模型或训练策略。
- 讨论 class imbalance、overfitting、data augmentation 和 generalization。
- 使用 Grad-CAM 或其他方法进行模型解释。
- 分析成功预测和失败预测案例。
- 提交 report、source code、model weights、README 和 AI dialogue records。

## 2. Suggested team workflow

建议按模块分工，避免所有人改同一个文件。

| Role | Main responsibility | Main folders/files |
|---|---|---|
| Data Lead | 数据路径、标签、划分、增强、类别不平衡处理 | `src/ham_in_dl/data/`, `scripts/make_splits.py` |
| Modeling Lead | Baseline CNN、transfer learning 模型 | `src/ham_in_dl/models/` |
| Training Lead | 训练循环、loss、optimizer、checkpoint、日志 | `src/ham_in_dl/training/`, `scripts/train_*.py` |
| Evaluation Lead | 指标、confusion matrix、test 结果、错误分析 | `src/ham_in_dl/evaluation/`, `scripts/evaluate.py` |
| Interpretation Lead | Grad-CAM、成功/失败样例可视化 | `src/ham_in_dl/interpretation/`, `scripts/run_gradcam.py` |
| Report Lead | PDF 报告、图表整理、AI 使用声明 | `report/`, `docs/`, `ai_dialogue_records/` |

如果小组人数较少，可以合并角色；如果人数较多，可以把 transfer learning、错误分析、报告写作再拆开。

## 3. Repository structure

```text
HAM_in_DL/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── baseline_cnn.yaml
│   ├── resnet18.yaml
│   └── convnext_tiny.yaml
├── data/
│   ├── README.md
│   ├── raw/
│   ├── processed/
│   └── splits/
├── src/
│   └── ham_in_dl/
│       ├── config.py
│       ├── constants.py
│       ├── seed.py
│       ├── data/
│       ├── models/
│       ├── training/
│       ├── evaluation/
│       ├── interpretation/
│       └── utils/
├── scripts/
├── outputs/
│   ├── figures/           # experiment summary figures
│   ├── gradcam/           # Grad-CAM heatmaps
│   └── error_analysis/    # error case CSVs
├── checkpoints/
├── report/
├── docs/
├── ai_dialogue_records/
└── tests/
```

## 4. Dataset placement

数据集按以下结构组织（`data/raw/` 和 `data/processed/` 不在 git 中）：

```text
data/
├── raw/
│   └── _downloads/
│       ├── kagglehub_cache/archive.zip   # Kaggle HAM10000 压缩包
│       ├── ham10000_kaggle/              # 解压后（10015 张图像）
│       └── ham10000_testset/             # 老师提供的 TestSet
├── processed/
│   ├── ham10000/                         # 整理后的训练数据
│   │   ├── images/
│   │   │   ├── HAM10000_images_part_1/
│   │   │   └── HAM10000_images_part_2/
│   │   └── metadata.csv
│   └── testset/                          # 整理后的测试数据
│       ├── images/
│       ├── metadata.csv
│       └── groundtruth.csv
└── splits/
    ├── train.csv                         # lesion-level 训练划分
    └── val.csv                           # lesion-level 验证划分
```

课程提供的压缩包不要上传到 GitHub。`setup_data.py` 会自动处理 `archive.zip` 的搬运与解压。

## 5. Environment setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 6. Common commands

### 6.1 Data setup

```bash
# 1) 将 archive.zip 放到仓库根目录（或手动放入 data/raw/_downloads/kagglehub_cache/）
# 2) 一键解压 + 整理 + 校验
python scripts/setup_data.py --force

# 3) 生成 lesion-level train/val 划分（三个模型共用）
python scripts/make_splits.py --config configs/resnet18.yaml --force
```

Split 按 `lesion_id` 分组，确保同一病灶的所有图片不会同时出现在 train 和 val 中。

### 6.2 Train models

```bash
# Baseline CNN（轻量，快速验证流程）
python scripts/train_baseline.py --config configs/baseline_cnn.yaml

# ResNet18 迁移学习
python scripts/train_transfer.py --config configs/resnet18.yaml

# ConvNeXt-Tiny 迁移学习
python scripts/train_transfer.py --config configs/convnext_tiny.yaml
```

### 6.3 Evaluate on validation set

```bash
python scripts/evaluate.py --config configs/baseline_cnn.yaml --split val
python scripts/evaluate.py --config configs/resnet18.yaml --split val
python scripts/evaluate.py --config configs/convnext_tiny.yaml --split val
```

### 6.4 Evaluate on test set (final report only)

```bash
python scripts/evaluate.py --config configs/convnext_tiny.yaml --split test
```

### 6.5 Error analysis

```bash
python scripts/analyze_errors.py --predictions outputs/predictions_convnext_tiny.csv
```

### 6.6 Grad-CAM visualization

```bash
# 单模型失败样本
python scripts/run_gradcam.py --config configs/convnext_tiny.yaml --predictions-csv outputs/predictions_convnext_tiny.csv --case failed --num-samples 5

# 一键对比模式：自动生成 correct + failed + gradcam_comparison.png 对照图
python scripts/run_gradcam.py --config configs/convnext_tiny.yaml --predictions-csv outputs/predictions_convnext_tiny.csv --comparison
```

Grad-CAM 依赖 `evaluate.py` 已生成的 predictions CSV。`--comparison` 模式会同时生成 correct/failed 单张图，并输出一张并排对照表 `outputs/gradcam/gradcam_comparison.png`。当前 `outputs/gradcam/` 中保存的是 ConvNeXt-Tiny 的结果。

### 6.7 One-click reproduction (full pipeline)

```bash
# === 数据准备 ===
python scripts/setup_data.py --force
python scripts/make_splits.py --config configs/resnet18.yaml --force

# === 训练 ===
python scripts/train_baseline.py --config configs/baseline_cnn.yaml
python scripts/train_transfer.py --config configs/resnet18.yaml
python scripts/train_transfer.py --config configs/convnext_tiny.yaml

# === Val 评估 ===
python scripts/evaluate.py --config configs/baseline_cnn.yaml --split val
python scripts/evaluate.py --config configs/resnet18.yaml --split val
python scripts/evaluate.py --config configs/convnext_tiny.yaml --split val

# === Test 评估 ===
python scripts/evaluate.py --config configs/convnext_tiny.yaml --split test

# === 错误分析 ===
python scripts/analyze_errors.py --predictions outputs/predictions_convnext_tiny.csv

# === Grad-CAM ===
python scripts/run_gradcam.py --config configs/convnext_tiny.yaml --predictions-csv outputs/predictions_convnext_tiny.csv --comparison
```

### 6.8 Experiment summary visualizations

```bash
python scripts/plot_visualizations.py
```

Generates five report-quality figures in `outputs/figures/`:

| Figure | Description |
|--------|-------------|
| `loss_curves.png` | Train & validation loss for all 3 models (side-by-side) |
| `accuracy_curves.png` | Train & validation accuracy for all 3 models |
| `parameter_comparison.png` | Bar chart of trainable parameter counts |
| `per_class_f1.png` | Grouped bar chart of per-class F1 scores |
| `overfitting_gap.png` | Train−Val Macro-F1 gap over epochs (quantifies overfitting) |

## 7. Branch and commit conventions

建议每位成员使用自己的 feature branch：

```bash
git checkout -b feature/data-loader
```

推荐 commit 格式：

```text
[data] implement HAM10000 dataset loader
[model] add baseline CNN skeleton
[train] add checkpoint saving
[eval] add confusion matrix plot
[report] add dataset description draft
```

## 8. Experiment logging

每次实验建议记录：

- 日期和负责人
- Git commit hash
- 模型名称
- 数据增强策略
- loss、optimizer、learning rate、batch size、epoch
- validation metrics
- test metrics，仅最终报告时使用
- 主要结论

可以复制 `docs/experiment_log_template.md`。

## 9. Report checklist

最终报告建议包括：

- Title and group members: student ID and Chinese name
- Abstract
- Introduction and motivation
- Dataset description and preprocessing
- Methodology
- Experimental results and analysis
- Model interpretation and error analysis
- Conclusion
- References
- AI tool usage statement

## 10. Submission checklist

最终提交 zip 文件前确认：

- [ ] `report/final_report.pdf`
- [ ] `src/` 和 `scripts/` 中的完整 PyTorch 代码
- [ ] `configs/` 三个模型配置文件
- [ ] `checkpoints/baseline_cnn_42.pt`（~141 MB）
- [ ] `checkpoints/resnet18_42.pt`（~134 MB）
- [ ] `checkpoints/convnext_tiny_42.pt`（~334 MB）
- [ ] `outputs/` 中所有评估产物（history CSV、predictions CSV、confusion matrix、Grad-CAM 图、错误分析）
- [ ] `data/splits/train.csv` + `val.csv`（lesion-level 划分文件）
- [ ] `README.md` 包含完整复现命令
- [ ] `ai_dialogue_records/` 中包含 AI 使用记录
- [ ] validation 和 test performance 均已报告
- [ ] test set 没有被用于模型选择或调参
- [ ] 报告中三个模型的 Val Macro-F1 分别为 37.77%, 71.12%, 78.10%

## 11. Academic integrity note

AI 工具可以用于 debugging、解释代码、语言润色、生成思路等，但必须在报告中说明。最终代码、实验和分析需要由小组成员理解、验证并负责。
