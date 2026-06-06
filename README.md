# HAM_in_DL

Skin Lesion Classification Using Deep Learning — HAM10000 / ISIC 2018 Task 3 group project scaffold.

本仓库是课程项目的基础架构，用于小组协作完成皮肤病灶图像 7 分类任务。项目目标是用 PyTorch 完成完整深度学习流程：数据处理、baseline CNN、迁移学习模型、训练、验证、测试、指标分析、Grad-CAM 可解释性、错误案例分析和最终报告。

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
│   └── resnet18.yaml
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
├── notebooks/
├── outputs/
│   ├── figures/
│   ├── tables/
│   ├── gradcam/
│   └── predictions/
├── checkpoints/
├── report/
├── docs/
├── ai_dialogue_records/
└── tests/
```

## 4. Dataset placement

不要把大体积图片数据直接上传到 GitHub。建议本地按以下方式放置：

```text
data/raw/
├── _downloads/          # 课程发放的原始压缩包（git 已忽略）
├── train/
├── val/
├── test/
└── metadata.csv        # 如果课程数据提供了标签表，可放这里
```

如果课程提供的数据已经有固定 train / val / test 划分，请保持原始划分，不要用 test set 调参。

课程提供的测试集压缩包不要上传到 GitHub。请每位组员在本地手动放到：

```text
data/raw/_downloads/HAM10000_TestSet.zip
```

数据准备的详细说明见 [data/README.md](file:///d:/%E6%A1%8C%E9%9D%A2/HAM_in_DL/data/README.md)。

## 5. Environment setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 6. Common commands

### 6.1 Data setup quickstart

```bash
# 1) Put archives under data/raw/_downloads/
# 2) Extract + prepare processed + validate
python scripts/setup_data.py --force
```

### 6.2 Train baseline CNN

```bash
python scripts/train_baseline.py --config configs/baseline_cnn.yaml
```

### 6.3 Train transfer learning model

```bash
python scripts/train_transfer.py --config configs/resnet18.yaml
```

### 6.4 Evaluate model

```bash
python scripts/evaluate.py --config configs/resnet18.yaml --checkpoint checkpoints/best_resnet18.pt --split test
```

### 6.5 Run Grad-CAM

```bash
python scripts/run_gradcam.py --config configs/resnet18.yaml --checkpoint checkpoints/best_resnet18.pt --image-path data/processed/testset/images/example.jpg
```

### 6.6 Analyze errors

```bash
python scripts/analyze_errors.py --predictions outputs/predictions/test_predictions.csv
```

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
- [ ] `checkpoints/` 中的训练权重，如果适用
- [ ] `README.md` 可复现实验
- [ ] `ai_dialogue_records/` 中包含 AI 使用记录，如果使用了 AI 工具
- [ ] validation 和 test performance 均已报告
- [ ] test set 没有被用于模型选择或调参

## 11. Academic integrity note

AI 工具可以用于 debugging、解释代码、语言润色、生成思路等，但必须在报告中说明。最终代码、实验和分析需要由小组成员理解、验证并负责。
