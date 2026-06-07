# Outputs — Experiment Results

本目录存放完整实验流程产出的所有训练历史、评估指标、预测结果、可视化图、错误分析和 Grad-CAM 热力图。

> **注意：** 本目录内容被 `.gitignore` 排除，提交 zip 时需要手动打包。

---

## 目录结构

```text
outputs/
├── README.md                          # 本文件
├── history_*.csv                      # 训练历史（epoch 级指标）
├── predictions_*.csv                  # 逐样本预测结果
├── val_confusion_matrix.{csv,png}     # 验证集混淆矩阵（最后运行的模型）
├── val_classification_report.csv      # 验证集分类报告（最后运行的模型）
├── val_metrics.json                   # 验证集指标汇总（最后运行的模型）
├── test_confusion_matrix.{csv,png}    # 测试集混淆矩阵（ConvNeXt-Tiny）
├── test_classification_report.csv     # 测试集分类报告（ConvNeXt-Tiny）
├── test_metrics.json                  # 测试集指标汇总（ConvNeXt-Tiny）
├── error_analysis/
│   ├── top_correct.csv                # 高置信度正确预测
│   ├── top_failed.csv                 # 高置信度失败预测
│   └── most_common_confusions.csv     # 最常见类别混淆对
├── gradcam/
│   └── failed_{n}_{original|gradcam}.jpg  # 失败样本热力图（最后运行的模型）
└── figures/
    ├── loss_curves.png                # 三模型 loss 曲线
    ├── accuracy_curves.png            # 三模型 accuracy 曲线
    ├── parameter_comparison.png       # 参数量对比柱状图
    ├── per_class_f1.png              # 类别级 F1 分组柱状图
    └── overfitting_gap.png           # 过拟合 gap 曲线
```

---

## 一、训练历史 (history_*.csv)

### 来源

由 `scripts/train_baseline.py` 和 `scripts/train_transfer.py` 在每个 epoch 结束后自动生成。

### 文件

| 文件 | 对应模型 | 训练配置 |
|------|---------|---------|
| `history_baseline.csv` | Baseline CNN | 30 epoch, lr=0.001, batch=256 |
| `history_resnet18.csv` | ResNet18 | 25 epoch (ES 停于 20), lr=0.0001, batch=128 |
| `history_convnext_tiny.csv` | ConvNeXt-Tiny | 25 epoch, lr=0.0001, batch=128 |

### 字段含义

| 列名 | 含义 |
|------|------|
| `epoch` | 训练轮次 |
| `train_loss` | 训练集交叉熵损失 |
| `train_accuracy` | 训练集准确率 |
| `train_macro_f1` | 训练集 Macro-Averaged F1 |
| `val_loss` | 验证集交叉熵损失 |
| `val_accuracy` | 验证集准确率 |
| `val_macro_f1` | 验证集 Macro-Averaged F1 |

### 核心结论

- **Baseline CNN**: 严重欠拟合，train 准确率最高仅 50.1%，val 在 32–54% 之间震荡。
- **ResNet18**: 最佳 val Macro-F1 71.12%（epoch 13），之后 val_loss 反弹，Early Stopping 在 epoch 20 触发。
- **ConvNeXt-Tiny**: 最佳 val Macro-F1 78.10%（epoch 20），train 准确率高达 97.3%，存在明显过拟合。

---

## 二、预测结果 (predictions_*.csv)

### 来源

由 `scripts/evaluate.py --split val` 对每个模型生成。

### 字段含义

| 列名 | 含义 |
|------|------|
| `image_id` | ISIC 图像 ID |
| `dataset` | 数据来源（空字符串表示 HAM10000） |
| `image_path` | 图像文件路径 |
| `true_label` | 真实类别标签 |
| `pred_label` | 模型预测标签 |
| `true_index` | 真实类别索引 (0–6) |
| `pred_index` | 预测类别索引 (0–6) |
| `confidence` | 预测置信度（softmax 最大值） |
| `correct` | 预测是否正确 (True/False) |

### 核心结论

| 指标 | Baseline CNN | ResNet18 | ConvNeXt-Tiny |
|------|:-----------:|:--------:|:-------------:|
| Val Accuracy | 54.15% | 79.73% | **85.49%** |
| Val Macro-F1 | 37.77% | 71.12% | **78.10%** |

三个模型构成清晰的性能梯度。MEL（黑色素瘤）是所有模型共同的弱点，大量 MEL 样本被误判为 NV（黑色素痣）。

---

## 三、验证集评估汇总

### 文件

| 文件 | 含义 | 注意 |
|------|------|------|
| `val_confusion_matrix.csv` | 混淆矩阵数值表 | 被最后运行的模型覆盖 |
| `val_confusion_matrix.png` | 混淆矩阵可视化图 | 同上 |
| `val_classification_report.csv` | 各类别 precision/recall/f1 | 同上 |
| `val_metrics.json` | 汇总指标 JSON | 同上 |

### 来源

由 `scripts/evaluate.py --split val` 生成。所有指标基于最佳 checkpoint 的 epoch 计算。

### JSON 指标字段

```json
{
  "accuracy": 0.855,
  "macro_precision": 0.770,
  "macro_recall": 0.800,
  "macro_f1": 0.781,
  "weighted_precision": 0.863,
  "weighted_recall": 0.855,
  "weighted_f1": 0.857,
  "num_examples": 1998,
  "num_classes": 7
}
```

- **macro_***: 七类平均（平等对待每类，对小样本类别敏感）
- **weighted_***: 按各类别样本数加权（受 NV 主导）

---

## 四、测试集评估汇总

### 来源

由 `scripts/evaluate.py --config configs/convnext_tiny.yaml --split test` 生成，仅对最佳模型 ConvNeXt-Tiny 运行。

### 文件

| 文件 | 含义 |
|------|------|
| `test_confusion_matrix.csv` | 测试集混淆矩阵 |
| `test_confusion_matrix.png` | 测试集混淆矩阵图 |
| `test_classification_report.csv` | 测试集各类别 precision/recall/f1 |
| `test_metrics.json` | 测试集指标汇总 JSON |

### 核心结论

测试集 1511 张图像（缺 `ISIC_0035068`），结果用于最终报告中的 test performance，未参与模型选择或超参调优。

---

## 五、错误分析 (error_analysis/)

### 来源

由 `scripts/analyze_errors.py --predictions outputs/predictions_convnext_tiny.csv` 生成，基于 ConvNeXt-Tiny 验证集预测结果。

### 文件

| 文件 | 含义 |
|------|------|
| `top_correct.csv` | 置信度最高且预测正确的 top-10 样本 |
| `top_failed.csv` | 置信度最高但预测错误的 top-10 样本（模型最自信的误判） |
| `most_common_confusions.csv` | 最常见的类别混淆对及频次 |

### 核心结论

- 最高频混淆：**MEL → NV**（黑色素瘤被误判为黑色素痣），这是临床风险最高的误判类型。
- **BKL → NV** 混淆也频繁出现，反映了良恶性角质病变之间的视觉相似性。

---

## 六、Grad-CAM 热力图 (gradcam/)

### 来源

由 `scripts/run_gradcam.py` 对每个模型生成 5 张失败样本的 Grad-CAM 热力图。

### 文件命名规则

```
{case}_{序号}_{original|gradcam}.jpg
```

- `original`: 原始皮肤镜图像
- `gradcam`: 叠加了热力图的版本，红色区域表示模型决策时最关注的部位

### 核心结论

- **Baseline CNN**: 注意力分散，关注区域不聚焦于病灶本身。
- **ResNet18**: 注意力集中在病灶区域，但部分样本关注到背景或毛发。
- **ConvNeXt-Tiny**: 注意力最聚焦，基本落在病灶核心区域，解释性最好。

> **注意**: 由于输出目录固定，仅保留最后运行模型的三组图像。如需保留每个模型独立的 Grad-CAM 图，可修改 `--output-dir` 参数。

---

## 七、实验汇总图 (figures/)

### 来源

由 `scripts/plot_visualizations.py` 从 history CSV 和 predictions CSV 生成。

### 文件

| 文件 | 含义 | 关键发现 |
|------|------|---------|
| `loss_curves.png` | 三模型 train/val loss 曲线 | ConvNeXt 收敛最快但过拟合最明显 |
| `accuracy_curves.png` | 三模型 train/val accuracy 曲线 | ResNet18 受 ES 保护，ConvNeXt 未收敛完全 |
| `parameter_comparison.png` | 参数量柱状图 | Baseline ~0.1M, ResNet18 ~11.2M, ConvNeXt ~27.8M |
| `per_class_f1.png` | 类别级 F1 分组柱状图 | DF 类从 CNN 0.15 跃升到 ConvNeXt 0.83 |
| `overfitting_gap.png` | train−val F1 gap 曲线 | CNN 贴近零（欠拟合），ConvNeXt 增长最快 |

---

## 复现命令

```bash
# 训练后评估
python scripts/evaluate.py --config configs/baseline_cnn.yaml --split val
python scripts/evaluate.py --config configs/resnet18.yaml --split val
python scripts/evaluate.py --config configs/convnext_tiny.yaml --split val

# 测试集评估
python scripts/evaluate.py --config configs/convnext_tiny.yaml --split test

# 错误分析
python scripts/analyze_errors.py --predictions outputs/predictions_convnext_tiny.csv

# Grad-CAM
python scripts/run_gradcam.py --config configs/baseline_cnn.yaml --predictions-csv outputs/predictions_baseline.csv --case failed --num-samples 5
python scripts/run_gradcam.py --config configs/resnet18.yaml --predictions-csv outputs/predictions_resnet18.csv --case failed --num-samples 5
python scripts/run_gradcam.py --config configs/convnext_tiny.yaml --predictions-csv outputs/predictions_convnext_tiny.csv --case failed --num-samples 5

# 汇总图
python scripts/plot_visualizations.py
```
