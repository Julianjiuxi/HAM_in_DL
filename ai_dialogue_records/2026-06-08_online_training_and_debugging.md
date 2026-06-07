# 2026-06-08 — 在线训练、GPU 调优、Bug 修复与实验结果分析

- **Date:** 2026-06-08
- **Tool:** Trae IDE (Claude)
- **Purpose:** 在 Autodl 服务器上完成三个模型的完整训练流程，排查并修复训练与可视化中的 bug，调优 GPU 利用率，分析实验结果。

---

## 关键交互

### 1. 数据搬运与准备工作
- `archive.zip` 已上传至仓库根目录，需要按照 `setup_data.py` 的规划归位
- 修改 `extract_archives.py`，新增 `stage_kaggle_archive()` 函数：自动检测根目录下的 `archive.zip` 并将其移动到 `data/raw/_downloads/kagglehub_cache/archive.zip`
- 运行 `setup_data.py --force` 完成数据解压、整理、校验全流程
- 验证：HAM10000 10015 张图像全部到位、TestSet 1511 张（缺 1 张 `ISIC_0035068` 已自动剔除）

### 2. 完整流程梳理
- 梳理了从 `setup_data.py → make_splits.py → train_*.py → evaluate.py → analyze_errors.py → run_gradcam.py` 的全套命令
- 确认三个模型（Baseline CNN、ResNet18、ConvNeXt-Tiny）共用同一套数据与 lesion-level split
- 修复 `train_transfer.py` 中 checkpoint 路径硬编码问题：`resnet18_{seed}.pt` → `{model_name}_{seed}.pt`

### 3. GPU 利用率调优
- 问题：48GB vGPU 上 batch_size=32、num_workers=0 时显存仅用 2.3GB，GPU 利用率 10%
- 诊断：数据加载为单线程串行 + batch 过小，GPU 大量空等
- 修改三个 config 文件：`baseline_cnn` batch_size 32→256, `resnet18` 32→128, `convnext_tiny` 32→128，全部 num_workers 0→4
- 效果：显存占用升至 ~15GB，GPU 利用率拉满

### 4. Git 分支管理与 SSH 配置
- 在 `online_training` 分支上推送服务器调优与修复
- 服务器 443 端口被封导致 HTTPS push 超时，配置 SSH key + ed25519 后成功推送至 GitHub
- 确认训练进程不受 git 操作影响

### 5. 代码风险审查与修复
- 审查 `online_training` 分支 4 个风险点：
  - **风险 1（已修复）**：Baseline CNN 无 `layer4`/`conv3` 属性，Grad-CAM target layer 会崩 → 改为 `model.features[6]`
  - **风险 2（已修复）**：ConvNeXt Grad-CAM target layer 改为 `model.features[-1][-1].block[5]`（最后 depthwise conv）
  - **风险 3（非阻塞）**：outputs 目录缺乏模型名前缀，后跑模型会覆盖前者的 `val_metrics.json` 等
  - **风险 4（当前不触发）**：`normalise_kagglehub_download()` 文件拍平会导致后续 `prepare_processed_data.py` 报错，但服务器走 `archive.zip` 路径不受影响
- 确认 `build_loss`、`build_model`、`evaluate.py` import 等接口均已修好

### 6. Grad-CAM Bug 修复
- `analyze_errors.py` 缺少 `sys.path` 注入，导致 `ModuleNotFoundError` → 补上 4 行
- Baseline CNN 的 `ReLU(inplace=True)` 与 Grad-CAM backward hook 冲突，报 `view/inplace` RuntimeError → 在 `run_gradcam.py` 加载模型后禁用所有 inplace ReLU（不改变权重）

### 7. 实验结果分析
- Baseline CNN：Val Macro-F1 37.77%，严重欠拟合，NV 偏向严重（precision 92% recall 58%）
- ResNet18：Val Macro-F1 71.12%（epoch 13 best），early stopping 在第 20 epoch 精准触发，过拟合 gap ~18%
- ConvNeXt-Tiny：Val Macro-F1 78.10%（epoch 20 best），train-val gap ~19%，epoch 20 后 val 明显恶化
- 三个模型构成清晰性能梯度，MEL 是所有模型共同的弱点（大量 MEL → NV 混淆）
- 无参数塌陷、梯度爆炸等问题，无需重训

### 8. 代码实验部分收尾确认
- 全部产物到位：3 个 checkpoint、3 份 history CSV、3 份 predictions CSV、test 评估、错误分析、Grad-CAM 热力图
- 确认后续只需写报告和打包提交

## 验证方式
- `setup_data.py --force` 通过：validate_data 校验 metadata 与图像一致
- `make_splits.py --force` 通过：train/val 10015 行、20% 划分、0 lesion overlap
- 三个模型 `evaluate.py --split val` 全部通过，产出完整指标
- `evaluate.py --split test` 通过，产出 test 混淆矩阵与 JSON
- `analyze_errors.py` 通过，产出 top correct/failed/confusions
- `run_gradcam.py` 三次通过，产出三组失败样本热力图
