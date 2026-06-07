# Final Report: Skin Lesion Classification on HAM10000

## 1 项目目标
本项目基于 ISIC 2018 Task 3 / HAM10000 数据集，实现并对比多个深度学习模型，完成皮肤病灶七分类任务。

## 2 数据集与预处理
- 数据来源：HAM10000（Kaggle）/ ISIC 2018 Task 3 测试集
- 类别：NV, MEL, BKL, BCC, AKIEC, VASC, DF
- 数据预处理：ImageNet 标准化 + 随机翻转/旋转/颜色抖动（训练）

## 3 实验设计与模型
- Baseline CNN：简单卷积网络
- 迁移学习模型：ResNet18（ImageNet 预训练）
- 迁移学习模型：ConvNeXt-Tiny（ImageNet 预训练）
- 分割策略：`Lesion-level stratified group split`（按 lesion_id 分组，确保同一病灶不出现在 train/val 两侧）
- 类别不平衡处理：class weights

## 4 结果与分析
- 训练历史（loss/accuracy/macro-F1）
- 验证集和测试集 metrics（accuracy / precision / recall / macro-F1 / weighted-F1）
- Confusion matrix
- 错误分析（最高置信度错误、最常见混淆对）

## 5 模型解释性
- Grad-CAM 热力图分析（成功/失败样本各取 top-5 高置信度）
- 关注区域分析

## 6 讨论
- 类别不平衡的影响
- 过拟合与数据增强的作用
- 泛化能力分析（val vs. test 对比）
- 模型局限性与改进方向
