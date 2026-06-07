# 2025-06-07 — 数据流水线搭建与跨分支代码整合

- **Date:** 2025-06-07
- **Tool:** Trae IDE (Claude)
- **Purpose:** 从零搭建 HAM10000 数据流水线，完成组员分支审查与选择性迁移，修复迁移后接口不一致与泄漏等问题。

---

## 关键交互

### 数据流水线
- 下载 Kaggle HAM10000 + 解压老师 TestSet
- 建成 `raw/_downloads → extracted → processed` 三层隔离
- 统一入口 `scripts/setup_data.py`（extract → prepare → validate）
- split 按 `lesion_id` 分组，消除 train/val 泄漏
- 缺失图片 `ISIC_0035068` 记录到 `missing_images.txt`，默认剔除

### 组员分支审查与迁移
- 审查 `codex/complete-dl-pipeline`（29 文件，1500+ 行改动）
- 选择性迁移到 main：Dataset、训练循环、评估指标、Grad-CAM、测试
- 不直接 merge，逐文件整合并修正设计问题

### 后续修复
- `model_factory` 接口统一为 `build_model(name, *, num_classes, ...)`
- Grad-CAM target layer 修正（`backbone.layer4` → `layer4`）
- `freeze_backbone` 显式只冻 backbone 不冻分类头
- 所有训练/评估脚本从 config 读取 `pretrained` / `freeze_backbone`
- 新增 ConvNeXt-Tiny 模型（~28.6M，三文件改动即可注册）

## 验证方式
- `validate_data.py` 通过：metadata 与图片一致性、标签合法性、类别分布
- `make_splits.py` 通过：lesion-level split 无重叠
- `pytest` 失败原因确认为本机 PyTorch DLL 环境问题，非代码缺陷
