# Data directory

This folder is intentionally kept mostly empty for GitHub.

## Local layout (suggested)

```text
data/raw/
├── _downloads/
├── train/
├── val/
├── test/
└── metadata.csv
```

Do not commit large image datasets to the repository unless your instructor explicitly requires it.

## Full workflow (download → extract → processed)

本仓库把“数据获取/解压/整理”为独立脚本流程，避免把网络下载逻辑塞进 `Dataset`，也避免把大文件提交到 git。

最省心的方式是用统一入口脚本（推荐）：

```bash
python scripts/setup_data.py --force
```

### Step 0: Put archives under raw/_downloads (git ignored)

课程/手动下载的压缩包建议统一放在 `data/raw/_downloads/` 下（默认被 git 忽略）。例如：

```text
data/raw/_downloads/HAM10000_TestSet.zip
data/raw/_downloads/kagglehub_cache/archive.zip
```

### Step 1: (Optional) Download Kaggle HAM10000 via kagglehub

如果你在服务器或网络环境良好处下载 Kaggle 数据，可运行：

```bash
python scripts/datadownload/download_ham10000_kagglehub.py
```

该脚本会把 kagglehub 缓存目录固定到 `data/raw/_downloads/kagglehub_cache/`（避免到处散落），并输出实际数据路径到：

```text
data/raw/_downloads/kagglehub_ham10000_path.txt
```

### Step 2: Extract archives

解压 Kaggle 训练集与老师给的 TestSet：

```bash
python scripts/datadownload/extract_archives.py --force
```

常用选项：
- 只解压其中一类：`--source kaggle` 或 `--source testset`
- 缺某个压缩包时跳过而不是报错：`--skip-missing`

解压后的目录默认是：

```text
data/raw/_downloads/ham10000_kaggle/
├── HAM10000_images_part_1/
├── HAM10000_images_part_2/
└── HAM10000_metadata.csv

data/raw/_downloads/ham10000_testset/
└── TestSet/
    ├── ISIC2018_Task3_Test_Images/
    └── ISIC2018_Task3_Test_GroundTruth.csv
```

### Step 3: Prepare processed dataset (recommended for training/eval)

将 raw/_downloads 下的“已解压数据”整理成后续训练/评估统一读取的 `data/processed/` 结构：

```bash
python scripts/dataprocess/prepare_processed_data.py --mode copy --force
```

说明：
- `--mode move`：把目录从 `raw/_downloads` 直接挪到 `processed`（节省空间）
- `--mode copy`：复制一份（更安全但占用空间更大）

输出目录结构：

```text
data/processed/
├── ham10000/
│   ├── images/
│   │   ├── HAM10000_images_part_1/*.jpg
│   │   └── HAM10000_images_part_2/*.jpg
│   └── metadata.csv
└── testset/
    ├── images/*.jpg
    ├── groundtruth.csv
    ├── metadata.csv
    └── missing_images.txt        # 若存在缺失图片，会在这里列出
```

`metadata.csv` 统一包含以下核心列（便于脚本/训练代码一致读取）：
- `image_id`
- `label`（映射到 7 类：MEL/NV/BCC/AKIEC/BKL/DF/VASC）
- `image_path`（相对 `data/processed/{ham10000|testset}` 的路径）

### Step 4: Validate processed data (recommended)

```bash
python scripts/dataprocess/validate_data.py
```

## Known issue: one missing TestSet image

当前老师给的 `HAM10000_TestSet.zip` 中，`ISIC2018_Task3_Test_GroundTruth.csv` 有 1512 行，但 `ISIC2018_Task3_Test_Images/` 只有 1511 张 jpg；缺失的 `image_id` 为：

```text
ISIC_0035068
```

已验证该文件在 zip 内也不存在，因此更像是“压缩包本身缺文件/groundtruth 多了一行”，而不是解压损坏或下载不全。

默认 `prepare_processed_data.py` 会：
- 输出 `missing_images.txt`
- 从 `data/processed/testset/metadata.csv` 中剔除缺失项，让 pipeline 先跑通

如需严格校验（缺一张就报错中止），使用：

```bash
python scripts/dataprocess/prepare_processed_data.py --mode copy --force --strict
```
