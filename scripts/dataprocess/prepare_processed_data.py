"""Prepare repo-local processed dataset layout from extracted HAM10000 sources."""

import argparse
import shutil
from pathlib import Path

import pandas as pd


DX_TO_LABEL = {
    "mel": "MEL",
    "nv": "NV",
    "bcc": "BCC",
    "akiec": "AKIEC",
    "bkl": "BKL",
    "df": "DF",
    "vasc": "VASC",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kaggle-dir",
        type=str,
        default="data/raw/_downloads/ham10000_kaggle",
    )
    parser.add_argument(
        "--testset-dir",
        type=str,
        default="data/raw/_downloads/ham10000_testset/TestSet",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/processed",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="move",
        choices=["move", "copy"],
        help="How to transfer files into processed directory.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any image referenced by metadata is missing.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    return parser.parse_args()


def _transfer(src: Path, dst: Path, *, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "move":
        shutil.move(str(src), str(dst))
        return
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
        return
    shutil.copy2(src, dst)


def _safe_rmtree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def build_ham10000_metadata(src_kaggle_dir: Path, dst_ham_dir: Path) -> pd.DataFrame:
    raw_meta_path = src_kaggle_dir / "HAM10000_metadata.csv"
    df = pd.read_csv(raw_meta_path)

    img_dirs = [
        dst_ham_dir / "images" / "HAM10000_images_part_1",
        dst_ham_dir / "images" / "HAM10000_images_part_2",
    ]
    image_map: dict[str, str] = {}
    for d in img_dirs:
        for p in d.glob("*.jpg"):
            image_map[p.stem] = str(p.relative_to(dst_ham_dir).as_posix())

    missing = [img_id for img_id in df["image_id"].tolist() if img_id not in image_map]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} images referenced by metadata.csv.")

    df = df.copy()
    df["label"] = df["dx"].map(lambda x: DX_TO_LABEL.get(str(x).lower()))
    if df["label"].isna().any():
        unknown = sorted(set(df.loc[df["label"].isna(), "dx"].astype(str).tolist()))
        raise RuntimeError(f"Unknown dx labels in metadata: {unknown}")

    df["image_path"] = df["image_id"].map(lambda x: image_map[str(x)])

    cols = [
        "image_id",
        "label",
        "image_path",
        "lesion_id",
        "dx",
        "dx_type",
        "age",
        "sex",
        "localization",
    ]
    return df[cols]


def build_testset_metadata(
    src_testset_dir: Path, dst_test_dir: Path, *, strict: bool
) -> tuple[pd.DataFrame, list[str]]:
    raw_gt = src_testset_dir / "ISIC2018_Task3_Test_GroundTruth.csv"
    df = pd.read_csv(raw_gt).copy()
    df["label"] = df["dx"].map(lambda x: DX_TO_LABEL.get(str(x).lower()))
    if df["label"].isna().any():
        unknown = sorted(set(df.loc[df["label"].isna(), "dx"].astype(str).tolist()))
        raise RuntimeError(f"Unknown dx labels in groundtruth: {unknown}")

    img_dir = dst_test_dir / "images"
    image_map = {p.stem: str(p.relative_to(dst_test_dir).as_posix()) for p in img_dir.glob("*.jpg")}
    missing = [img_id for img_id in df["image_id"].tolist() if img_id not in image_map]
    if missing:
        if strict:
            raise RuntimeError(
                f"Missing {len(missing)} test images referenced by groundtruth.csv."
            )
        df = df.loc[~df["image_id"].isin(missing)].copy()

    df["image_path"] = df["image_id"].map(lambda x: image_map[str(x)])

    cols = [
        "image_id",
        "label",
        "image_path",
        "lesion_id",
        "dx",
        "dx_type",
        "age",
        "sex",
        "localization",
        "dataset",
    ]
    return df[cols], missing


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]

    src_kaggle_dir = (repo_root / args.kaggle_dir).resolve()
    src_testset_dir = (repo_root / args.testset_dir).resolve()
    out_dir = (repo_root / args.out_dir).resolve()

    dst_ham_dir = out_dir / "ham10000"
    dst_test_dir = out_dir / "testset"

    if args.force:
        _safe_rmtree(dst_ham_dir)
        _safe_rmtree(dst_test_dir)

    (dst_ham_dir / "images").mkdir(parents=True, exist_ok=True)
    dst_test_dir.mkdir(parents=True, exist_ok=True)

    part1 = src_kaggle_dir / "HAM10000_images_part_1"
    part2 = src_kaggle_dir / "HAM10000_images_part_2"
    raw_meta = src_kaggle_dir / "HAM10000_metadata.csv"
    if not part1.exists() or not part2.exists() or not raw_meta.exists():
        raise FileNotFoundError("Expected Kaggle extracted structure not found.")

    _transfer(part1, dst_ham_dir / "images" / part1.name, mode=args.mode)
    _transfer(part2, dst_ham_dir / "images" / part2.name, mode=args.mode)

    test_img_src = src_testset_dir / "ISIC2018_Task3_Test_Images"
    test_gt_src = src_testset_dir / "ISIC2018_Task3_Test_GroundTruth.csv"
    if not test_img_src.exists() or not test_gt_src.exists():
        raise FileNotFoundError("Expected TestSet extracted structure not found.")

    _transfer(test_img_src, dst_test_dir / "images", mode=args.mode)
    _transfer(test_gt_src, dst_test_dir / "groundtruth.csv", mode=args.mode)

    ham_df = build_ham10000_metadata(src_kaggle_dir, dst_ham_dir)
    ham_df.to_csv(dst_ham_dir / "metadata.csv", index=False, encoding="utf-8")

    test_df, missing = build_testset_metadata(src_testset_dir, dst_test_dir, strict=args.strict)
    test_df.to_csv(dst_test_dir / "metadata.csv", index=False, encoding="utf-8")
    if missing:
        (dst_test_dir / "missing_images.txt").write_text(
            "\n".join(missing) + "\n", encoding="utf-8"
        )
        print(f"Warning: missing {len(missing)} test images, list saved to missing_images.txt")

    print("Processed data prepared:")
    print("-", dst_ham_dir)
    print("-", dst_test_dir)


if __name__ == "__main__":
    main()
