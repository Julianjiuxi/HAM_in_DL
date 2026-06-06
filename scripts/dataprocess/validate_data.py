"""Validate processed dataset integrity."""

import argparse
from pathlib import Path

import pandas as pd


CLASS_NAMES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def _require_columns(df: pd.DataFrame, cols: list[str], *, name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name}: missing columns: {missing}")


def _validate_metadata(meta_path: Path, root_dir: Path, *, strict: bool) -> dict:
    df = pd.read_csv(meta_path)
    _require_columns(df, ["image_id", "label", "image_path"], name=str(meta_path))

    invalid_labels = sorted(set(df["label"].astype(str)) - set(CLASS_NAMES))
    if invalid_labels:
        raise RuntimeError(f"{meta_path}: invalid labels: {invalid_labels}")

    img_paths = [root_dir / Path(p) for p in df["image_path"].astype(str).tolist()]
    missing = [str(p) for p in img_paths if not p.exists()]
    if missing and strict:
        raise RuntimeError(f"{meta_path}: missing {len(missing)} image files")

    return {
        "rows": len(df),
        "missing_images": len(missing),
        "unique_labels": sorted(df["label"].astype(str).unique().tolist()),
    }


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    processed = (repo_root / args.processed_dir).resolve()

    ham_root = processed / "ham10000"
    test_root = processed / "testset"

    ham_meta = ham_root / "metadata.csv"
    test_meta = test_root / "metadata.csv"

    if not ham_meta.exists():
        raise FileNotFoundError(str(ham_meta))
    if not test_meta.exists():
        raise FileNotFoundError(str(test_meta))

    ham_report = _validate_metadata(ham_meta, ham_root, strict=args.strict)
    test_report = _validate_metadata(test_meta, test_root, strict=args.strict)

    missing_list = test_root / "missing_images.txt"
    if missing_list.exists():
        ids = [x.strip() for x in missing_list.read_text(encoding="utf-8").splitlines() if x.strip()]
        if ids:
            print("Known missing TestSet image_ids:", ", ".join(ids))

    print("Validation OK")
    print("HAM10000:", ham_report)
    print("TestSet:", test_report)


if __name__ == "__main__":
    main()

