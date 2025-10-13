#!/usr/bin/env python3
# Overwrite ket_qua_cuoi_part_001..055.csv by removing rows whose gender is Female (or variants).
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import shutil

def find_gender_col(df: pd.DataFrame):
    for c in df.columns:
        if c.lower() == "gender":
            return c
    # fallback: try any column containing 'gender'
    for c in df.columns:
        if "gender" in c.lower():
            return c
    return None

def normalize(s):
    if pd.isna(s):
        return ""
    return str(s).strip().lower()

def is_female_variant(s: str):
    # treat anything that starts with 'f' as female-ish (covers 'female', 'felmail', 'femal', 'f', etc.)
    return s.startswith("f")

def process_file(path: Path, inplace: bool = True, backup: bool = False, dry_run: bool = False):
    df = pd.read_csv(path, encoding='utf-8', dtype=str)
    gender_col = find_gender_col(df)
    if gender_col is None:
        return {"file": path.name, "status": "no_gender_col", "in": len(df), "out": len(df)}
    genders = df[gender_col].fillna("").map(normalize)
    mask_female = genders.map(is_female_variant)
    kept = df[~mask_female].copy()
    if not dry_run:
        if inplace:
            if backup:
                bak = path.with_suffix(path.suffix + ".bak")
                shutil.copy2(path, bak)
            kept.to_csv(path, index=False, encoding='utf-8')
        else:
            out_dir = path.parent / "male_only"
            out_dir.mkdir(parents=True, exist_ok=True)
            kept.to_csv(out_dir / path.name, index=False, encoding='utf-8')
    return {"file": path.name, "status": "ok", "in": len(df), "out": len(kept), "removed": int(mask_female.sum())}

def main():
    p = argparse.ArgumentParser(description="Filter out female rows from ket_qua_cuoi_part_001..055.csv and overwrite.")
    p.add_argument("--dir", "-d", default="d:/ViSpeech/super_metadata", help="Folder with part CSVs")
    p.add_argument("--start", type=int, default=1)
    p.add_argument("--end", type=int, default=55)
    p.add_argument("--backup", action="store_true", help="Keep .bak copy before overwriting")
    p.add_argument("--dry-run", action="store_true", help="Do not write files, just report")
    args = p.parse_args()

    base = Path(args.dir)
    results = []
    for i in tqdm(range(args.start, args.end + 1), desc="Files"):
        fname = f"ket_qua_cuoi_part_{i:03d}.csv"
        path = base / fname
        if not path.exists():
            results.append({"file": fname, "status": "missing"})
            continue
        r = process_file(path, inplace=True, backup=args.backup, dry_run=args.dry_run)
        results.append(r)

    # brief summary
    processed = [r for r in results if r.get("status") == "ok"]
    missing = [r for r in results if r.get("status") == "missing"]
    no_col = [r for r in results if r.get("status") == "no_gender_col"]
    total_in = sum(r.get("in",0) for r in processed)
    total_out = sum(r.get("out",0) for r in processed)
    total_removed = sum(r.get("removed",0) for r in processed)
    print()
    print(f"Processed {len(processed)} files, missing {len(missing)}, no_gender_col {len(no_col)}")
    print(f"Rows in: {total_in}, out: {total_out}, removed: {total_removed}")
    if args.dry_run:
        print("Dry-run: no files were written.")

if __name__ == "__main__":
    main()