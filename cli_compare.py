# cli_compare.py
import argparse
from validator_core import CompareConfig, compare_and_render

def main():
    ap = argparse.ArgumentParser(description="CSV/Excel Compare & Validate")
    ap.add_argument("--file1", required=True)
    ap.add_argument("--file2", required=True)
    ap.add_argument("--out", default="validation_output.xlsx")
    ap.add_argument("--keys", help="Comma-separated key columns", default="")
    ap.add_argument("--mismatches-only", action="store_true")
    ap.add_argument("--num-abs", type=float, default=0.0)
    ap.add_argument("--num-rel", type=float, default=0.0)
    ap.add_argument("--date-tol", type=int, default=0)
    args = ap.parse_args()

    cfg = CompareConfig(
        output_mode="mismatches_only" if args.mismatches_only else "all_rows",
        numeric_abs_tol=args.num_abs,
        numeric_rel_tol=args.num_rel,
        date_tolerance_days=args.date_tol,
    )
    keys = [k.strip() for k in args.keys.split(",") if k.strip()] or None

    result = compare_and_render(
        left_source=args.file1, right_source=args.file2, cfg=cfg, explicit_keys=keys
    )
    with open(args.out, "wb") as f:
        f.write(result["excel_bytes"])
    print(f"✅ Wrote: {args.out}")
    if result["summary"]["keys_used"]:
        print("🔑 Keys:", ", ".join(result["summary"]["keys_used"]))
    else:
        print("🔑 Paired by row order")

if __name__ == "__main__":
    main()
