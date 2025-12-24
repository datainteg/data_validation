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


 IF (f26<1.5 OR f26 = .) THEN DO;
    IF (f8<0.5 OR f8 = .) THEN DO;
        IF (f233<1.5 OR f233 = .) THEN DO;
            IF (f616<2.5 OR f616 = .) THEN DO;
                IF (f232<15.5 OR f232 = .) THEN DO;
                    sum_score = sum_score + (-0.141490221);
                END;
                ELSE DO;
                    sum_score = sum_score + (-0.1768962);
                END;
            END;
            ELSE DO;
                IF (f88<3.5 OR f88 = .) THEN DO;
                    sum_score = sum_score + (-0.154331714);
                END;
                ELSE DO;
                    sum_score = sum_score + (-0.117181979);
                END;
            END;
        END;
        ELSE DO;
            IF (f31<1.5 OR f31 = .) THEN DO;
                IF (f22<1.5 OR f22 = .) THEN DO;
                    sum_score = sum_score + (-0.184386268);
                END;
                ELSE DO;
                    sum_score = sum_score + (-0.0539518893);
                END;
            END;
            ELSE DO;
                IF (f13<2227.5 OR f13 = .) THEN DO;
                    sum_score = sum_score + (-0.161398217);
                END;
                ELSE DO;
                    sum_score = sum_score + (-0.182989329);
                END;
            END;
        END;
    END;
    ELSE DO;
        IF (f396<19.5 OR f396 = .) THEN DO;
            IF (f13<1662.5 OR f13 = .) THEN DO;
                IF (f124<0.5 OR f124 = .) THEN DO;
                    sum_score = sum_score + (0.0328746177);
                END;
                ELSE DO;
                    sum_score = sum_score + (-0.0433662049);
                END;
            END;
            ELSE DO;
                IF (f223<6.5 OR f223 = .) THEN DO;
                    sum_score = sum_score + (-0.00423728814);
                END;
                ELSE DO;
                    sum_score = sum_score + (-0.116010062);
                END;
            END;
        END;
        ELSE DO;
            IF (f616<2.5 OR f616 = .) THEN DO;
                IF (f13<4624.5 OR f13 = .) THEN DO;
                    sum_score = sum_score + (-0.135549352);
                END;
                ELSE DO;
                    sum_score = sum_score + (-0.161402315);
                END;
            END;
            ELSE DO;
                IF (f88<3.5 OR f88 = .) THEN DO;
                    sum_score = sum_score + (-0.116239391);
                END;
                ELSE DO;
                    sum_score = sum_score + (-0.0431666225);
                END;
            END;
        END;
    END;
END;
ELSE DO;
    IF (f8<1.5 OR f8 = .) THEN DO;
        IF (f25<41 OR f25 = .) THEN DO;
            IF (f88<3.5 OR f88 = .) THEN DO;
                IF (f8<0.5 OR f8 = .) THEN DO;
                    sum_score = sum_score + (-0.100159876);
                END;
                ELSE DO;
                    sum_score = sum_score + (-0.0562722199);
                END;
            END;
            ELSE DO;
                IF (f616<2.5 OR f616 = .) THEN DO;
                    sum_score = sum_score + (-0.0709036812);
                END;
                ELSE DO;
                    sum_score = sum_score + (-0.00515068509);
                END;
            END;
        END;
        ELSE DO;
            IF (f425<20.5 OR f425 = .) THEN DO;
                sum_score = sum_score + (-0.0940476209);
            END;
            ELSE DO;
                IF (f247<1778 OR f247 = .) THEN DO;
                    sum_score = sum_score + (0.00751879672);
                END;
                ELSE DO;
                    sum_score = sum_score + (0.0706896558);
                END;
            END;
        END;
    END;
    ELSE DO;
        IF (f37<5.5 OR f37 = .) THEN DO;
            IF (f25<19 OR f25 = .) THEN DO;
                IF (f616<2.5 OR f616 = .) THEN DO;
                    sum_score = sum_score + (0.106128551);
                END;
                ELSE DO;
                    sum_score = sum_score + (0.15152514);
                END;
            END;
            ELSE DO;
                IF (f396<44.5 OR f396 = .) THEN DO;
                    sum_score = sum_score + (0.113057196);
                END;
                ELSE DO;
                    sum_score = sum_score + (0.0571751408);
                END;
            END;
        END;
        ELSE DO;
            IF (f395<35.5 OR f395 = .) THEN DO;
                IF (f616<2.5 OR f616 = .) THEN DO;
                    sum_score = sum_score + (0.0497482382);
                END;
                ELSE DO;
                    sum_score = sum_score + (0.111547343);
                END;
            END;
            ELSE DO;
                IF (f7<409.49 OR f7 = .) THEN DO;
                    sum_score = sum_score + (-0.148760334);
                END;
                ELSE DO;
                    sum_score = sum_score + (0.0210886467);
                END;
            END;
        END;
    END;
END;

