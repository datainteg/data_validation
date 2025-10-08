# app_streamlit.py
import streamlit as st
from validator_core import CompareConfig, compare_and_render

st.set_page_config(page_title="CSV/Excel Comparator", layout="wide")

st.title("🔍 CSV/Excel Compare & Validate")

left_file = st.file_uploader("Upload File 1 (CSV/XLSX)", type=["csv", "xlsx", "xls"], key="f1")
right_file = st.file_uploader("Upload File 2 (CSV/XLSX)", type=["csv", "xlsx", "xls"], key="f2")

with st.expander("⚙️ Options"):
    col1, col2, col3 = st.columns(3)
    with col1:
        case_ins = st.checkbox("Case-insensitive headers", value=True)
        strip_ws = st.checkbox("Trim whitespace", value=True)
        output_mode = st.selectbox("Output mode", ["all_rows", "mismatches_only"], index=0)
        placeholder = st.text_input("Missing placeholder", value="N/A")
    with col2:
        num_abs = st.number_input("Numeric abs tolerance", min_value=0.0, value=0.0, step=0.0001, format="%.6f")
        num_rel = st.number_input("Numeric rel tolerance", min_value=0.0, value=0.0, step=0.0001, format="%.6f")
        date_tol = st.number_input("Date tolerance (days)", min_value=0, value=0, step=1)
        treat_empty = st.checkbox("Treat empty as missing", value=True)
    with col3:
        detect_numbers = st.checkbox("Detect numeric columns", value=True)
        detect_dates = st.checkbox("Detect date columns", value=True)
        max_combo = st.slider("Max key columns (auto)", min_value=1, max_value=3, value=2)
        uniq_thr = st.slider("Key uniqueness threshold", min_value=0.70, max_value=0.99, value=0.90, step=0.01)

    explicit_keys_text = st.text_input("Explicit key columns (comma-separated). Leave blank for auto.", value="")
    explicit_keys = [c.strip() for c in explicit_keys_text.split(",") if c.strip()] or None

cfg = CompareConfig(
    case_insensitive_headers=case_ins,
    strip_whitespace=strip_ws,
    placeholder=placeholder,
    numeric_abs_tol=num_abs,
    numeric_rel_tol=num_rel,
    date_tolerance_days=date_tol,
    treat_empty_as_missing=treat_empty,
    output_mode=output_mode,
    max_key_combo=max_combo,
    key_uniqueness_threshold=uniq_thr,
    detect_dates=detect_dates,
    detect_numbers=detect_numbers
)

run = st.button("Compare & Generate")

if run:
    if not left_file or not right_file:
        st.warning("Please upload both files.")
    else:
        with st.spinner("Processing..."):
            try:
                result = compare_and_render(
                    left_source=left_file,
                    right_source=right_file,
                    cfg=cfg,
                    explicit_keys=explicit_keys
                )
                summary = result["summary"]
                st.success("Completed!")

                # Summary cards
                with st.container():
                    st.write("**Keys used:**", ", ".join(summary["keys_used"]) if summary["keys_used"] else "Row order")
                    st.write("**Numeric columns:**", ", ".join(summary["numeric_cols"]) or "None")
                    st.write("**Date columns:**", ", ".join(summary["date_cols"]) or "None")
                    st.write("**Preview rows:**", summary["preview_count"])

                st.dataframe(result["preview_df"], use_container_width=True)

                st.download_button(
                    label="📥 Download Excel (validation_output.xlsx)",
                    data=result["excel_bytes"],
                    file_name="validation_output.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            except Exception as e:
                st.error(f"Error: {e}")
