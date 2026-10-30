"""주기 실행 결과 요약: 기종별 코드 분포, 검토 필요 비율."""
import pandas as pd


def summarize(result, aircraft_col):
    sheets = {}
    by = [aircraft_col] if aircraft_col in result.columns else []
    sheets["개요"] = pd.DataFrame({
        "항목": ["전체 건수", "검토 필요 건수", "검토 필요 비율"],
        "값": [len(result), int(result["review_needed"].sum()),
              round(float(result["review_needed"].mean()), 4) if len(result) else 0],
    })
    for code in ("hmc", "atc", "wuc"):
        col = f"pred_{code}"
        if col not in result.columns:
            continue
        g = result.groupby(by + [col]).size().rename("건수").reset_index()
        desc = f"pred_{code}_desc" if code != "wuc" else "pred_wuc_name"
        if desc in result.columns:
            g = g.merge(result[[col, desc]].drop_duplicates(col), on=col, how="left")
        sheets[f"{code.upper()} 분포"] = g.sort_values(by + ["건수"], ascending=[True] * len(by) + [False])
    return sheets


def write_summary(result, aircraft_col, path):
    with pd.ExcelWriter(path) as w:
        for name, df in summarize(result, aircraft_col).items():
            df.to_excel(w, sheet_name=name, index=False)
