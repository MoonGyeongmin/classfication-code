import logging
import re
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

HMC_RE = re.compile(r"^\d{3}$")
ATC_RE = re.compile(r"^[0-9A-HJ-NP-Z]$")
WUC_RE = re.compile(r"^[0-9A-Z]{5}$")
ILLEGAL_XLSX = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def read_table(path, sheet=None):
    """CSV/엑셀을 모두 문자열로 읽는다. 숫자로 읽히면 HMC 앞자리 0이 사라지므로 dtype=str이 필수다."""
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        df = pd.read_excel(path, dtype=str, sheet_name=sheet or 0)
    else:
        df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    return df.fillna("")


def write_table(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".xlsx":
        # 엑셀이 허용하지 않는 제어문자가 원문에 섞여 있으면 저장이 실패하므로 지운다.
        df = df.apply(lambda c: c.map(lambda v: ILLEGAL_XLSX.sub("", v) if isinstance(v, str) else v))
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False, encoding="utf-8-sig")


def norm_hmc(v):
    v = str(v).strip()
    if v.endswith(".0"):
        v = v[:-2]
    return v.zfill(3) if v.isdigit() and len(v) <= 3 else v


def norm_atc(v):
    return str(v).strip().upper()


def norm_wuc(v):
    return str(v).strip().upper().replace(" ", "").replace("-", "")


def standardize(df, cols):
    """설정의 컬럼 매핑으로 표준 컬럼명(id, aircraft, defect, action, part_number, hmc, atc, wuc)을 만든다."""
    out = pd.DataFrame(index=df.index)
    for key, src in cols.items():
        out[key] = df[src].astype(str).str.strip() if src and src in df.columns else ""
    if (out["id"] == "").all():
        out["id"] = [str(i) for i in range(len(out))]
    if "lang" in df.columns:
        out["lang"] = df["lang"]
    out["hmc"] = out["hmc"].map(norm_hmc)
    out["atc"] = out["atc"].map(norm_atc)
    out["wuc"] = out["wuc"].map(norm_wuc)
    return out


def build_text(row):
    parts = []
    if row.get("aircraft"):
        parts.append(f"[기종] {row['aircraft']}")
    if row.get("defect"):
        parts.append(f"[결함] {row['defect']}")
    if row.get("action"):
        parts.append(f"[조치] {row['action']}")
    if row.get("part_number"):
        parts.append(f"[PN] {row['part_number']}")
    return " ".join(parts)


def add_text(df):
    df = df.copy()
    df["text"] = [build_text(r) for r in df.to_dict("records")]
    return df


def filter_labels(df, codebook):
    """정의에 없는 HMC/ATC 라벨이 붙은 행은 학습에서 뺀다. 빠진 건수와 코드는 로그로 남긴다."""
    ok_hmc = df["hmc"].isin(codebook.hmc.keys())
    ok_atc = df["atc"].isin(codebook.atc.keys())
    bad = df[~(ok_hmc & ok_atc)]
    if len(bad):
        log.warning("코드표에 없는 라벨 %d건 제외. HMC 예: %s / ATC 예: %s", len(bad),
                    bad.loc[~ok_hmc, "hmc"].value_counts().head(10).to_dict(),
                    bad.loc[~ok_atc, "atc"].value_counts().head(10).to_dict())
    return df[ok_hmc & ok_atc].reset_index(drop=True)
