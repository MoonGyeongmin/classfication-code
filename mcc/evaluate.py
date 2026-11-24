"""mcc evaluate: 사람이 정답(HMC/ATC)을 붙인 작은 평가 파일로 학습된 모델을 채점한다.

외국어 현지 기록 수십~수백 건에 정답을 달아 두면, 번역 증강 없이 학습한 모델과 증강해서 학습한 모델을
같은 파일로 채점해 비교할 수 있다. 파일에 lang 컬럼(en, pl, th ...)이 있으면 언어별로, 기종 컬럼이 있으면
기종별로도 점수를 낸다. 결과는 엑셀(요약 시트 + 건별 시트)과 화면 표로 나온다.
"""
import logging
from pathlib import Path

import pandas as pd

from .codebook import Codebook
from .data import add_text, filter_labels, read_table, require_columns, standardize, write_table
from .errors import MccError
from .predict import load_model, predict_df

log = logging.getLogger(__name__)


def score(g, k):
    hmc_ok = g["pred_hmc"] == g["hmc"]
    atc_ok = g["pred_atc"] == g["atc"]
    top = lambda name: [gold in {x.split(":")[0].strip() for x in s.split("|")}  # noqa: E731
                        for gold, s in zip(g[name], g[f"{name}_top{k}"])]
    return {"건수": len(g),
            "HMC 정확도": round(hmc_ok.mean(), 4), f"HMC top{k}": round(pd.Series(top("hmc")).mean(), 4),
            "ATC 정확도": round(atc_ok.mean(), 4), f"ATC top{k}": round(pd.Series(top("atc")).mean(), 4),
            "HMC+ATC 모두 맞음": round((hmc_ok & atc_ok).mean(), 4),
            "검토 필요 비율": round(g["review_needed"].mean(), 4)}


def run(cfg, input_path, output_path=None, model_dir=None):
    codebook = Codebook.load(cfg["codebook_dir"])
    model, tokenizer, meta, device, model_dir = load_model(cfg, model_dir)
    cols = cfg["columns"]["data"]
    input_path = Path(input_path)
    raw = read_table(input_path)
    require_columns(raw, cols, ["hmc", "atc"], input_path.name, any_of=("defect", "action"))
    df = filter_labels(add_text(standardize(raw, cols)), codebook)
    if df.empty:
        raise MccError("E36", f"{input_path.name}: 전체 {len(raw)}건")
    if "lang" not in df.columns:
        df["lang"] = ""

    k = cfg["predict"]["top_k"]
    out = predict_df(df, model, tokenizer, codebook, cfg, device)
    out["hmc_맞음"] = out["pred_hmc"] == out["hmc"]
    out["atc_맞음"] = out["pred_atc"] == out["atc"]

    rows = [{"구분": "전체", "값": "", **score(out, k)}]
    for by, name in (("lang", "언어"), ("aircraft", "기종")):
        vals = out[by].fillna("").astype(str)
        if vals.nunique() > 1 or (vals != "").all():
            for v, g in out.groupby(vals):
                rows.append({"구분": name, "값": v or "(빈칸)", **score(g, k)})
    summary = pd.DataFrame(rows)

    aug = meta.get("augment_files")
    info = pd.DataFrame({"항목": ["평가 파일", "모델 폴더", "모델 epoch", "번역 증강 파일"],
                         "값": [str(input_path), str(model_dir), meta.get("epoch"),
                                "(기록 없음)" if aug is None else (", ".join(aug) or "없음(한국어만 학습)")]})
    # 기본 저장 위치: 학습 결과 폴더 옆 eval/ (모델 폴더가 읽기 전용이어도 되도록)
    default = Path(cfg["train"]["output_dir"]).parent / "eval" / f"{input_path.stem}__{model_dir.name}.xlsx"
    output_path = Path(output_path or default)
    detail_cols = ["id", "lang", "aircraft", "defect", "action", "hmc", "pred_hmc", "pred_hmc_conf", "hmc_맞음",
                   "atc", "pred_atc", "pred_atc_conf", "atc_맞음", f"hmc_top{k}", f"atc_top{k}", "rule_flags",
                   "review_needed"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".xlsx":
        with pd.ExcelWriter(output_path) as w:
            summary.to_excel(w, sheet_name="요약", index=False)
            info.to_excel(w, sheet_name="모델", index=False)
            out[detail_cols].to_excel(w, sheet_name="건별", index=False)
    else:
        write_table(summary, output_path)
    print(f"\n평가: {input_path.name} ({len(out)}건) / 모델: {model_dir} / 번역 증강: {info['값'][3]}", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"결과: {output_path}", flush=True)
    return summary
