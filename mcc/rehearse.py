"""mcc rehearse: 가짜 데이터로 전체 흐름을 한 번에 돌려 설치가 제대로 됐는지 확인한다.

점검(doctor) -> 학습 -> 평가(evaluate) -> 예측 -> 배치 예측+요약 보고서. LiteLLM(Qwen)은 쓰지 않는다.
--augment이면 학습 전에 번역 증강 단계(가짜 OpenAI 호환 서버)를 넣어 증강 경로도 검증한다.
기본은 학습에 아주 작은 가짜 인코더를 써서 CPU로도 몇 분 안에 끝나고, 설정의 실제 인코더(bge-m3)는
점검과 WUC 유사도 검색(추론만)에 쓴다. --full이면 학습도 실제 인코더로 한다(RAM 16GB 이상).
네트워크가 끊긴 컨테이너(--network none)에서도 돈다. 결과는 <out>/결과 에 모인다.
"""
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

from .errors import MccError

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]


def step(n, total, title):
    print(f"\n========== [{n}/{total}] {title} ==========", flush=True)


def run(cfg, out, full=False, augment=False):
    sys.path[:0] = [str(ROOT / "tests")]
    import fake_openai
    from make_fixtures import build

    from . import doctor, evaluate, predict, train, translate
    from .cli import run_batch

    t0 = time.time()
    out = Path(out)
    shutil.rmtree(out, ignore_errors=True)
    print(f"리허설 폴더: {out}", flush=True)
    fx = build(out / "fx")  # 가짜 FA-50 정비이력 240건, GBL, -06 WUC 목록, 작은 인코더
    real = Path(cfg["model"]["encoder"])
    has_real = (real / "config.json").is_file()
    if not has_real:
        print(f"[주의] 실제 인코더 {real} 가 없어 WUC 검색도 가짜 인코더로 합니다.", flush=True)
    if full and not has_real:
        raise MccError("E20", str(real))

    srv = None
    r = yaml.safe_load(yaml.safe_dump(cfg))
    r["codebook_dir"] = cfg["codebook_dir"]
    r["model"].update(encoder=str(real if full else fx / "encoder"), max_length=64)
    r["train"].update(train_file=str(fx / "train.xlsx"), augment_files=[str(out / "aug.xlsx")] if augment else [],
                      output_dir=str(out / "model"), epochs=1 if full else 15, batch_size=8 if full else 16,
                      lr=2e-5 if full else 1e-3, head_lr=1e-2, val_ratio=0.2)
    r["translate"]["enabled"] = augment
    if augment:
        srv, url = fake_openai.start()
        api = dict(cfg["translate"]["api"], base_url=url, model="qwen-test")
        os.environ[api.get("api_key_env", "LITELLM_API_KEY")] = "sk-test"
        r["translate"].update(backend="api", api=api, langs=["en", "th"])
    r["wuc"].update(enabled=True, embed_backend="local", embedder=str(real if has_real else fx / "encoder"),
                    gbl_files=[{"path": str(fx / "gbl.xlsx"), "aircraft": "FA-50"}],
                    wuc_table_files=[str(fx / "wuc_table.csv")])
    (out / "cfg.yaml").write_text(yaml.safe_dump(r, allow_unicode=True), encoding="utf-8")
    total = 6 if augment else 5
    n = iter(range(1, total + 1))

    step(next(n), total, "점검 (mcc doctor --deep)")
    pre = dict(r, train=dict(r["train"], augment_files=[]))  # 번역 결과는 다음 단계에서 생기므로 점검에서는 뺀다
    rep = doctor.Report()
    if doctor.run(pre, deep=True, report=rep):
        first = next(c for c in rep.codes if c.startswith("E"))
        raise MccError(first, "점검에서 막혀 리허설을 멈춤. 위의 [E..] 줄 참고")

    src = pd.read_excel(fx / "train.xlsx", dtype=str)
    if augment:
        step(next(n), total, "번역 증강 (가짜 OpenAI 호환 API, en/th)")
        src.to_excel(out / "ko.xlsx", index=False)
        translate.run(r, out / "ko.xlsx", out / "aug.xlsx")
        aug = pd.read_excel(out / "aug.xlsx", dtype=str)
        print(aug[["lang", "defect_text", "hmc", "atc"]].head(3).to_string(index=False), flush=True)

    step(next(n), total, f"학습 ({'실제 인코더' if full else '작은 가짜 인코더'}, 번역 증강 {'포함' if augment else '없음'}, CPU/GPU 자동)")
    hist = pd.DataFrame(train.run(r))
    print(hist.filter(regex="epoch|/hmc_acc|/atc_acc").tail(3).to_string(index=False), flush=True)

    step(next(n), total, "평가 (mcc evaluate, 정답 붙은 30건으로 채점)")
    labeled = src.head(30).assign(lang="ko")
    if augment:  # 같은 기록의 번역본도 넣어 언어별 점수가 나오는지 본다
        labeled = pd.concat([labeled, aug[aug["record_id"].isin(labeled["record_id"])]], ignore_index=True)
    labeled.to_excel(out / "eval_labeled.xlsx", index=False)
    evaluate.run(r, out / "eval_labeled.xlsx", out / "eval.xlsx")

    step(next(n), total, "예측 (mcc predict, WUC 포함)")
    new = pd.read_excel(fx / "train.xlsx", dtype=str).drop(columns=["hmc", "atc", "wuc"]).head(30)
    new.loc[0, "part_number"] = ""  # PN이 없는 행은 -06 유사도 검색으로 WUC를 찾는다
    new.to_excel(out / "new_records.xlsx", index=False)
    res = predict.run(r, out / "new_records.xlsx", out / "pred.xlsx")
    print(res[["record_id", "pred_hmc", "pred_atc", "pred_wuc", "wuc_source", "review_needed"]].head(5)
          .to_string(index=False), flush=True)

    step(next(n), total, "배치 예측 + 요약 보고서 (mcc batch)")
    inbox = out / "inbox"
    inbox.mkdir()
    shutil.copy(out / "new_records.xlsx", inbox / "new_records.xlsx")
    run_batch(r, inbox, out / "out")
    d = next((out / "out").iterdir())

    res_dir = out / "결과"
    res_dir.mkdir()
    files = [(fx / "train.xlsx", "1_학습데이터_가짜.xlsx"), (out / "aug.xlsx", "2_번역증강.xlsx"),
             (out / "new_records.xlsx", "3_예측입력_가짜.xlsx"), (d / "new_records_pred.xlsx", "4_예측결과.xlsx"),
             (d / "new_records_summary.xlsx", "5_요약보고서.xlsx"), (out / "eval.xlsx", "7_평가결과.xlsx")]
    for src_, name in files:
        if src_.exists():
            shutil.copy(src_, res_dir / name)
    hist.to_excel(res_dir / "6_학습기록.xlsx", index=False)
    if srv:
        srv.shutdown()
    print(f"\n리허설 성공 ({(time.time() - t0) / 60:.1f}분). 결과: {res_dir}", flush=True)
    for f in sorted(res_dir.iterdir()):
        print(f"  {f.name}", flush=True)
