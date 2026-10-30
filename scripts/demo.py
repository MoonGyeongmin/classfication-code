"""가짜 데이터로 학습 -> 배치 예측 -> 요약까지 한 번 돌려 결과 엑셀을 OUT 폴더에 모은다.

사용법: python scripts/demo.py <결과폴더> <작업폴더>
"""
import sys, shutil
from pathlib import Path
import pandas as pd, yaml
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
from make_fixtures import build
from mcc.cli import main
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True); work = Path(sys.argv[2]); shutil.rmtree(work, ignore_errors=True)
fx = build(work / "fx")
cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
cfg["codebook_dir"] = str(ROOT / "codebook"); cfg["model"].update(encoder=str(fx / "encoder"), max_length=64)
cfg["train"].update(train_file=str(fx / "train.xlsx"), output_dir=str(work / "model"), epochs=30, batch_size=16, lr=1e-3, head_lr=1e-2, val_ratio=0.2)
cfg["wuc"].update(enabled=True, embedder=str(fx / "encoder"), gbl_files=[{"path": str(fx / "gbl.xlsx"), "aircraft": "FA-50"}], wuc_table_files=[str(fx / "wuc_table.csv")])
cp = work / "cfg.yaml"; cp.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
main(["--config", str(cp), "train"])
inbox = work / "inbox"; inbox.mkdir()
new = pd.read_excel(fx / "train.xlsx", dtype=str).drop(columns=["hmc", "atc", "wuc"]).head(30)
new.loc[0, "part_number"] = ""
new.to_excel(inbox / "new_records.xlsx", index=False)
shutil.copy(inbox / "new_records.xlsx", OUT / "3_예측입력_가짜.xlsx")
main(["--config", str(cp), "batch", "--inbox", str(inbox), "--outbox", str(work / "out")])
d = next((work / "out").iterdir())
shutil.copy(fx / "train.xlsx", OUT / "1_학습데이터_가짜.xlsx")
shutil.copy(fx / "gbl.xlsx", OUT / "2a_GBL_가짜.xlsx")
pd.read_csv(fx / "wuc_table.csv").to_excel(OUT / "2b_06_WUC목록_가짜.xlsx", index=False)
shutil.copy(d / "new_records_pred.xlsx", OUT / "4_예측결과.xlsx")
shutil.copy(d / "new_records_summary.xlsx", OUT / "5_요약보고서.xlsx")
pd.read_csv(work / "model/history.csv").to_excel(OUT / "6_학습기록.xlsx", index=False)
