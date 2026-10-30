import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from make_fixtures import build  # noqa: E402
from mcc.cli import main  # noqa: E402
from mcc.codebook import Codebook  # noqa: E402


def test_codebook():
    cb = Codebook.load(ROOT / "codebook")
    assert len(cb.atc) == 34 and "I" not in cb.atc and "O" not in cb.atc
    assert len(cb.hmc) == 318 and "001" in cb.hmc
    assert cb.violations("H", "799") == []
    assert cb.violations("H", "242")
    assert cb.violations("R", "875")


def test_train_predict_batch(tmp_path):
    fx = build(tmp_path / "fx")
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    cfg["codebook_dir"] = str(ROOT / "codebook")
    cfg["model"]["encoder"] = str(fx / "encoder")
    cfg["model"]["max_length"] = 64
    cfg["train"].update(train_file=str(fx / "train.xlsx"), output_dir=str(tmp_path / "model"),
                        epochs=8, batch_size=16, lr=1e-3, head_lr=1e-2, val_ratio=0.2)
    cfg["wuc"].update(enabled=True, embedder=str(fx / "encoder"),
                      gbl_files=[{"path": str(fx / "gbl.xlsx"), "aircraft": "FA-50"}],
                      wuc_table_files=[str(fx / "wuc_table.csv")])
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    main(["--config", str(cfg_path), "train"])
    hist = pd.read_csv(tmp_path / "model/history.csv")
    assert hist["ko/hmc_acc"].max() > 0.9 and hist["ko/atc_acc"].max() > 0.9

    inbox, outbox = tmp_path / "inbox", tmp_path / "outbox"
    inbox.mkdir()
    new = pd.read_excel(fx / "train.xlsx", dtype=str).drop(columns=["hmc", "atc", "wuc"]).head(30)
    new.loc[0, "part_number"] = ""  # GBL로 못 찾는 경우 -06 유사도 경로를 탄다
    new.to_csv(inbox / "new.csv", index=False)
    main(["--config", str(cfg_path), "batch", "--inbox", str(inbox), "--outbox", str(outbox)])

    out_dir = next(outbox.iterdir())
    pred = pd.read_excel(out_dir / "new_pred.xlsx", dtype=str)
    assert len(pred) == 30
    assert pred["pred_hmc"].str.fullmatch(r"\d{3}").all()
    assert pred["pred_atc"].str.fullmatch(r"[0-9A-HJ-NP-Z]").all()
    assert set(pred["wuc_source"]) == {"GBL", "-06 유사도"}
    gbl_rows = pred[pred["wuc_source"] == "GBL"]
    gold = pd.read_excel(fx / "gbl.xlsx", dtype=str).groupby("part_number")["wuc"].apply(set)
    assert all(w in gold[p] for p, w in zip(gbl_rows["part_number"], gbl_rows["pred_wuc"]))
    assert (out_dir / "new_summary.xlsx").exists()
    assert not (inbox / "new.csv").exists()
