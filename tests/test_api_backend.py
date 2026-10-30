import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import fake_openai  # noqa: E402
from make_fixtures import build  # noqa: E402
from mcc.cli import main  # noqa: E402


def _cfg(tmp_path, fx, url):
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    cfg["codebook_dir"] = str(ROOT / "codebook")
    cfg["model"].update(encoder=str(fx / "encoder"), max_length=64)
    cfg["translate"]["api"].update(base_url=url, model="qwen-test", concurrency=4)
    cfg["wuc"].update(enabled=True, embed_backend="api",
                      gbl_files=[{"path": str(fx / "gbl.xlsx"), "aircraft": "FA-50"}],
                      wuc_table_files=[str(fx / "wuc_table.csv")])
    cfg["wuc"]["embed_api"].update(base_url=url, model="bge-m3-test")
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return cfg, path


def test_translate_and_wuc_via_api(tmp_path, monkeypatch):
    srv, url = fake_openai.start()
    monkeypatch.setenv("LITELLM_API_KEY", "sk-test")
    fx = build(tmp_path / "fx")
    cfg, cfg_path = _cfg(tmp_path, fx, url)

    src = pd.read_excel(fx / "train.xlsx", dtype=str)  # 전체를 번역해야 검증셋 원문의 번역본이 생긴다
    src.loc[3, "action_text"] = ""  # 빈 칸은 번역 요청을 보내지 않는다
    src.to_excel(tmp_path / "ko.xlsx", index=False)
    main(["--config", str(cfg_path), "translate", "--input", str(tmp_path / "ko.xlsx"),
          "--output", str(tmp_path / "aug.xlsx"), "--langs", "en", "th"])
    aug = pd.read_excel(tmp_path / "aug.xlsx", dtype=str).fillna("")
    n = len(src)
    assert len(aug) == 2 * n and set(aug["lang"]) == {"en", "th"}
    assert aug.loc[aug["lang"] == "th", "defect_text"].str.startswith("[Thai] ").all()
    assert (aug.loc[aug["lang"] == "en", "hmc"].values == src["hmc"].values).all()  # 라벨은 그대로
    assert aug.loc[[3, n + 3], "action_text"].eq("").all()
    chats = [c for c in fake_openai.Handler.calls if c[0].endswith("/chat/completions")]
    assert len(chats) == 2 * (n + n - 1) and all(c[1] == "Bearer sk-test" for c in chats)

    # 증강 파일을 넣어 학습하면 언어별 검증 점수가 나온다.
    cfg["train"].update(train_file=str(fx / "train.xlsx"), augment_files=[str(tmp_path / "aug.xlsx")],
                        output_dir=str(tmp_path / "model"), epochs=2, batch_size=16, lr=1e-3, head_lr=1e-2,
                        val_ratio=0.2)
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    main(["--config", str(cfg_path), "train"])
    hist = pd.read_csv(tmp_path / "model/history.csv")
    assert {"ko/hmc_acc", "en/hmc_acc", "th/hmc_acc"} <= set(hist.columns)

    # WUC 임베딩을 API로 받는 경로
    new = pd.read_excel(fx / "train.xlsx", dtype=str).drop(columns=["hmc", "atc", "wuc"]).head(5)
    new.loc[0, "part_number"] = ""
    new.to_csv(tmp_path / "new.csv", index=False)
    main(["--config", str(cfg_path), "predict", "--input", str(tmp_path / "new.csv"),
          "--output", str(tmp_path / "pred.xlsx")])
    pred = pd.read_excel(tmp_path / "pred.xlsx", dtype=str)
    assert pred.loc[0, "wuc_source"] == "-06 유사도" and pred["pred_wuc"].str.fullmatch(r"[0-9A-Z]{5}").all()
    assert any(c[0].endswith("/embeddings") for c in fake_openai.Handler.calls)
    srv.shutdown()


def test_missing_key_is_clear(tmp_path, monkeypatch):
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    fx = build(tmp_path / "fx")
    _, cfg_path = _cfg(tmp_path, fx, "http://127.0.0.1:9/v1")
    try:
        main(["--config", str(cfg_path), "translate", "--input", str(fx / "train.xlsx"),
              "--output", str(tmp_path / "aug.xlsx")])
    except RuntimeError as e:
        assert "LITELLM_API_KEY" in str(e)
    else:
        raise AssertionError("키가 없으면 오류가 나야 한다")
