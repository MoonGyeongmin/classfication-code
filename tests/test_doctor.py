import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import fake_openai  # noqa: E402
from make_fixtures import build  # noqa: E402
from mcc import doctor  # noqa: E402
from mcc.cli import main  # noqa: E402
from mcc.errors import CODES, MccError  # noqa: E402


def _cfg(tmp_path):
    fx = build(tmp_path / "fx")
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    cfg["codebook_dir"] = str(ROOT / "codebook")
    cfg["model"]["encoder"] = str(fx / "encoder")
    cfg["train"].update(train_file=str(fx / "train.xlsx"), output_dir=str(tmp_path / "model"))
    return cfg, fx


def test_doctor_ok_and_api(tmp_path, monkeypatch, capsys):
    cfg, _ = _cfg(tmp_path)
    srv, url = fake_openai.start()
    cfg["translate"]["api"].update(base_url=url, model="qwen-test")
    monkeypatch.setenv("LITELLM_API_KEY", "sk-test")
    assert doctor.run(cfg) == 0
    cfg["translate"]["api"]["model"] = "없는모델"
    assert doctor.run(cfg) == 1 and "[E52]" in capsys.readouterr().out
    monkeypatch.setenv("LITELLM_API_KEY", "sk-wrong")
    doctor.run(cfg)
    assert "[E53]" in capsys.readouterr().out
    srv.shutdown()


def test_doctor_codes(tmp_path, monkeypatch, capsys):
    cfg, fx = _cfg(tmp_path)
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    cfg["columns"]["data"]["defect"] = "결함내용"
    cfg["columns"]["data"]["action"] = "조치내용"
    cfg["model"]["encoder"] = str(tmp_path / "nope")
    cfg["translate"]["api"]["base_url"] = "http://127.0.0.1:9/v1"
    assert doctor.run(cfg) == 1
    out = capsys.readouterr().out
    assert "[E20]" in out and "[E32]" in out and "[W50]" in out
    assert "결과 코드:" in out.splitlines()[-1]


def test_train_raises_short_codes(tmp_path):
    cfg, fx = _cfg(tmp_path)
    cfg["columns"]["data"]["hmc"] = "HMC코드"
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    with pytest.raises(MccError) as e:
        main(["--config", str(p), "train"])
    assert e.value.code == "E32" and "HMC코드" in e.value.detail
    with pytest.raises(MccError) as e:
        main(["--config", str(p), "predict", "--input", str(fx / "train.xlsx"), "--output", str(tmp_path / "o.xlsx")])
    assert e.value.code == "E35"


def test_cli_prints_code_line(tmp_path):
    """pytest 밖에서 실행하면 traceback 대신 '[오류 E01]' 한 줄과 조치가 나오고 종료 코드 2."""
    env = {k: v for k, v in os.environ.items() if k != "PYTEST_CURRENT_TEST"}
    env["MCC_LOG_DIR"] = str(tmp_path / "logs")
    r = subprocess.run([sys.executable, "-m", "mcc", "--config", str(tmp_path / "없음.yaml"), "train"],
                       cwd=ROOT, env=env, capture_output=True, text=True)
    assert r.returncode == 2
    assert "[오류 E01]" in r.stderr and "조치:" in r.stderr and "Traceback" not in r.stderr
    assert list((tmp_path / "logs").glob("mcc_*.log"))


def test_codes_documented():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert all(f"| {c} |" in readme for c in CODES), [c for c in CODES if f"| {c} |" not in readme]
