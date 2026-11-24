"""mcc doctor: 실행 전에 환경·모델·설정·데이터와 (쓰는 경우에만) LiteLLM을 점검한다.

한 줄에 하나씩 [OK ] / [W11] / [E32] 처럼 찍고, 마지막 줄에 문제 코드만 모아 보여 준다.
폐쇄망에서는 그 마지막 줄만 옮겨 적으면 된다. 오류(E)가 하나라도 있으면 종료 코드 1.
"""
import importlib
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from .errors import MccError, describe


class Report:
    def __init__(self):
        self.codes = []

    def ok(self, msg):
        print(f"[OK ] {msg}", flush=True)

    def bad(self, code, detail=""):
        what, fix = describe(code)
        print(f"[{code}] {what}" + (f": {detail}" if detail else ""), flush=True)
        print(f"       조치: {fix}", flush=True)
        self.codes.append(code)

    def check(self, name, fn):
        """fn이 MccError를 내면 그 코드로, 다른 예외면 E99로 적는다. 성공하면 fn이 돌려준 문장을 OK로 적는다."""
        try:
            msg = fn()
            if msg:
                self.ok(msg)
        except MccError as e:
            self.bad(e.code, e.detail)
        except Exception as e:  # noqa: BLE001
            self.bad("E99", f"{name}: {type(e).__name__} {str(e)[:120]}")


def check_packages():
    bad = []
    for m in ("torch", "transformers", "tokenizers", "pandas", "openpyxl", "sklearn", "yaml", "openai"):
        try:
            importlib.import_module(m)
        except Exception as e:  # noqa: BLE001
            bad.append(f"{m}({type(e).__name__})")
    if bad:
        raise MccError("E10", ", ".join(bad))
    import torch
    import transformers
    return f"파이썬 패키지 (torch {torch.__version__}, transformers {transformers.__version__})"


def check_gpu(r):
    import torch
    if torch.cuda.is_available():
        names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        free, total = torch.cuda.mem_get_info(0)
        r.ok(f"GPU {len(names)}장: {names[0]} (여유 {free / 2**30:.0f}/{total / 2**30:.0f} GB)")
    else:
        r.bad("W11", "torch.cuda.is_available() = False")


def check_encoder(path, deep):
    p = Path(path)
    if not p.is_dir():
        raise MccError("E20", str(p))
    names = {f.name for f in p.iterdir()}
    missing = []
    if "config.json" not in names:
        missing.append("config.json")
    if not names & {"model.safetensors", "pytorch_model.bin"}:
        missing.append("model.safetensors 또는 pytorch_model.bin")
    if not names & {"tokenizer.json", "sentencepiece.bpe.model", "vocab.txt"}:
        missing.append("tokenizer.json")
    if missing:
        raise MccError("E21", f"{p}: {', '.join(missing)}")
    from transformers import AutoConfig, AutoTokenizer
    try:
        conf = AutoConfig.from_pretrained(p)
        tok = AutoTokenizer.from_pretrained(p)
        tok("점검 test")
        if deep:
            from transformers import AutoModel
            AutoModel.from_pretrained(p)
    except Exception as e:  # noqa: BLE001
        raise MccError("E22", f"{p}: {type(e).__name__} {str(e)[:100]}") from e
    return f"인코더 {p} ({conf.model_type}, hidden {conf.hidden_size}{', 가중치 로드 확인' if deep else ''})"


def check_codebook(path):
    from .codebook import Codebook
    try:
        cb = Codebook.load(path)
    except Exception as e:  # noqa: BLE001
        raise MccError("E03", f"{path}: {type(e).__name__} {str(e)[:80]}") from e
    return f"코드표 HMC {len(cb.hmc)}종, ATC {len(cb.atc)}종, 규칙 {len(cb.rules)}묶음"


def check_train_file(cfg, r):
    from .codebook import Codebook
    from .data import read_table, require_columns, standardize

    path = Path(cfg["train"]["train_file"])
    if not path.is_file():
        raise MccError("E30", str(path))
    raw = read_table(path)
    cols = cfg["columns"]["data"]
    require_columns(raw, cols, ["hmc", "atc"], path.name, any_of=("defect", "action"))
    df = standardize(raw, cols)
    cb = Codebook.load(cfg["codebook_dir"])
    ok = df["hmc"].isin(cb.hmc.keys()) & df["atc"].isin(cb.atc.keys())
    if ok.sum() < 10:
        raise MccError("E34", f"{path.name}: 전체 {len(df)}건 중 코드표에 맞는 행 {int(ok.sum())}건")
    if (~ok).mean() > 0.05:
        r.bad("W33", f"{path.name}: {int((~ok).sum())}/{len(df)}건 제외 예정. 예: "
                     f"{df.loc[~ok, ['hmc', 'atc']].head(3).values.tolist()}")
    for f in cfg["train"].get("augment_files") or []:
        if not Path(f).is_file():
            raise MccError("E90", f"train.augment_files: {f}")
    return f"학습 파일 {path.name}: {len(df)}건, 학습 가능 {int(ok.sum())}건, HMC {df.loc[ok, 'hmc'].nunique()}종"


def check_model(cfg, r):
    d = Path(cfg["train"]["output_dir"])
    meta = d / "meta.json"
    if not meta.is_file():
        print(f"[ - ] 학습된 모델 없음({d}). 처음이면 정상이고, predict 전에 mcc train", flush=True)
        return None
    m = json.loads(meta.read_text(encoding="utf-8"))
    ko = (m.get("scores") or {}).get("ko", {})
    return f"학습된 모델 {d} (epoch {m.get('epoch')}, 검증 HMC {ko.get('hmc_acc', 0):.3f} / ATC {ko.get('atc_acc', 0):.3f})"


def check_workdir(cfg):
    d = Path(cfg["train"]["output_dir"]).parent
    try:
        d.mkdir(parents=True, exist_ok=True)
        t = d / ".mcc_write_test"
        t.write_text("x")
        t.unlink()
    except OSError as e:
        raise MccError("E60", f"{d}: {e.strerror}") from e
    return f"작업 폴더 쓰기 가능 {d}"


def check_wuc(cfg):
    from .wuc import load_refs
    w = cfg["wuc"]
    if not (w.get("gbl_files") or w.get("wuc_table_files")):
        raise MccError("E40", "wuc.enabled: true인데 gbl_files, wuc_table_files가 비어 있음")
    gbl = load_refs(w.get("gbl_files"), cfg["columns"]["gbl"], ["aircraft", "part_number", "wuc", "name"])
    table = load_refs(w.get("wuc_table_files"), cfg["columns"]["wuc_table"], ["aircraft", "wuc", "name"])
    return f"WUC 참고자료: GBL {len(gbl)}건, -06 목록 {len(table)}건"


def check_api(label, api, want_model, r):
    """OpenAI 호환 /v1/models로 주소, 키, 모델 이름을 확인한다."""
    env = api.get("api_key_env", "LITELLM_API_KEY")
    key = os.environ.get(env, "")
    if not key:
        r.bad("W50", f"{label}: 환경 변수 {env} 없음")
        return
    url = api["base_url"].rstrip("/") + "/models"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=api.get("check_timeout", 5)) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:150]
        r.bad("E53" if e.code in (401, 403) else "E54", f"{label}: {url} -> HTTP {e.code} {body}")
        return
    except Exception as e:  # noqa: BLE001
        r.bad("E51", f"{label}: {url} ({type(e).__name__}: {str(getattr(e, 'reason', e))[:80]})")
        return
    ids = [m.get("id") for m in data.get("data", [])]
    if want_model not in ids:
        r.bad("E52", f"{label}: '{want_model}' 없음. 있는 모델: {', '.join(map(str, ids[:10]))}")
        return
    r.ok(f"{label}: {api['base_url']} 연결, 모델 {want_model} 있음")


def run(cfg, deep=False, offline=False, report=None):
    r = report or Report()
    print("== mcc doctor", flush=True)
    r.check("packages", check_packages)
    if "E10" in r.codes:
        return finish(r)
    r.check("gpu", lambda: check_gpu(r))
    r.check("codebook", lambda: check_codebook(cfg["codebook_dir"]))
    r.check("encoder", lambda: check_encoder(cfg["model"]["encoder"], deep))
    r.check("workdir", lambda: check_workdir(cfg))
    r.check("train_file", lambda: check_train_file(cfg, r))
    r.check("model", lambda: check_model(cfg, r))

    w = cfg.get("wuc") or {}
    if w.get("enabled"):
        r.check("wuc", lambda: check_wuc(cfg))
        if w.get("embed_backend", "local") == "local":
            emb = w.get("embedder") or cfg["model"]["encoder"]
            if emb != cfg["model"]["encoder"]:
                r.check("wuc_embedder", lambda: check_encoder(emb, deep))
    else:
        print("[ - ] WUC 꺼짐(wuc.enabled: false)", flush=True)

    # 번역 증강은 선택이다. 꺼져 있으면(기본) LiteLLM이 없어도 점검이 통과한다.
    t = cfg.get("translate") or {}
    use_translate = t.get("enabled", False) and t.get("backend", "api") == "api" and t.get("api")
    use_embed_api = w.get("enabled") and w.get("embed_backend") == "api"
    if not t.get("enabled", False):
        print("[ - ] 번역 증강 꺼짐(translate.enabled: false). LiteLLM 없이 한국어 데이터만으로 학습", flush=True)
    if offline and (use_translate or use_embed_api):
        print("[ - ] LiteLLM 점검 건너뜀(--offline)", flush=True)
    elif not offline:
        if use_translate:
            r.check("translate_api", lambda: check_api("번역 API", t["api"], t["api"].get("model"), r))
        if use_embed_api:
            r.check("embed_api", lambda: check_api("임베딩 API", w["embed_api"], w["embed_api"].get("model"), r))
    return finish(r)


def finish(r):
    errors = [c for c in r.codes if c.startswith("E")]
    print("", flush=True)
    if r.codes:
        print(f"결과 코드: {' '.join(dict.fromkeys(r.codes))}   (뜻은 README '오류 코드표')", flush=True)
    else:
        print("결과 코드: 없음 (모두 정상)", flush=True)
    return 1 if errors else 0
