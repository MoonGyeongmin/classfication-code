import argparse
import logging
import os
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path

import yaml

from .errors import MccError, describe

REQUIRED = {"columns": ["data"], "model": ["encoder", "max_length"], "train": ["train_file", "output_dir"],
            "predict": ["batch_size"]}


def load_config(path):
    try:
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        raise MccError("E01", f"{path}: {str(e).splitlines()[0]}") from e
    if not isinstance(cfg, dict):
        raise MccError("E01", f"{path}: 내용이 비었거나 형식이 다름")
    missing = [f"{a}.{b}" for a, keys in REQUIRED.items() for b in keys if b not in (cfg.get(a) or {})]
    if missing:
        raise MccError("E02", ", ".join(missing))
    return cfg


def classify(e):
    """예외를 짧은 오류 코드로 바꾼다."""
    if isinstance(e, MccError):
        return e.code, e.detail
    name = type(e).__name__
    if name == "OutOfMemoryError":
        return "E70", ""
    if isinstance(e, MemoryError):
        return "E71", ""
    if name in ("APIConnectionError", "APITimeoutError"):
        return "E51", str(e)
    if name in ("AuthenticationError", "PermissionDeniedError"):
        return "E53", ""
    if name == "NotFoundError":
        return "E52", str(e)[:150]
    if name in ("BadRequestError", "APIStatusError", "UnprocessableEntityError", "InternalServerError", "RateLimitError"):
        return "E54", str(e)[:150]
    if isinstance(e, FileNotFoundError):
        return "E90", str(e.filename or e)
    if isinstance(e, PermissionError):
        return "E60", str(e.filename or e)
    tb = traceback.extract_tb(e.__traceback__)
    where = f"{Path(tb[-1].filename).name}:{tb[-1].lineno}" if tb else ""
    return "E99", f"{name}: {str(e)[:120]} ({where})"


def setup_logging():
    """화면 로그는 그대로 두고, 같은 내용을 /work/logs(없으면 ./logs)에 파일로도 남긴다."""
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)
    if "PYTEST_CURRENT_TEST" in os.environ:
        return None
    for d in (os.environ.get("MCC_LOG_DIR"), "/work/logs", "logs"):
        if not d:
            continue
        try:
            Path(d).mkdir(parents=True, exist_ok=True)
            path = Path(d) / f"mcc_{datetime.now():%Y%m%d_%H%M%S}.log"
            h = logging.FileHandler(path, encoding="utf-8")
            h.setFormatter(logging.Formatter(fmt))
            logging.getLogger().addHandler(h)
            return path
        except OSError:
            continue
    return None


def main(argv=None):
    """명령을 실행한다. 실패하면 마지막에 '[오류 E32] ...' 한 줄과 로그 파일 위치를 찍고 종료 코드 2로 끝난다."""
    log_path = setup_logging()
    try:
        rc = _main(argv)
    except SystemExit:
        raise
    except BaseException as e:  # noqa: BLE001
        if isinstance(e, KeyboardInterrupt):
            raise
        code, detail = classify(e)
        logging.getLogger("mcc").error("오류 %s: %s", code, detail)
        if log_path:  # traceback은 화면이 아니라 로그 파일에만 남긴다
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(traceback.format_exc())
        if "PYTEST_CURRENT_TEST" in os.environ:
            raise
        what, fix = describe(code)
        print(f"\n[오류 {code}] {what}" + (f": {detail}" if detail else ""), file=sys.stderr)
        print(f"  조치: {fix}", file=sys.stderr)
        if log_path:
            print(f"  자세한 내용: {str(log_path).replace('/work/', os.environ.get('MCC_HOST_WORK', '/work').rstrip('/') + '/', 1)}", file=sys.stderr)
        sys.exit(2)
    if rc and "PYTEST_CURRENT_TEST" not in os.environ:
        sys.exit(rc)
    return rc


def _main(argv=None):
    p = argparse.ArgumentParser(prog="mcc", description="정비이력 HMC/ATC/WUC 분류")
    p.add_argument("--config", default="configs/default.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("train", help="HMC/ATC 모델 학습")

    t = sub.add_parser("translate", help="로컬 LLM으로 학습 데이터 번역 증강")
    t.add_argument("--input", required=True)
    t.add_argument("--output", required=True)
    t.add_argument("--langs", nargs="*")

    pr = sub.add_parser("predict", help="파일 하나 예측")
    pr.add_argument("--input", required=True)
    pr.add_argument("--output", required=True)
    pr.add_argument("--model-dir")

    b = sub.add_parser("batch", help="inbox 폴더의 새 파일을 모두 예측하고 요약 보고서를 만든다(주기 실행용)")
    b.add_argument("--inbox", required=True)
    b.add_argument("--outbox", required=True)
    b.add_argument("--model-dir")

    d = sub.add_parser("doctor", help="환경·모델·설정·데이터·LiteLLM 점검. 문제를 짧은 코드(E32 등)로 보여 준다")
    d.add_argument("--deep", action="store_true", help="인코더 가중치까지 실제로 불러 본다(수십 초)")
    d.add_argument("--offline", action="store_true", help="LiteLLM 연결 점검을 건너뛴다")

    r = sub.add_parser("rehearse", help="가짜 데이터로 번역(가짜 API)→학습→예측→보고서를 끝까지 돌려 설치를 검증")
    r.add_argument("--full", action="store_true", help="학습도 실제 bge-m3로 한다(RAM 16GB 이상, CPU면 수십 분)")
    r.add_argument("--out", default="/work/rehearsal")

    args = p.parse_args(argv)
    cfg = load_config(args.config)

    if args.cmd == "train":
        from . import train
        train.run(cfg)
    elif args.cmd == "translate":
        from . import translate
        translate.run(cfg, args.input, args.output, args.langs)
    elif args.cmd == "predict":
        from . import predict
        predict.run(cfg, args.input, args.output, args.model_dir)
    elif args.cmd == "batch":
        run_batch(cfg, Path(args.inbox), Path(args.outbox), args.model_dir)
    elif args.cmd == "doctor":
        from . import doctor
        return doctor.run(cfg, deep=args.deep, offline=args.offline)
    elif args.cmd == "rehearse":
        from . import rehearse
        rehearse.run(cfg, Path(args.out), full=args.full)


def run_batch(cfg, inbox, outbox, model_dir=None):
    from . import predict
    from .report import write_summary

    log = logging.getLogger("mcc.batch")
    files = sorted(f for f in inbox.glob("*") if f.suffix.lower() in (".csv", ".xlsx", ".xls"))
    if not files:
        log.info("새 파일 없음: %s", inbox)
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = outbox / stamp
    done_dir = inbox / "processed" / stamp
    done_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        result = predict.run(cfg, f, out_dir / f"{f.stem}_pred.xlsx", model_dir)
        write_summary(result, cfg["columns"]["data"]["aircraft"], out_dir / f"{f.stem}_summary.xlsx")
        shutil.move(str(f), done_dir / f.name)
    log.info("%d개 파일 처리 완료 -> %s", len(files), out_dir)


if __name__ == "__main__":
    main()
