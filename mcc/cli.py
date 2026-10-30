import argparse
import logging
import shutil
from datetime import datetime
from pathlib import Path

import yaml


def load_config(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def main(argv=None):
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

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
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
