"""TO 00-20-2 텍스트(PDF에서 추출한 .txt)에서 ATC(부록 E), HMC(부록 G) 코드표를 CSV로 추출한다.

사용법:
    python scripts/extract_codebook.py <TO 00-20-2 .txt> codebook/
"""
import csv
import re
import sys
from pathlib import Path

CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
PAGE_NOISE = re.compile(r"^(TO 00-20-2|[A-Z]-\d+(/\(.*\))?|Code (Type Defect|On Equip Off Equip) Definition)$")


def section(lines, start_pat, end_pat):
    start = next(i for i, l in enumerate(lines) if re.fullmatch(start_pat, l.strip()) and i > 1000)
    end = next(i for i in range(start + 1, len(lines)) if re.fullmatch(end_pat, lines[i].strip()))
    return [l.rstrip() for l in lines[start:end] if l.strip() and not PAGE_NOISE.match(l.strip())]


def parse(body, row_pat):
    rows, cur = [], None
    for line in body:
        m = row_pat.match(line)
        if m:
            cur = m.groupdict()
            cur["definition"] = cur["definition"].strip()
            rows.append(cur)
        elif cur is not None:
            cur["definition"] += " " + line.strip()
    return rows


def main(src, out_dir):
    lines = CONTROL.sub("", Path(src).read_text(encoding="utf-8")).splitlines()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    atc_body = section(lines, "APPENDIX E", "APPENDIX F")
    atc_body = atc_body[atc_body.index(next(l for l in atc_body if l.startswith("0- "))):]
    atc = parse(atc_body, re.compile(r"^(?P<cfar>\*?)(?P<code>[0-9A-Z])- (?P<on_equip>[01]) (?P<off_equip>[03]) (?P<definition>.*)$"))

    hmc_body = section(lines, "APPENDIX G", "APPENDIX H")
    hmc_body = hmc_body[hmc_body.index(next(l for l in hmc_body if l.startswith("001 "))):]
    hmc = parse(hmc_body, re.compile(r"^(?P<code>\d{3}) (?P<type_defect>[126]) (?P<definition>.*)$"))

    with open(out / "atc.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["code", "on_equip", "off_equip", "cfar", "definition"])
        w.writeheader()
        for r in atc:
            w.writerow({**r, "cfar": int(bool(r["cfar"]))})
    with open(out / "hmc.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["code", "type_defect", "definition"])
        w.writeheader()
        w.writerows(hmc)
    print(f"ATC {len(atc)}종, HMC {len(hmc)}종 -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
