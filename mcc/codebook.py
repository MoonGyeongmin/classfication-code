import csv
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Codebook:
    hmc: dict            # code -> definition
    atc: dict
    rules: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path):
        path = Path(path)

        def read(name):
            with open(path / name, encoding="utf-8") as f:
                return {r["code"]: r["definition"] for r in csv.DictReader(f)}

        rules = {}
        if (path / "rules.yaml").exists():
            rules = yaml.safe_load((path / "rules.yaml").read_text(encoding="utf-8")) or {}
        return cls(hmc=read("hmc.csv"), atc=read("atc.csv"), rules=rules)

    def violations(self, atc, hmc):
        """ATC-HMC 조합이 TO 규칙에 어긋나면 사유 목록을 돌려준다."""
        out = []
        req = self.rules.get("atc_requires_hmc", {}).get(atc)
        if req and hmc not in req:
            out.append(f"ATC {atc}는 HMC {'/'.join(req)}와만 사용")
        if hmc in self.rules.get("atc_forbids_hmc", {}).get(atc, []):
            out.append(f"ATC {atc}는 HMC {hmc} 사용 불가")
        req = self.rules.get("hmc_requires_atc", {}).get(hmc)
        if req and atc not in req:
            out.append(f"HMC {hmc}는 ATC {'/'.join(req)}와만 사용")
        return out
