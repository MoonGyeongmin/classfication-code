import logging

import numpy as np
import pandas as pd
import torch

from .codebook import Codebook
from .data import add_text, read_table, standardize, write_table
from .model import MultiHeadClassifier

log = logging.getLogger(__name__)


def pick_pair(p_hmc, p_atc, hmc_codes, atc_codes, codebook, k_hmc=10, k_atc=5):
    """HMC/ATC 상위 후보 조합 중 TO 규칙을 지키면서 확률곱이 가장 큰 쌍을 고른다. 없으면 None."""
    hi = np.argsort(-p_hmc)[:k_hmc]
    ai = np.argsort(-p_atc)[:k_atc]
    best = None
    for h in hi:
        for a in ai:
            s = p_hmc[h] * p_atc[a]
            if (best is None or s > best[0]) and not codebook.violations(atc_codes[a], hmc_codes[h]):
                best = (s, h, a)
    return best


@torch.no_grad()
def predict_df(df, model, tokenizer, codebook, cfg, device):
    pcfg = cfg["predict"]
    max_length = cfg["model"]["max_length"]
    maps = model.label_maps
    probs = {k: [] for k in maps}
    texts = list(df["text"])
    for s in range(0, len(texts), pcfg["batch_size"]):
        enc = tokenizer(texts[s:s + pcfg["batch_size"]], padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        for k in maps:
            probs[k].append(logits[k].float().softmax(-1).cpu().numpy())
    probs = {k: np.concatenate(v) if v else np.zeros((0, len(maps[k]))) for k, v in probs.items()}

    k = pcfg["top_k"]
    rows = []
    for i in range(len(df)):
        ph, pa = probs["hmc"][i], probs["atc"][i]
        h, a = int(ph.argmax()), int(pa.argmax())
        row = {}
        adjusted = False
        if pcfg.get("apply_rules", True) and codebook.violations(maps["atc"][a], maps["hmc"][h]):
            pair = pick_pair(ph, pa, maps["hmc"], maps["atc"], codebook)
            if pair:
                _, h, a = pair
                adjusted = True
        hmc, atc = maps["hmc"][h], maps["atc"][a]
        row["pred_hmc"], row["pred_hmc_conf"] = hmc, round(float(ph[h]), 4)
        row["pred_atc"], row["pred_atc_conf"] = atc, round(float(pa[a]), 4)
        row["pred_hmc_desc"] = codebook.hmc.get(hmc, "")
        row["pred_atc_desc"] = codebook.atc.get(atc, "")
        for name, p in (("hmc", ph), ("atc", pa)):
            top = np.argsort(-p)[:k]
            row[f"{name}_top{k}"] = " | ".join(f"{maps[name][j]}:{p[j]:.2f}" for j in top)
        flags = codebook.violations(atc, hmc)
        if adjusted:
            flags.insert(0, "규칙에 맞게 후보 조정")
        row["rule_flags"] = "; ".join(flags)
        row["review_needed"] = bool(
            min(row["pred_hmc_conf"], row["pred_atc_conf"]) < pcfg["review_threshold"] or flags)
        rows.append(row)
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def run(cfg, input_path, output_path, model_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codebook = Codebook.load(cfg["codebook_dir"])
    model, tokenizer, _ = MultiHeadClassifier.load(model_dir or cfg["train"]["output_dir"], device)

    raw = read_table(input_path)
    df = add_text(standardize(raw, cfg["columns"]["data"]))
    out = predict_df(df, model, tokenizer, codebook, cfg, device)

    if cfg.get("wuc", {}).get("enabled"):
        from .wuc import WucMatcher
        out = WucMatcher(cfg).annotate(out)

    # 원본 컬럼을 그대로 두고 예측 컬럼을 뒤에 붙인다.
    pred_cols = [c for c in out.columns if c.startswith(("pred_", "hmc_top", "atc_top", "wuc_top"))
                 or c in ("rule_flags", "review_needed", "wuc_source")]
    result = pd.concat([raw.reset_index(drop=True), out[pred_cols]], axis=1)
    write_table(result, output_path)
    log.info("예측 %d건 -> %s (검토 필요 %d건)", len(result), output_path, int(result["review_needed"].sum()))
    return result
