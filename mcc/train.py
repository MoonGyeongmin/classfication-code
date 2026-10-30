import logging
import math
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.nn import functional as F
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .codebook import Codebook
from .data import add_text, filter_labels, read_table, standardize
from .model import MultiHeadClassifier

log = logging.getLogger(__name__)
TASKS = ("hmc", "atc")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split(df, val_ratio, seed):
    """HMC 기준 층화 분할. 1건뿐인 코드는 학습셋에만 넣는다."""
    counts = df["hmc"].value_counts()
    rare = df["hmc"].map(counts) < 2
    tr, va = train_test_split(df[~rare], test_size=val_ratio, random_state=seed, stratify=df.loc[~rare, "hmc"])
    return pd.concat([tr, df[rare]]).reset_index(drop=True), va.reset_index(drop=True)


def batches(df, tokenizer, label_maps, batch_size, max_length, shuffle=False):
    idx = np.arange(len(df))
    if shuffle:
        np.random.shuffle(idx)
    index = {k: {c: i for i, c in enumerate(v)} for k, v in label_maps.items()}
    for s in range(0, len(idx), batch_size):
        part = df.iloc[idx[s:s + batch_size]]
        enc = tokenizer(list(part["text"]), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        labels = {k: torch.tensor([index[k][c] for c in part[k]]) for k in TASKS if k in part}
        yield enc, labels


@torch.no_grad()
def evaluate(model, df, tokenizer, cfg, device):
    model.eval()
    probs = {k: [] for k in TASKS}
    for enc, _ in batches(df, tokenizer, model.label_maps, cfg["predict"]["batch_size"], cfg["model"]["max_length"]):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        for k in TASKS:
            probs[k].append(logits[k].float().softmax(-1).cpu())
    out = {}
    for k in TASKS:
        p = torch.cat(probs[k])
        gold = np.array([model.label_maps[k].index(c) for c in df[k]])
        pred = p.argmax(-1).numpy()
        top3 = p.topk(min(3, p.shape[1]), -1).indices.numpy()
        out[f"{k}_acc"] = accuracy_score(gold, pred)
        out[f"{k}_top3"] = float((top3 == gold[:, None]).any(1).mean())
        out[f"{k}_macro_f1"] = f1_score(gold, pred, average="macro", labels=np.unique(gold))
    return out


def run(cfg):
    tcfg = cfg["train"]
    set_seed(tcfg["seed"])
    codebook = Codebook.load(cfg["codebook_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = filter_labels(add_text(standardize(read_table(tcfg["train_file"]), cfg["columns"]["data"])), codebook)
    df["lang"] = "ko"
    train_df, val_df = split(df, tcfg["val_ratio"], tcfg["seed"])
    log.info("학습 %d건, 검증 %d건", len(train_df), len(val_df))

    # 번역 증강 데이터: 검증셋 원문의 번역본은 학습에서 빼고 언어별 검증셋으로 쓴다(누수 방지).
    val_sets = {"ko": val_df}
    val_ids = set(val_df["id"])
    for path in tcfg.get("augment_files") or []:
        aug = filter_labels(add_text(standardize(read_table(path), cfg["columns"]["data"])), codebook)
        for lang, g in aug.groupby("lang"):
            val_sets[lang] = pd.concat([val_sets.get(lang, g.iloc[:0]), g[g["id"].isin(val_ids)]])
        train_df = pd.concat([train_df, aug[~aug["id"].isin(val_ids)]]).reset_index(drop=True)
    log.info("증강 포함 학습 %d건, 언어별 검증 %s", len(train_df), {k: len(v) for k, v in val_sets.items()})

    label_maps = {"hmc": sorted(codebook.hmc), "atc": sorted(codebook.atc)}
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["encoder"])
    model = MultiHeadClassifier(cfg["model"]["encoder"], label_maps).to(device)

    # 데이터에 거의 없는 코드가 많으므로 빈도의 역수 제곱근으로 손실 가중치를 준다.
    weights = {}
    for k in TASKS:
        cnt = train_df[k].value_counts().reindex(label_maps[k]).fillna(0).values
        w = 1.0 / np.sqrt(np.maximum(cnt, 1))
        weights[k] = torch.tensor(w / w[cnt > 0].mean(), dtype=torch.float, device=device)

    params = [
        {"params": model.encoder.parameters(), "lr": tcfg["lr"]},
        {"params": model.heads.parameters(), "lr": tcfg["head_lr"]},
    ]
    opt = torch.optim.AdamW(params, weight_decay=0.01)
    steps = math.ceil(len(train_df) / tcfg["batch_size"]) * tcfg["epochs"]
    sched = get_linear_schedule_with_warmup(opt, int(steps * tcfg["warmup_ratio"]), steps)

    best, history = -1.0, []
    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        total = 0.0
        for step, (enc, labels) in enumerate(batches(train_df, tokenizer, label_maps, tcfg["batch_size"],
                                                     cfg["model"]["max_length"], shuffle=True), 1):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
            loss = sum(F.cross_entropy(logits[k].float(), labels[k].to(device), weight=weights[k],
                                       label_smoothing=tcfg["label_smoothing"]) for k in TASKS)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            total += loss.item()
            if step % 100 == 0:
                log.info("epoch %d step %d loss %.4f", epoch, step, total / step)

        scores = {lang: evaluate(model, v, tokenizer, cfg, device) for lang, v in val_sets.items() if len(v)}
        history.append({"epoch": epoch, **{f"{lang}/{m}": s for lang, d in scores.items() for m, s in d.items()}})
        log.info("epoch %d 검증: %s", epoch, history[-1])
        # 모든 언어 검증셋의 HMC+ATC 정확도 평균으로 최고 모델을 고른다.
        score = np.mean([d["hmc_acc"] + d["atc_acc"] for d in scores.values()])
        if score > best:
            best = score
            model.save(tcfg["output_dir"], tokenizer, {"epoch": epoch, "scores": scores,
                                                         "max_length": cfg["model"]["max_length"]})
            log.info("최고 모델 저장: %s", tcfg["output_dir"])
    pd.DataFrame(history).to_csv(f"{tcfg['output_dir']}/history.csv", index=False)
    return history
