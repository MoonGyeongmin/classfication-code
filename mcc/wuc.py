"""WUC 후보 찾기. 학습 라벨이 없으므로 GBL(PN-WUC 대응표)과 -06 교범의 WUC 테이블을 검색해서 찾는다.

1) 결함품 PN이 GBL에 있으면 그 WUC를 쓴다. 한 PN이 여러 WUC에 걸리면 텍스트 유사도로 순위를 매긴다.
2) GBL에 없으면 결함/조치 텍스트와 -06 WUC 명칭의 임베딩 유사도로 후보를 찾는다.
"""
import logging
import os
import re

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from .data import WUC_RE, norm_wuc, read_table

log = logging.getLogger(__name__)


def norm_pn(v):
    return re.sub(r"\s+", "", str(v)).upper()


def norm_ac(v):
    return re.sub(r"\s+", "", str(v)).upper()


def load_refs(files, cols, keys):
    """GBL / -06 WUC 목록 파일들을 읽어 표준 컬럼(aircraft, part_number, wuc, name)으로 합친다."""
    frames = []
    for f in files or []:
        f = {"path": f} if isinstance(f, str) else f
        t = read_table(f["path"], f.get("sheet"))
        missing = [cols[k] for k in keys if cols.get(k) and cols[k] not in t.columns and not (k == "aircraft" and f.get("aircraft"))]
        if missing:
            raise KeyError(f"{f['path']}에 컬럼 {missing}이 없습니다. configs의 columns 설정을 확인하세요. 실제 컬럼: {list(t.columns)}")
        d = pd.DataFrame(index=t.index)
        for k in keys:
            d[k] = t[cols[k]].astype(str).str.strip() if cols.get(k) and cols[k] in t.columns else ""
        if f.get("aircraft"):
            d["aircraft"] = f["aircraft"]
        frames.append(d)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=keys)
    df["aircraft"] = df["aircraft"].map(norm_ac)
    df["wuc"] = df["wuc"].map(norm_wuc)
    if "part_number" in df:
        df["part_number"] = df["part_number"].map(norm_pn)
    bad = ~df["wuc"].str.match(WUC_RE)
    if bad.any():
        log.warning("형식이 5자리 영숫자가 아닌 WUC %d건 제외: %s", int(bad.sum()), df.loc[bad, "wuc"].head(5).tolist())
    return df[~bad]


class Embedder:
    def __init__(self, path, max_length=128, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path).to(self.device).eval()
        self.max_length = max_length

    @torch.no_grad()
    def __call__(self, texts, batch_size=256):
        out = []
        for s in range(0, len(texts), batch_size):
            enc = self.tok(texts[s:s + batch_size], padding=True, truncation=True,
                           max_length=self.max_length, return_tensors="pt").to(self.device)
            h = self.model(**enc).last_hidden_state[:, 0]  # bge-m3 dense 임베딩은 CLS 토큰을 쓴다
            out.append(torch.nn.functional.normalize(h.float(), dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 1))


class ApiEmbedder:
    """OpenAI 호환 /v1/embeddings로 임베딩을 받는다(LiteLLM에 올라간 bge-m3 등)."""

    def __init__(self, api):
        from openai import OpenAI

        key = os.environ.get(api.get("api_key_env", "LITELLM_API_KEY"), "")
        if not key:
            raise RuntimeError(f"환경 변수 {api.get('api_key_env', 'LITELLM_API_KEY')}에 API 키를 넣어 주세요.")
        self.client = OpenAI(base_url=api["base_url"], api_key=key, timeout=api.get("timeout", 120),
                             max_retries=api.get("max_retries", 3))
        self.model = api["model"]
        self.batch_size = api.get("batch_size", 64)

    def __call__(self, texts):
        out = []
        for s in range(0, len(texts), self.batch_size):
            r = self.client.embeddings.create(model=self.model, input=texts[s:s + self.batch_size])
            out.extend(d.embedding for d in sorted(r.data, key=lambda d: d.index))
        v = np.asarray(out, dtype=np.float32).reshape(len(texts), -1)
        return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-9, None)


class WucMatcher:
    def __init__(self, cfg):
        w = cfg["wuc"]
        self.top_k = w.get("top_k", 3)
        if w.get("embed_backend", "local") == "api":
            self.embed = ApiEmbedder(w["embed_api"])
        else:
            self.embed = Embedder(w.get("embedder") or cfg["model"]["encoder"])

        self.table = load_refs(w.get("wuc_table_files"), cfg["columns"]["wuc_table"], ["aircraft", "wuc", "name"])
        self.table = self.table.drop_duplicates(["aircraft", "wuc"]).reset_index(drop=True)
        self.table_vec = self.embed(list(self.table["name"])) if len(self.table) else None
        log.info("-06 WUC 목록 %d건 로드", len(self.table))

        self.gbl = load_refs(w.get("gbl_files"), cfg["columns"]["gbl"], ["aircraft", "part_number", "wuc", "name"])
        self.gbl = self.gbl.drop_duplicates(["aircraft", "part_number", "wuc"]).reset_index(drop=True)
        log.info("GBL %d건 로드", len(self.gbl))
        self.alias = {norm_ac(k): [norm_ac(x) for x in v] for k, v in (w.get("aircraft_alias") or {}).items()}

    def _aircraft_keys(self, ac):
        ac = norm_ac(ac)
        return [ac] + self.alias.get(ac, [])

    def _name(self, ac_keys, wuc):
        hit = self.table[(self.table["wuc"] == wuc) & self.table["aircraft"].isin(ac_keys)]
        return hit["name"].iloc[0] if len(hit) else ""

    def annotate(self, df):
        qvec = self.embed(list(df["text"])) if len(self.table) or len(self.gbl) else None
        rows = []
        for i, r in enumerate(df.to_dict("records")):
            keys = self._aircraft_keys(r.get("aircraft", ""))
            cands, source = [], ""
            pn = norm_pn(r.get("part_number", ""))
            if pn and len(self.gbl):
                g = self.gbl[self.gbl["part_number"] == pn]
                if len(g) and g["aircraft"].ne("").any():
                    g = g[g["aircraft"].isin(keys)] if g["aircraft"].isin(keys).any() else g
                if len(g):
                    names = [n or self._name(keys, w) or w for n, w in zip(g["name"], g["wuc"])]
                    sims = self.embed(names) @ qvec[i] if len(g) > 1 else np.ones(1)
                    cands = [(w, float(s)) for w, s in sorted(zip(g["wuc"], sims), key=lambda x: -x[1])]
                    source = "GBL"
            if not cands and len(self.table):
                mask = self.table["aircraft"].isin(keys).values
                if not mask.any():
                    mask[:] = True
                sims = self.table_vec[mask] @ qvec[i]
                idx = np.argsort(-sims)[:self.top_k]
                cands = [(self.table["wuc"].values[mask][j], float(sims[j])) for j in idx]
                source = "-06 유사도"
            cands = cands[:self.top_k]
            rows.append({
                "pred_wuc": cands[0][0] if cands else "",
                "pred_wuc_name": self._name(keys, cands[0][0]) if cands else "",
                f"wuc_top{self.top_k}": " | ".join(f"{w}:{s:.2f}" for w, s in cands),
                "wuc_source": source,
            })
        return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
