"""번역 증강(translate-train). 사내 LLM API(LiteLLM 등 OpenAI 호환) 또는 vLLM으로 한국어 학습 데이터의 결함/조치 텍스트를 다른 언어로 옮긴다.

라벨(HMC/ATC), PN, 기종은 그대로 두고 텍스트만 번역한 행을 lang 컬럼과 함께 저장한다.
결과 파일을 설정의 train.augment_files에 넣으면 학습에 쓰이고, 검증셋 원문의 번역본은 언어별 검증셋이 된다.
"""
import logging
import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from .data import read_table, write_table

log = logging.getLogger(__name__)

LANG_NAMES = {"en": "English", "pl": "Polish", "th": "Thai", "tl": "Filipino (Tagalog)", "id": "Indonesian",
              "ko": "Korean"}

SYSTEM = ("You are a professional translator for military aircraft maintenance records. "
          "Translate the user's text into {lang} the way a {lang}-speaking aircraft mechanic would write it. "
          "Keep part numbers, reference designators, TO/manual numbers, numbers, units and English technical "
          "abbreviations (e.g. MFD, ECS, BIT, CB, LRU) unchanged. Output only the translation.")


def messages(text, lang):
    return [{"role": "system", "content": SYSTEM.format(lang=LANG_NAMES.get(lang, lang))},
            {"role": "user", "content": text}]


class ApiTranslator:
    """OpenAI 호환 API(LiteLLM 등)로 번역한다. 키는 설정 파일이 아니라 환경 변수에서 읽는다."""

    def __init__(self, tcfg):
        from openai import OpenAI

        api = tcfg["api"]
        key = os.environ.get(api.get("api_key_env", "LITELLM_API_KEY"), "")
        if not key:
            raise RuntimeError(f"환경 변수 {api.get('api_key_env', 'LITELLM_API_KEY')}에 API 키를 넣어 주세요.")
        self.client = OpenAI(base_url=api["base_url"], api_key=key, timeout=api.get("timeout", 120),
                             max_retries=api.get("max_retries", 3))
        self.model = api["model"]
        self.workers = api.get("concurrency", 8)
        self.temperature = tcfg.get("temperature", 0.3)
        self.max_tokens = tcfg.get("max_tokens", 512)

    def _one(self, text, lang):
        r = self.client.chat.completions.create(model=self.model, messages=messages(text, lang),
                                                temperature=self.temperature, max_tokens=self.max_tokens)
        return (r.choices[0].message.content or "").strip()

    def __call__(self, texts, lang):
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(lambda t: self._one(t, lang), texts))


class VllmTranslator:
    """서버 안에서 vLLM으로 모델을 직접 띄워 번역한다(API가 없을 때)."""

    def __init__(self, tcfg):
        from vllm import LLM, SamplingParams  # 폐쇄망 이미지에만 있으면 되므로 여기서 불러온다

        self.llm = LLM(model=tcfg["llm"], tensor_parallel_size=tcfg.get("tensor_parallel", 1),
                       max_model_len=tcfg.get("max_model_len", 4096), gpu_memory_utilization=tcfg.get("gpu_mem", 0.9))
        self.tok = self.llm.get_tokenizer()
        self.params = SamplingParams(temperature=tcfg.get("temperature", 0.3), max_tokens=tcfg.get("max_tokens", 512))

    def __call__(self, texts, lang):
        prompts = [self.tok.apply_chat_template(messages(t, lang), tokenize=False, add_generation_prompt=True)
                   for t in texts]
        return [o.outputs[0].text.strip() for o in self.llm.generate(prompts, self.params)]


def run(cfg, input_path, output_path, langs=None):
    tcfg = cfg["translate"]
    langs = langs or tcfg["langs"]
    cols = cfg["columns"]["data"]
    src = read_table(input_path)
    text_cols = [cols[k] for k in ("defect", "action") if cols.get(k) and cols[k] in src.columns]
    translate = ApiTranslator(tcfg) if tcfg.get("backend", "api") == "api" else VllmTranslator(tcfg)

    outs = []
    for lang in langs:
        part = src.copy()
        for col in text_cols:
            texts = part[col].tolist()
            todo = [i for i, t in enumerate(texts) if t.strip()]
            log.info("%s: %s %d건 번역", lang, col, len(todo))
            for i, t in zip(todo, translate([texts[i] for i in todo], lang)):
                texts[i] = t
            part[col] = texts
        part["lang"] = lang
        outs.append(part)
    result = pd.concat(outs, ignore_index=True)
    write_table(result, output_path)
    log.info("번역 증강 %d건 -> %s", len(result), output_path)
    return result
