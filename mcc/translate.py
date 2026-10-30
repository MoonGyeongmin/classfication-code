"""번역 증강(translate-train). 로컬 LLM으로 한국어 학습 데이터의 결함/조치 텍스트를 다른 언어로 옮긴다.

라벨(HMC/ATC), PN, 기종은 그대로 두고 텍스트만 번역한 행을 lang 컬럼과 함께 저장한다.
결과 파일을 설정의 train.augment_files에 넣으면 학습에 쓰이고, 검증셋 원문의 번역본은 언어별 검증셋이 된다.
"""
import logging

import pandas as pd

from .data import read_table, write_table

log = logging.getLogger(__name__)

LANG_NAMES = {"en": "English", "pl": "Polish", "th": "Thai", "tl": "Filipino (Tagalog)", "id": "Indonesian",
              "ko": "Korean"}

SYSTEM = ("You are a professional translator for military aircraft maintenance records. "
          "Translate the user's text into {lang} the way a {lang}-speaking aircraft mechanic would write it. "
          "Keep part numbers, reference designators, TO/manual numbers, numbers, units and English technical "
          "abbreviations (e.g. MFD, ECS, BIT, CB, LRU) unchanged. Output only the translation.")


def run(cfg, input_path, output_path, langs=None):
    from vllm import LLM, SamplingParams  # 폐쇄망 이미지에만 있으면 되므로 여기서 불러온다

    tcfg = cfg["translate"]
    langs = langs or tcfg["langs"]
    cols = cfg["columns"]["data"]
    src = read_table(input_path)
    text_cols = [cols[k] for k in ("defect", "action") if cols.get(k) and cols[k] in src.columns]

    llm = LLM(model=tcfg["llm"], tensor_parallel_size=tcfg.get("tensor_parallel", 1),
              max_model_len=tcfg.get("max_model_len", 4096), gpu_memory_utilization=tcfg.get("gpu_mem", 0.9))
    tok = llm.get_tokenizer()
    params = SamplingParams(temperature=tcfg.get("temperature", 0.3), max_tokens=tcfg.get("max_tokens", 512))

    outs = []
    for lang in langs:
        part = src.copy()
        for col in text_cols:
            texts = part[col].tolist()
            todo = [i for i, t in enumerate(texts) if t.strip()]
            prompts = [tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM.format(lang=LANG_NAMES.get(lang, lang))},
                 {"role": "user", "content": texts[i]}], tokenize=False, add_generation_prompt=True) for i in todo]
            log.info("%s: %s %d건 번역", lang, col, len(prompts))
            for i, o in zip(todo, llm.generate(prompts, params)):
                texts[i] = o.outputs[0].text.strip()
            part[col] = texts
        part["lang"] = lang
        outs.append(part)
    result = pd.concat(outs, ignore_index=True)
    write_table(result, output_path)
    log.info("번역 증강 %d건 -> %s", len(result), output_path)
    return result
