"""외부 모델 없이 테스트하기 위한 가짜 데이터와 아주 작은 인코더를 만든다."""
import random
from pathlib import Path

import pandas as pd
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

# (결함, 조치, HMC, ATC, PN, WUC)
PATTERNS = [
    ("MFD 화면 꺼짐 현상 발생", "MFD 교환 후 작동점검 양호", "242", "R", "4-91234-1", "51AB0"),
    ("유압 라인 누유 발견", "O-링 교환 후 누유점검 양호", "381", "G", "MS28775-123", "45BC0"),
    ("타이어 마모 한계 초과", "타이어 교환", "020", "R", "AT12345-01", "13CA0"),
    ("BIT 경고등 점등 재현 불가", "지상 작동점검 결과 이상 없음", "799", "H", "", "71AA0"),
    ("배선 커넥터 접촉 불량", "커넥터 세척 및 재장착", "006", "V", "D38999/26", "55AA0"),
    ("부품 동류전용", "타 항공기로 부품 동류전용 장탈", "875", "T", "4-91234-1", "51AB0"),
]


def make_data(path, n=240, seed=0):
    rnd = random.Random(seed)
    rows = []
    for i in range(n):
        d, a, h, t, pn, w = rnd.choice(PATTERNS)
        rows.append({"record_id": f"R{i:05d}", "aircraft_type": "FA-50", "defect_text": d + rnd.choice(["", " (비행 후)", " 확인"]),
                     "action_text": a, "part_number": pn, "hmc": int(h), "atc": t, "wuc": ""})
    df = pd.DataFrame(rows)
    df.to_excel(path, index=False)  # HMC를 숫자로 저장해서 앞자리 0 복원도 같이 검증한다
    return df


def make_refs(dir_):
    # GBL은 기종별 엑셀 파일로 받는 경우를 흉내 내어 기종 컬럼 없이 만든다.
    gbl = pd.DataFrame([{"part_number": p[4], "wuc": p[5], "nomenclature": p[0]}
                        for p in PATTERNS if p[4]]).drop_duplicates(["part_number", "wuc"])
    table = pd.DataFrame([{"aircraft_type": "FA-50", "wuc": p[5], "wuc_name": p[0]} for p in PATTERNS]).drop_duplicates("wuc")
    gbl.to_excel(dir_ / "gbl.xlsx", index=False)
    table.to_csv(dir_ / "wuc_table.csv", index=False)


def make_encoder(path, corpus):
    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.NFKC()
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.train_from_iterator(corpus, trainers.WordPieceTrainer(
        vocab_size=500, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]))
    from tokenizers.processors import TemplateProcessing
    tok.post_processor = TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[
        ("[CLS]", tok.token_to_id("[CLS]")), ("[SEP]", tok.token_to_id("[SEP]"))])
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]",
                                   cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")
    cfg = BertConfig(vocab_size=fast.vocab_size, hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                     intermediate_size=64, max_position_embeddings=128)
    BertModel(cfg).save_pretrained(path)
    fast.save_pretrained(path)


def build(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    df = make_data(root / "train.xlsx")
    make_refs(root)
    corpus = [f"[기종] FA-50 [결함] {r.defect_text} [조치] {r.action_text} [PN] {r.part_number}" for r in df.itertuples()]
    make_encoder(root / "encoder", corpus)
    return root
