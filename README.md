# 정비이력 코드 분류 (HMC / ATC / WUC)

항공기 결함·정비이력 텍스트와 결함품 PN으로 TO 00-20-2의 How Malfunction Code(HMC), Action Taken Code(ATC)와
기종별 Work Unit Code(WUC)를 추천한다. 인터넷이 없는 폐쇄망(H200, Docker)에서 학습과 예측을 모두 돌리도록 만들었다.

## 방식

| 코드 | 방법 | 근거 데이터 |
|---|---|---|
| HMC (318종) / ATC (34종) | 다국어 인코더(BGE-M3) 하나에 분류 헤드 2개를 붙여 학습 | 정제된 한국어 FA-50 이력 |
| WUC | 학습 없이 검색. ① GBL에서 결함품 PN 조회 → ② 없으면 결함·조치 텍스트와 -06 부품명의 의미 유사도 | GBL 엑셀, -06 WUC 목록 |

- **다국어 대응**: 다국어 인코더는 한국어로만 학습해도 영어·폴란드어·태국어·필리핀어 입력을 어느 정도 이해한다.
  여기에 더해 `mcc translate`가 로컬 LLM(기본 Qwen2.5-32B-Instruct)으로 한국어 학습 데이터를 다른 언어로 번역해 학습에 넣는다.
  검증셋 원문의 번역본은 학습에서 빼고 **언어별 검증 점수**를 내는 데 쓴다.
- **TO 규칙 검증**: `codebook/rules.yaml`의 ATC-HMC 조합 규칙(예: ATC H는 HMC 672/799/812/948만)에 어긋나면
  상위 후보 중 규칙에 맞는 조합으로 바꾸고 `rule_flags`에 사유를 남긴다.
- **검토 대상 표시**: 신뢰도가 `review_threshold`보다 낮거나 규칙 위반이 있으면 `review_needed=True`.
- HMC/ATC는 기종 공통 코드라 FA-50으로 학습한 모델을 다른 기종에도 쓴다. 기종 정보는 입력 텍스트에 함께 들어간다.

## 폴더

```
mcc/            파이프라인 코드 (train, translate, predict, batch)
codebook/       TO 00-20-2에서 뽑은 atc.csv, hmc.csv와 rules.yaml
configs/        default.yaml: 컬럼명, 경로, 하이퍼파라미터를 모두 여기서 설정
docker/         Dockerfile, 반입 파일 빌드(build_offline.sh), 실행(run.sh), cron 예시
scripts/        extract_codebook.py (TO 개정판이 나오면 코드표 재생성)
tests/          가짜 데이터와 아주 작은 모델로 전체 흐름을 검증하는 테스트
```

## 1. 반입 파일 만들기 (인터넷 PC)

```bash
bash docker/build_offline.sh        # dist/ 에 이미지(.tar.gz), models.tar, SHA256SUMS 생성
```
이미지는 약 15GB, 모델은 BGE-M3 약 2.3GB + Qwen2.5-32B 약 65GB다. 번역 LLM이 너무 크면
`LLM=Qwen/Qwen2.5-14B-Instruct bash docker/build_offline.sh` 처럼 바꾸거나 `SKIP_LLM=1`로 뺀다.
이 저장소 폴더도 함께 반입한다(설정 파일과 실행 스크립트를 쓰기 때문).

## 2. 폐쇄망에 설치

```bash
sha256sum -c SHA256SUMS
docker load < mcc_0.1.0.tar.gz
sudo mkdir -p /srv/mcc/{data,work} && sudo tar -C /srv/mcc -xf models.tar   # -> /srv/mcc/models/...
nvidia-smi                                                                    # 드라이버 550 이상이어야 CUDA 12.4 이미지가 돈다
bash docker/run.sh --help
```

## 3. 데이터 준비

**컬럼명은 `configs/default.yaml` 맨 위 `columns:` 한곳에서만 바꾼다.** 왼쪽 키는 그대로 두고 오른쪽 값을 실제 엑셀 컬럼명으로 고치면 된다.

| 파일 | 기본 컬럼 | 비고 |
|---|---|---|
| 정비이력 (학습/예측) | record_id, aircraft_type, defect_text, action_text, part_number, hmc, atc, wuc | 예측 파일에는 hmc/atc/wuc가 없어도 된다 |
| GBL 엑셀 | aircraft_type, part_number, wuc, nomenclature | 기종별 파일이면 `wuc.gbl_files`에 `aircraft`를 적고 기종 컬럼은 없어도 된다 |
| -06 WUC 목록 | aircraft_type, wuc, wuc_name | IETM에서 WUC와 부품명을 엑셀로 내보낸 것. PN은 필요 없다 |

- HMC는 엑셀에서 숫자로 저장돼 앞자리 0이 빠져도(`20` → `020`) 자동으로 복원한다.
- TO 코드표에 없는 HMC/ATC가 붙은 행은 학습에서 빠지고 건수가 로그에 남는다.
- 데이터 기종명이 참고자료에 없으면 `wuc.aircraft_alias`의 대체 기종(예: FA-50PL → FA-50)에서 찾는다.

기본 경로(컨테이너 안 기준): 학습 파일 `/data/train/fa50_ko.xlsx`, 참고자료 `/data/ref/`, 모델 `/models/`, 결과 `/work/`.

## 4. 실행

```bash
# (선택) 번역 증강: 학습 데이터를 en/pl/th/tl로 번역. H200 1장 기준 수 시간 이내 예상
bash docker/run.sh translate --input /data/train/fa50_ko.xlsx --output /data/train/fa50_aug.xlsx
#   -> configs의 train.augment_files에 /data/train/fa50_aug.xlsx 추가

# 학습: /work/models/hmc_atc 에 최고 모델과 history.csv(에폭별·언어별 정확도) 저장
bash docker/run.sh train

# 파일 하나 예측
bash docker/run.sh predict --input /data/inbox/sample.xlsx --output /work/out/sample_pred.xlsx

# 주기 실행: inbox의 새 파일을 모두 예측 + 요약 보고서, 처리한 파일은 inbox/processed로 이동
bash docker/run.sh batch --inbox /data/inbox --outbox /work/out
```
매일 자동 실행은 `docker/cron.example`을 crontab에 등록한다.

## 결과 컬럼

원본 컬럼 뒤에 다음이 붙는다.

| 컬럼 | 내용 |
|---|---|
| pred_hmc, pred_hmc_conf, pred_hmc_desc | HMC 추천값, 신뢰도(0~1), TO 정의 |
| pred_atc, pred_atc_conf, pred_atc_desc | ATC 추천값, 신뢰도, TO 정의 |
| hmc_top3, atc_top3 | 상위 3개 후보와 확률 (`코드:확률 \| ...`) |
| pred_wuc, pred_wuc_name, wuc_top3, wuc_source | WUC 추천값, -06 부품명, 후보, 근거(GBL 또는 -06 유사도) |
| rule_flags | TO 조합 규칙 관련 메모 |
| review_needed | 사람이 확인해야 하는 건 |

`*_summary.xlsx`에는 기종별 HMC/ATC/WUC 분포와 검토 필요 비율이 들어간다.

## 테스트

```bash
docker run --rm --entrypoint python mcc:0.1.0 -m pytest -q tests   # 폐쇄망에서도 외부 모델 없이 돈다
```

## 알려진 한계

- 다른 언어의 성능은 번역본 검증셋으로 추정한 값이다. 실제 현지 기록으로 일부라도 라벨을 확인해 점검하는 것이 좋다.
- WUC는 GBL에 PN이 없고 텍스트에 부품명이 분명하지 않으면 정확도가 낮다. 이런 건은 `wuc_source`와 점수로 걸러 검토한다.
- 학습 데이터에 거의 없는 HMC는 잘 맞히지 못한다. 빈도 역수 가중치로 일부 보정한다.
