# 정비이력 코드 분류 (HMC / ATC / WUC)

항공기 결함·정비이력 텍스트와 결함품 PN으로 TO 00-20-2의 How Malfunction Code(HMC), Action Taken Code(ATC)와
기종별 Work Unit Code(WUC)를 추천한다. 인터넷이 없는 폐쇄망(H200, Docker)에서 학습과 예측을 모두 돌리도록 만들었다.

## 방식

| 코드 | 방법 | 근거 데이터 |
|---|---|---|
| HMC (318종) / ATC (34종) | 다국어 인코더(BGE-M3) 하나에 분류 헤드 2개를 붙여 학습 | 정제된 한국어 FA-50 이력 |
| WUC | 학습 없이 검색. ① GBL에서 결함품 PN 조회 → ② 없으면 결함·조치 텍스트와 -06 부품명의 의미 유사도 | GBL 엑셀, -06 WUC 목록 |

- **다국어 대응**: 다국어 인코더(BGE-M3)는 한국어로만 학습해도 영어·폴란드어·태국어·필리핀어 입력을 어느 정도 이해한다.
  **기본은 번역 증강 없이 한국어 데이터만으로 학습하며 LiteLLM/Qwen이 필요 없다.**
  외국어 정확도가 부족하면 그때 `translate.enabled: true`로 켜고 `mcc translate`(사내 LiteLLM의 Qwen 등)로
  한국어 학습 데이터를 다른 언어로 번역해 학습에 넣는다. 이때 검증셋 원문의 번역본은 학습에서 빼고 언어별 검증 점수를 낸다.
- **외국어 평가**: 현지 기록 일부에 사람이 정답을 붙여 `mcc evaluate`로 채점하면, 증강 전후 모델을 같은 기준으로 비교할 수 있다.
- **TO 규칙 검증**: `codebook/rules.yaml`의 ATC-HMC 조합 규칙(예: ATC H는 HMC 672/799/812/948만)에 어긋나면
  상위 후보 중 규칙에 맞는 조합으로 바꾸고 `rule_flags`에 사유를 남긴다.
- **검토 대상 표시**: 신뢰도가 `review_threshold`보다 낮거나 규칙 위반이 있으면 `review_needed=True`.
- HMC/ATC는 기종 공통 코드라 FA-50으로 학습한 모델을 다른 기종에도 쓴다. 기종 정보는 입력 텍스트에 함께 들어간다.

## 폴더

```
mcc/            파이프라인 코드 (train, predict, batch, evaluate, doctor, rehearse, 선택: translate)
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
이미지는 디스크에서 약 16GB, 반입 파일(.tar.gz)로 약 5.7GB다. 모델은 BGE-M3 약 2.3GB뿐이다.
(선택) 번역 증강용 LLM까지 반입하려면 `LLM=Qwen/Qwen2.5-32B-Instruct bash docker/build_offline.sh`(약 65GB 추가).
빌드 PC에는 디스크 여유가 40GB 이상(LLM 포함 시 110GB 이상) 있어야 한다.

빌드가 막힐 때:
- Docker Hub `429 Too Many Requests` → `BASE_IMAGE=mirror.gcr.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`
- pip/허깅페이스 `CERTIFICATE_VERIFY_FAILED`(사내 프록시) → `CA_CERT=/경로/사내CA.crt`

둘 다 환경 변수로 앞에 붙인다. 예) `BASE_IMAGE=... CA_CERT=... bash docker/build_offline.sh`
이 저장소 폴더도 함께 반입한다(설정 파일과 실행 스크립트를 쓰기 때문).

### 회사 PC에서 개발 환경 없이 받기 (GitHub Releases)

사내 PC에 Docker나 허깅페이스 접속이 없으면 GitHub Actions가 대신 빌드해 Releases에 올린다.

1. 저장소 Actions 탭 > `offline-bundle` > Run workflow (`llm`은 비워 둔다. 번역 증강은 기본으로 쓰지 않는다)
2. 끝나면 Releases의 해당 태그(예: `offline-v0.1.1`)에서 파일을 **전부** 한 폴더로 받는다.
   이미지·코드(`mcc-app.tar.gz`)·bge-m3가 모두 들어 있으므로 저장소 ZIP은 따로 받지 않는다.
3. 그 폴더째 폐쇄망으로 반입해서 `sudo bash install_bundle.sh` 한 번 실행한다.
   체크섬 확인, 조각 합치기, `docker load`, `/srv/mcc/models`에 모델 배치, `/srv/mcc/app`에 코드 설치, `mcc` 명령 등록까지 한다.

이미지 약 5.7GB(조각 4개) + BGE-M3 약 2.3GB.
코드만 바뀐 경우(의존성 그대로)는 `code_only`를 체크해 `mcc-app.tar.gz`와 `install_bundle.sh`만 새로 받아 반입하면 된다.

## 2. 폐쇄망에 설치

```bash
sudo bash install_bundle.sh     # 받은 파일이 있는 폴더에서
nvidia-smi                      # 드라이버 550 이상이어야 CUDA 12.4 이미지가 돈다
mcc --help                      # = bash /srv/mcc/app/docker/run.sh --help
```
그다음 `/srv/mcc/app/configs/default.yaml`에서 `columns:`의 컬럼명만 맞추면 된다. LiteLLM(Qwen)은 필요 없다.
다시 설치해도 이 설정 폴더는 그대로 유지된다. 데이터는 `/srv/mcc/data`(컨테이너 안 `/data`), 결과는 `/srv/mcc/work`(`/work`)에 둔다.

`build_offline.sh`로 직접 만든 파일(`models.tar`)을 반입했다면 예전처럼 `sha256sum -c SHA256SUMS`, `docker load`,
`tar -C /srv/mcc -xf models.tar` 후 저장소 폴더에서 `bash docker/run.sh`로 실행한다.

### (선택) 번역 증강: 폐쇄망의 LiteLLM(Qwen)으로 학습 데이터 번역

기본 흐름에는 필요 없다. 한국어로만 학습한 모델의 외국어 점수(`mcc evaluate`)가 부족할 때 켠다.
번역 LLM은 반입할 필요가 없다. `configs/default.yaml`의 `translate`를 켜고 LiteLLM 주소와 모델 이름을 적고,
키는 파일에 적지 말고 실행할 때 환경 변수로 넘긴다. `enabled: true`이면 `mcc doctor`가 LiteLLM 연결도 점검한다.

```yaml
translate:
  enabled: true
  backend: api
  api:
    base_url: http://<LiteLLM 주소>:4000/v1
    model: <LiteLLM에 등록된 Qwen 모델 이름>
```
```bash
LITELLM_API_KEY=sk-... mcc translate --input /data/train/fa50_ko.xlsx --output /data/train/fa50_aug.xlsx
# LiteLLM이 같은 서버에 있으면 DOCKER_NETWORK=host 를 앞에 붙이고 base_url을 http://localhost:4000/v1 로 둔다
# -> train.augment_files에 /data/train/fa50_aug.xlsx 를 넣고, train.output_dir을 바꿔(예: /work/models/hmc_atc_aug) 다시 mcc train
```

**HMC/ATC 학습에는 bge-m3 가중치 파일이 직접 필요하다.** API는 문장 임베딩만 돌려줄 뿐 모델을 다시 학습시킬 수 없기 때문이다.
서버에 이미 있는 bge-m3 폴더(`config.json`, `model.safetensors` 또는 `pytorch_model.bin`, `tokenizer.json`/`sentencepiece.bpe.model`이 든 폴더)를
`MODELS_DIR`로 마운트하고 `model.encoder` 경로를 맞추면 따로 반입하지 않아도 된다.
WUC 유사도 검색의 임베딩은 `wuc.embed_backend: api`로 바꾸면 LiteLLM의 bge-m3를 쓸 수도 있다.

### 점검과 리허설

```bash
mcc doctor             # 패키지, GPU, 인코더, 설정, 학습 파일 컬럼, WUC 참고자료 점검(번역 증강을 켰으면 LiteLLM도)
mcc doctor --deep      # 인코더 가중치까지 실제로 불러 봄
mcc rehearse           # 가짜 데이터로 점검→학습→평가→예측→배치+요약을 끝까지. 네트워크를 끊고 돈다. LiteLLM 불필요
mcc rehearse --augment # 번역 증강(가짜 API) 단계까지 넣어 검증
```
`doctor`는 줄마다 `[OK ]`, `[W11]`, `[E32]`처럼 찍고 마지막 줄에 문제 코드만 모은다(예: `결과 코드: E32 W11`).
다른 명령도 실패하면 traceback 대신 `[오류 E32] ...` 한 줄과 조치, 로그 파일 위치(`/srv/mcc/work/logs/`)를 보여 준다.
폐쇄망에서 막히면 **그 코드와 한 줄만** 옮겨 적으면 된다. E는 멈춘 오류, W는 경고다.
GPU가 없는 서버에서는 `run.sh`가 알아서 CPU로 돌린다(`GPUS=none`/`all`로 강제 가능).

#### 오류 코드표

| 코드 | 뜻 | 조치 |
|---|---|---|
| E01 | 설정 파일을 읽을 수 없음 | CONFIG 경로와 YAML 들여쓰기를 확인 |
| E02 | 설정에 필요한 항목이 없음 | configs/default.yaml 원본과 비교해 빠진 항목을 채움 |
| E03 | 코드표(codebook) 읽기 실패 | codebook/hmc.csv, atc.csv가 있는지 확인 |
| E10 | 파이썬 패키지 불러오기 실패 | 이미지가 손상됨. install_bundle.sh 다시 실행 |
| W11 | GPU가 안 보임, CPU로 실행 | nvidia-smi 확인. 드라이버 550 이상, nvidia-container-toolkit 필요 |
| E20 | 인코더(bge-m3) 폴더 없음 | model.encoder 경로와 /srv/mcc/models/bge-m3 확인 |
| E21 | 인코더 파일 일부 없음 | config.json, 가중치, 토크나이저 파일이 다 있는지 확인 |
| E22 | 인코더 불러오기 실패(파일 손상 등) | 모델 파일을 다시 복사 |
| E30 | 학습 파일 없음 | train.train_file 경로 확인(컨테이너 안 /data = /srv/mcc/data) |
| E31 | 엑셀/CSV 읽기 실패 | 파일 형식, 시트, 암호 여부 확인 |
| E32 | 필요한 컬럼이 파일에 없음 | configs의 columns: 오른쪽 값을 실제 컬럼명으로 고침 |
| W33 | 코드표에 없는 HMC/ATC 라벨이 많음 | HMC 3자리, ATC 1자리인지 확인. 해당 행은 학습에서 빠짐 |
| E34 | 학습에 쓸 수 있는 행이 없음 | HMC/ATC 컬럼 값과 columns 설정 확인 |
| E35 | 학습된 모델 없음 | 먼저 mcc train 실행, 또는 --model-dir 지정 |
| E36 | 평가 파일에 정답 HMC/ATC가 있는 행이 없음 | hmc, atc 컬럼에 사람이 확인한 정답 코드를 채움 |
| E40 | WUC 참고 파일(GBL/-06) 없음 | wuc.gbl_files, wuc.wuc_table_files 경로 확인 |
| E41 | WUC 참고 파일에 컬럼 없음 | columns.gbl / columns.wuc_table 값 확인 |
| E50 | LiteLLM 키 없음 | LITELLM_API_KEY=sk-... mcc translate ... 로 실행 |
| W50 | LiteLLM 키 없음(번역 안 씀이면 무시) | LITELLM_API_KEY=sk-... mcc ... 로 실행 |
| E51 | LiteLLM 연결 안 됨 | base_url 확인. 같은 서버면 DOCKER_NETWORK=host, 주소는 localhost |
| E52 | LiteLLM에 그 모델 이름이 없음 | translate.api.model을 LiteLLM 모델 이름과 똑같이 |
| E53 | LiteLLM 키가 거부됨(401) | 키 값 확인 |
| E54 | LiteLLM 요청 오류 | 로그 파일의 응답 내용 확인 |
| E60 | 작업 폴더에 쓸 수 없음 | /srv/mcc/work 권한 확인(sudo chown) |
| E70 | GPU 메모리 부족 | train.batch_size 또는 model.max_length를 줄임 |
| E71 | RAM 부족 | batch_size를 줄이거나 다른 작업 종료 |
| E90 | 파일 없음 | 명령에 쓴 경로가 컨테이너 안 경로(/data, /work)인지 확인 |
| E99 | 예상 못 한 오류 | 화면의 마지막 줄(오류 종류와 파일:줄)을 그대로 적어 옴 |

### VS Code Remote-SSH를 폐쇄망에서 쓰기

Remote-SSH는 처음 접속할 때 서버에 VS Code Server를 인터넷에서 받으려 하므로, 폐쇄망에서는 미리 반입해야 한다.
서버에 `~/.vscode-server`가 이미 있고 접속이 되면 할 일이 없다.

1. 워크스테이션 VS Code의 **도움말 > 정보**에서 `커밋` 값(40자리)을 복사한다. 버전이 정확히 같아야 한다.
2. 인터넷 PC에서 Actions > `offline-bundle` > Run workflow에 `vscode_commit`을 넣는다
   (이미지가 이미 있으면 `code_only` 체크). 릴리스에 `vscode-server_<커밋>.tar.gz`, `vscode-cli_<커밋>.tar.gz`가 붙는다.
3. 같이 반입해서 `sudo bash install_bundle.sh`를 하면 접속할 계정의 `~/.vscode-server`에 풀어 둔다.
4. 확장(Python 등)이 필요하면 `.vsix`를 같은 폴더에 넣어 반입한다. `/srv/mcc/vsix`에 복사되고, 원격 창 확장 탭 > `...` > VSIX에서 설치를 고른다.

## 3. 데이터 준비

**컬럼명은 `configs/default.yaml` 맨 위 `columns:` 한곳에서만 바꾼다.** 왼쪽 키는 그대로 두고 오른쪽 값을 실제 엑셀 컬럼명으로 고치면 된다.

| 파일 | 기본 컬럼 | 비고 |
|---|---|---|
| 정비이력 (학습/예측) | record_id, aircraft_type, defect_text, action_text, part_number, hmc, atc, wuc | 예측 파일에는 hmc/atc/wuc가 없어도 된다 |
| GBL 엑셀 | part_number, wuc, nomenclature(품명) (+ 선택: aircraft_type) | 기종별 파일이면 `wuc.gbl_files`에 `aircraft`를 적고 기종 컬럼은 없어도 된다 |
| -06 WUC 목록 | aircraft_type, wuc, wuc_name | IETM의 WUC와 부품명(예: TIRE, MAIN LANDING GEAR)을 엑셀로 옮긴 것. PN은 필요 없다 |

- HMC는 엑셀에서 숫자로 저장돼 앞자리 0이 빠져도(`20` → `020`) 자동으로 복원한다.
- TO 코드표에 없는 HMC/ATC가 붙은 행은 학습에서 빠지고 건수가 로그에 남는다.
- 데이터 기종명이 참고자료에 없으면 `wuc.aircraft_alias`의 대체 기종(예: FA-50PL → FA-50)에서 찾는다.

기본 경로(컨테이너 안 기준): 학습 파일 `/data/train/fa50_ko.xlsx`, 참고자료 `/data/ref/`, 모델 `/models/`, 결과 `/work/`.

## 4. 실행

```bash
# 학습(한국어 데이터만): /work/models/hmc_atc 에 최고 모델과 history.csv(에폭별 정확도) 저장
mcc train

# (선택) 외국어 평가: 정답을 붙인 현지 기록으로 채점
mcc evaluate --input /data/eval/foreign_labeled.xlsx

# 파일 하나 예측
mcc predict --input /data/inbox/sample.xlsx --output /work/out/sample_pred.xlsx

# 주기 실행: inbox의 새 파일을 모두 예측 + 요약 보고서, 처리한 파일은 inbox/processed로 이동
mcc batch --inbox /data/inbox --outbox /work/out
```
매일 자동 실행은 `docker/cron.example`을 crontab에 등록한다.

### 외국어 평가 파일 (`mcc evaluate`)

현지(폴란드·태국·필리핀 등) 정비 기록 수십~수백 건에 사람이 정답 HMC/ATC를 붙인 엑셀이면 된다.
컬럼은 학습 파일과 같고(`columns.data` 설정을 따름), **`lang` 컬럼**(en, pl, th, tl ...)을 넣으면 언어별 점수가 따로 나온다.
기종 컬럼이 있으면 기종별 점수도 나온다. 정답이 코드표에 없는 행은 빠진다.

| record_id | aircraft_type | defect_text | action_text | part_number | hmc | atc | lang |
|---|---|---|---|---|---|---|---|
| PL-001 | FA-50PL | Brak wskazania na MFD | Wymieniono MFD | ... | 242 | R | pl |

결과는 `/work/models/eval/<파일명>__<모델폴더명>.xlsx`(또는 `--output`)에 `요약`(전체·언어별·기종별 HMC/ATC 정확도, top3,
검토 필요 비율), `모델`(번역 증강 사용 여부), `건별`(정답과 예측) 시트로 저장되고 화면에도 표로 나온다.

**번역 증강 효과 비교**: 지금 모델(한국어만)로 한 번 평가해 두고, 나중에 증강을 켜서 `train.output_dir`을 다른 폴더
(예: `/work/models/hmc_atc_aug`)로 학습한 뒤 같은 파일로 `mcc evaluate --model-dir /work/models/hmc_atc_aug`를 돌려
두 결과 파일의 `요약` 시트를 비교한다.

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
docker run --rm --entrypoint python mcc:latest -m pytest -q tests   # 폐쇄망에서도 외부 모델 없이 돈다
```

가짜 데이터로 전체 흐름을 돌려 입력·출력 엑셀을 직접 보려면:
```bash
docker run --rm -v /srv/mcc/work:/work --entrypoint python mcc:latest scripts/demo.py /work/demo /work/demo_tmp
```

## 알려진 한계

- 학습 중 `history.csv`의 점수는 한국어 검증셋(증강을 켰으면 번역본 검증셋 포함) 기준이다. 실제 외국어 성능은
  현지 기록에 정답을 붙여 `mcc evaluate`로 확인한다.
- WUC는 GBL에 PN이 없고 텍스트에 부품명이 분명하지 않으면 정확도가 낮다. 이런 건은 `wuc_source`와 점수로 걸러 검토한다.
- 학습 데이터에 거의 없는 HMC는 잘 맞히지 못한다. 빈도 역수 가중치로 일부 보정한다.
