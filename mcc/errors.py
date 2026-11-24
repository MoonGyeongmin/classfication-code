"""짧은 오류 코드. 폐쇄망에서 오류를 밖으로 옮겨 적을 때 코드 몇 글자만 적으면 원인을 알 수 있게 한다.

E로 시작하면 멈춘 오류, W는 경고(계속 진행). 같은 표가 README '오류 코드표'에 있다.
"""

CODES = {
    "E01": ("설정 파일을 읽을 수 없음", "CONFIG 경로와 YAML 들여쓰기를 확인"),
    "E02": ("설정에 필요한 항목이 없음", "configs/default.yaml 원본과 비교해 빠진 항목을 채움"),
    "E03": ("코드표(codebook) 읽기 실패", "codebook/hmc.csv, atc.csv가 있는지 확인"),
    "E10": ("파이썬 패키지 불러오기 실패", "이미지가 손상됨. install_bundle.sh 다시 실행"),
    "W11": ("GPU가 안 보임, CPU로 실행", "nvidia-smi 확인. 드라이버 550 이상, nvidia-container-toolkit 필요"),
    "W12": ("이 GPU를 이미지의 torch가 지원하지 않음", "doctor에 찍힌 sm_ 값과 torch 빌드 목록을 적어 옴. 이미지 재빌드 필요"),
    "E20": ("인코더(bge-m3) 폴더 없음", "model.encoder 경로와 /srv/mcc/models/bge-m3 확인"),
    "E21": ("인코더 파일 일부 없음", "config.json, 가중치, 토크나이저 파일이 다 있는지 확인"),
    "E22": ("인코더 불러오기 실패(파일 손상 등)", "모델 파일을 다시 복사"),
    "E30": ("학습 파일 없음", "train.train_file 경로 확인(컨테이너 안 /data = /srv/mcc/data)"),
    "E31": ("엑셀/CSV 읽기 실패", "파일 형식, 시트, 암호 여부 확인"),
    "E32": ("필요한 컬럼이 파일에 없음", "configs의 columns: 오른쪽 값을 실제 컬럼명으로 고침"),
    "W33": ("코드표에 없는 HMC/ATC 라벨이 많음", "HMC 3자리, ATC 1자리인지 확인. 해당 행은 학습에서 빠짐"),
    "E34": ("학습에 쓸 수 있는 행이 없음", "HMC/ATC 컬럼 값과 columns 설정 확인"),
    "E35": ("학습된 모델 없음", "먼저 mcc train 실행, 또는 --model-dir 지정"),
    "E36": ("평가 파일에 정답 HMC/ATC가 있는 행이 없음", "hmc, atc 컬럼에 사람이 확인한 정답 코드를 채움"),
    "E40": ("WUC 참고 파일(GBL/-06) 없음", "wuc.gbl_files, wuc.wuc_table_files 경로 확인"),
    "E41": ("WUC 참고 파일에 컬럼 없음", "columns.gbl / columns.wuc_table 값 확인"),
    "E50": ("LiteLLM 키 없음", "LITELLM_API_KEY=sk-... mcc translate ... 로 실행"),
    "W50": ("LiteLLM 키 없음(번역 안 씀이면 무시)", "LITELLM_API_KEY=sk-... mcc ... 로 실행"),
    "E51": ("LiteLLM 연결 안 됨", "base_url 확인. 같은 서버면 DOCKER_NETWORK=host, 주소는 localhost"),
    "E52": ("LiteLLM에 그 모델 이름이 없음", "translate.api.model을 LiteLLM 모델 이름과 똑같이"),
    "E53": ("LiteLLM 키가 거부됨(401)", "키 값 확인"),
    "E54": ("LiteLLM 요청 오류", "로그 파일의 응답 내용 확인"),
    "E60": ("작업 폴더에 쓸 수 없음", "/srv/mcc/work 권한 확인(sudo chown)"),
    "E70": ("GPU 메모리 부족", "train.batch_size 또는 model.max_length를 줄임"),
    "E71": ("RAM 부족", "batch_size를 줄이거나 다른 작업 종료"),
    "E90": ("파일 없음", "명령에 쓴 경로가 컨테이너 안 경로(/data, /work)인지 확인"),
    "E99": ("예상 못 한 오류", "화면의 마지막 줄(오류 종류와 파일:줄)을 그대로 적어 옴"),
}


class MccError(RuntimeError):
    def __init__(self, code, detail=""):
        self.code = code
        self.detail = detail
        super().__init__(f"{code} {CODES.get(code, ('', ''))[0]}: {detail}".rstrip(": "))


def describe(code):
    what, fix = CODES.get(code, ("알 수 없는 코드", ""))
    return what, fix
