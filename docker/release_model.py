"""[GitHub Actions] 허깅페이스 모델을 파일 하나씩 받아 Release에 올린다(러너 디스크보다 큰 LLM도 되게).

사용법: python docker/release_model.py <HF 저장소> <폴더명> <릴리스 태그>
2GB가 넘는 파일은 1900MB 조각(<파일>.part-aa, -ab ...)으로 나눈다. 자산 이름은
models__<폴더명>__<경로의 /를 __로> 이고, docker/install_bundle.sh가 원래 경로로 되돌린다.
"""
import fnmatch, hashlib, os, subprocess, sys, tempfile
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

PART = 1900 * 1024 * 1024
KEEP = ["*.json", "*.safetensors", "*.bin", "*.model", "*.txt", "*.py", "tokenizer*"]
SKIP = ["onnx/*", "*.pt", "*.onnx", "*.h5", "*.msgpack", "*.ot", "imgs/*"]


def pick(files):
    files = [f for f in files if any(fnmatch.fnmatch(f, p) for p in KEEP) and not any(fnmatch.fnmatch(f, p) for p in SKIP)]
    if any(f.endswith(".safetensors") for f in files):  # 같은 가중치를 두 번 받지 않는다
        files = [f for f in files if not f.endswith(".bin")]
    return files


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main(repo, name, tag):
    files = pick(HfApi().list_repo_files(repo))
    print(f"{repo}: {len(files)}개 파일", flush=True)
    sums = Path(f"SHA256SUMS.models__{name}")
    with tempfile.TemporaryDirectory(dir=".") as tmp, open(sums, "w") as out:
        for f in files:
            src = Path(hf_hub_download(repo, f, local_dir=tmp))
            asset = Path(tmp) / f"models__{name}__{f.replace('/', '__')}"
            src.rename(asset)
            if asset.stat().st_size > PART:
                subprocess.run(["split", "-b", str(PART), str(asset), f"{asset}.part-"], check=True)
                asset.unlink()
            for a in sorted(Path(tmp).glob(asset.name + "*")):
                out.write(f"{sha256(a)}  {a.name}\n"); out.flush()
                subprocess.run(["gh", "release", "upload", tag, str(a), "--clobber"], check=True)
                print(f"업로드 {a.name} ({a.stat().st_size / 1e9:.2f} GB)", flush=True)
                a.unlink()
    subprocess.run(["gh", "release", "upload", tag, str(sums), "--clobber"], check=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
