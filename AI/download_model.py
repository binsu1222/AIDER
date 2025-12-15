import os
from huggingface_hub import snapshot_download

# 저장할 로컬 폴더 이름
local_dir = "./local_model"

print(f"[{local_dir}] 폴더에 모델 다운로드를 시작합니다...")

# 모델을 특정 폴더로 강제 다운로드
snapshot_download(
    repo_id="jhgan/ko-sroberta-multitask",
    local_dir=local_dir,
    local_dir_use_symlinks=False  # 심볼릭 링크 사용 안 함 (실제 파일 저장)
)

print("다운로드 완료! 이제 인터넷 연결 없이도 실행 가능합니다.")