# 1. 파이썬 3.11 슬림 버전 사용
FROM python:3.11-slim

# 2. 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 3. 작업 디렉토리 설정
WORKDIR /app

# 4. 의존성 파일 복사 (수정됨: AI/ 제거)
# 현재 폴더(.)에 있는 requirements.txt를 컨테이너로 복사
COPY requirements.txt .

# 5. 라이브러리 설치
RUN pip install --no-cache-dir -r requirements.txt

# 6. 소스 코드 복사 (수정됨: AI/ 제거)
# 현재 폴더(.)의 모든 파일을 컨테이너(.)로 복사
COPY . .

# 7. 환경 변수 설정
ENV PYTHONUNBUFFERED=1

# 8. 서버 실행 명령
CMD uvicorn main_api:app --host 0.0.0.0 --port $PORT