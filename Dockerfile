# 1. Python slim base image
FROM python:3.12-slim

# 2. 시스템 의존 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libffi-dev \
    gcc \
    g++ \
    default-libmysqlclient-dev \
    curl \
    git \
    openjdk-17-jdk \
    python3-dev \
    python3-pip \
    fonts-nanum \
    && apt-get clean

# 3. konlpy에서 필요로 하는 Java 환경 변수 설정
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
#ENV PDM_VENV_IN_PROJECT=true

# 4. pdm 설치
RUN pip install --upgrade pip && pip install pdm

# 5. 작업 디렉토리 설정
WORKDIR /app

# 6. pyproject.toml과 pdm.lock 복사
COPY pyproject.toml pdm.lock ./

# 7. 패키지 설치
#RUN pdm install --prod || (cat /root/.local/state/pdm/log/* && false)
RUN pip install -U pdm
# disable update check
ENV PDM_CHECK_UPDATE=false
RUN pdm install --check --prod --no-editable

# 8. 전체 소스 복사
COPY . .

# 9. streamlit 실행 명령어 (필요 시 변경)
CMD ["pdm", "run", "streamlit", "run", "main.py", "--server.port", "8501", "--server.address", "127.0.0.1"]

