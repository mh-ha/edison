FROM pytorchlightning/pytorch_lightning:base-cuda-py3.10-torch2.2-cuda12.1.0
# FROM pytorchlightning/pytorch_lightning:base-cuda-py3.10-torch2.0-cuda11.8.0
# FROM pytorchlightning/pytorch_lightning:base-cuda-py3.9-torch1.11-cuda11.3.1

# pip 업그레이드
RUN pip3 install --upgrade pip

# 작업 디렉토리 설정
WORKDIR /workspace

# requirements.txt 파일을 작업 디렉토리에 복사
COPY requirements.txt /workspace/

# requirements.txt에 명시된 Python 패키지 설치
RUN pip3 install -r requirements.txt

# 프로젝트 파일 복사 (Docker build context의 모든 파일을 복사)
COPY . /workspace

# 진입점을 설정하여 컨테이너 실행 시 기본 명령어 설정
ENTRYPOINT ["python3", "run.py"]

# 기본 인자를 CMD로 설정
CMD ["--edison_ae"]