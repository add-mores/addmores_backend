"""
질병 모듈 상수 정의
위치: ~/backend/app/utils/disease_constants.py

🎯 목적: 질병 모듈에서 사용하는 모든 상수 값 관리
📋 기능: 설정값, 경로, 매핑 정보 등 중앙 집중 관리
⚙️ 사용: 모든 질병 관련 서비스에서 import
"""

import os
from pathlib import Path

# =============================================================================
# 파일 경로 설정
# =============================================================================

# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent.parent.parent

# FAISS 인덱스 디렉토리
DISEASE_FAISS_DIR = PROJECT_ROOT / "disease_faiss_indexes"

# FAISS 파일명들
FAISS_FILES = {
    "disease_key": "disease_key_index.index",
    "disease_full": "disease_full_index.index", 
    "rag_qa": "rag_qa_index.index",
    "rag_medical": "rag_medical_index.index"
}

# 메타데이터 파일명들
METADATA_FILES = {
    "disease": "disease_metadata.pkl",
    "rag_qa": "rag_qa_metadata.pkl", 
    "rag_medical": "rag_medical_metadata.pkl"
}

# =============================================================================
# 임베딩 모델 설정
# =============================================================================

EMBEDDING_MODEL_NAME = "madatnlp/km-bert"
EMBEDDING_MAX_LENGTH = 128  # 토큰 최대 길이

# =============================================================================
# EXAONE LLM 설정
# =============================================================================

EXAONE_BASE_URL = "http://localhost:11434"
EXAONE_MODEL_NAME = "exaone3.5:7.8b"
EXAONE_CONFIG = {
    "temperature": 0.1,
    "top_p": 0.9,
    "top_k": 40,
    "repeat_penalty": 1.1,
    "num_predict": 3000,
    "stop": ["사용자:", "환자:", "Human:", "Assistant:"]
}

# =============================================================================
# 입력 검증 설정
# =============================================================================

# 메시지 길이 제한
MIN_MESSAGE_LENGTH = 2
MAX_MESSAGE_LENGTH = 1000

# 허용되지 않는 문자 패턴
INVALID_PATTERNS = [
    r'^[^가-힣a-zA-Z0-9\s,.!?]+$',  # 특수문자만 있는 경우
    r'^[ㄱ-ㅎㅏ-ㅣ]+$',              # 자음/모음만 있는 경우
]

# =============================================================================
# 질병별 진료과 매핑 테이블
# =============================================================================

DEPARTMENT_MAPPING = {
    # 일반 내과 질환
    "감기": "내과",
    "독감": "내과", 
    "몸살": "내과",
    "발열": "내과",
    "기침": "내과",
    "콧물": "내과",
    "인후통": "내과",
    "목감기": "내과",
    
    # 소화기 질환
    "복통": "내과",
    "위염": "내과", 
    "장염": "내과",
    "설사": "내과",
    "구토": "내과",
    "소화불량": "내과",
    "변비": "내과",
    
    # 신경과 질환
    "두통": "신경과",
    "편두통": "신경과",
    "어지러움": "신경과",
    "현기증": "신경과",
    "머리아픔": "신경과",
    
    # 정형외과 질환
    "요통": "정형외과",
    "허리아픔": "정형외과",
    "목아픔": "정형외과",
    "어깨아픔": "정형외과",
    "무릎아픔": "정형외과",
    "관절염": "정형외과",
    "근육통": "정형외과",
    "삐끗": "정형외과",
    
    # 피부과 질환
    "습진": "피부과",
    "아토피": "피부과",
    "두드러기": "피부과",
    "가려움": "피부과",
    "발진": "피부과",
    
    # 이비인후과 질환
    "중이염": "이비인후과",
    "귀아픔": "이비인후과",
    "코막힘": "이비인후과",
    "비염": "이비인후과",
    "부비동염": "이비인후과",
    
    # 안과 질환
    "눈아픔": "안과",
    "결막염": "안과",
    "다래끼": "안과",
    
    # 비뇨기과 질환
    "방광염": "비뇨기과",
    "요로감염": "비뇨기과",
    
    # 부인과 질환
    "생리통": "산부인과",
    "생리불순": "산부인과",
}

# 기본 진료과 (매핑에 없을 경우)
DEFAULT_DEPARTMENT = "내과"

# =============================================================================
# 응답 관련 상수
# =============================================================================

# 면책 조항
DISCLAIMER_TEXT = "⚠️ 이는 참고용이며 실제 진료를 대체하지 않습니다. 증상이 지속되거나 악화될 경우 반드시 의료진의 진료를 받으시기 바랍니다."

# 응답 시간 제한 (초)
MAX_RESPONSE_TIME = 60.0

# 신뢰도 임계값
MIN_CONFIDENCE_THRESHOLD = 0.3

# =============================================================================
# 로깅 설정
# =============================================================================

# 로그 형식
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# 로그 레벨
LOG_LEVEL = "INFO"

# =============================================================================
# 헬퍼 함수들
# =============================================================================

def get_faiss_file_path(file_key: str) -> str:
    """FAISS 파일 전체 경로 반환"""
    if file_key in FAISS_FILES:
        return os.path.join(str(DISEASE_FAISS_DIR), FAISS_FILES[file_key])
    elif file_key in METADATA_FILES:
        return os.path.join(str(DISEASE_FAISS_DIR), METADATA_FILES[file_key])
    else:
        raise ValueError(f"알 수 없는 파일 키: {file_key}")


def get_department(disease_name: str) -> str:
    """질병명으로 진료과 반환 (매핑에 없으면 기본값)"""
    if not disease_name:
        return DEFAULT_DEPARTMENT
    
    disease_lower = disease_name.lower()
    
    # 직접 매핑 검사
    for disease, department in DEPARTMENT_MAPPING.items():
        if disease in disease_lower:
            return department
    
    return DEFAULT_DEPARTMENT


def is_valid_faiss_dir() -> bool:
    """FAISS 디렉토리 존재 여부 확인"""
    return DISEASE_FAISS_DIR.exists()


# =============================================================================
# Export할 상수들
# =============================================================================

__all__ = [
    # 파일 경로
    "DISEASE_FAISS_DIR",
    "FAISS_FILES", 
    "METADATA_FILES",
    
    # 모델 설정
    "EMBEDDING_MODEL_NAME",
    "EMBEDDING_MAX_LENGTH",
    "EXAONE_BASE_URL",
    "EXAONE_MODEL_NAME", 
    "EXAONE_CONFIG",
    
    # 검증 설정
    "MIN_MESSAGE_LENGTH",
    "MAX_MESSAGE_LENGTH",
    "INVALID_PATTERNS",
    
    # 진료과 매핑
    "DEPARTMENT_MAPPING",
    "DEFAULT_DEPARTMENT",
    
    # 응답 설정
    "DISCLAIMER_TEXT",
    "MAX_RESPONSE_TIME",
    "MIN_CONFIDENCE_THRESHOLD",
    
    # 로깅
    "LOG_FORMAT",
    "LOG_LEVEL",
    
    # 헬퍼 함수
    "get_faiss_file_path",
    "get_department",
    "is_valid_faiss_dir",
]