"""
질병 LLM API 엔드포인트
위치: ~/backend/app/api/llm_disease_api.py

🎯 목적: 질병 진단 LLM API 엔드포인트 제공
📋 기능:
   - POST /api/llm/disease - 질병 진단
   - POST /api/llm/disease/info - 질병 정보 (POST 방식으로 변경)
   - GET /api/llm/disease/status - 서비스 상태
   - 에러 핸들링 및 로깅
   - 세션 관련 코드 제거됨

⚙️ 의존성: FastAPI, Pydantic
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Query, Path
from fastapi.responses import JSONResponse

from ..models.disease_models import DiseaseRequest, DiseaseResponse
from ..services.disease_service import get_disease_service, DiseaseService
from ..utils.disease_exceptions import (
    DiseaseValidationError, DiseaseDiagnosisError, ExaoneConnectionError,
    FaissLoadError, EmbeddingModelLoadError, RagSearchError
)

logger = logging.getLogger(__name__)

# 라우터 생성
router = APIRouter(prefix="/api/llm", tags=["질병 LLM"])


# 의존성 주입: 질병 서비스
def get_initialized_disease_service() -> DiseaseService:
    """초기화된 질병 서비스 반환"""
    service = get_disease_service()
    
    if not service.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="질병 서비스가 초기화되지 않았습니다. 서버 관리자에게 문의하세요."
        )
    
    return service


@router.post("/disease", response_model=DiseaseResponse)
async def diagnose_disease(
    request: DiseaseRequest,
    service: DiseaseService = Depends(get_initialized_disease_service)
) -> DiseaseResponse:
    """
    질병 진단 API
    
    **주요 기능:**
    - 증상 기반 질병 진단
    - FAISS 벡터 검색 + RAG 검색
    - EXAONE LLM 추론
    - 진료과 자동 매핑
    
    **요청 예시:**
    ```json
    {
        "message": "머리가 아프고 열이 나요",
        "context": {}
    }
    ```
    
    **응답 예시:**
    ```json
    {
        "diagnosis": "감기",
        "confidence": 0.85,
        "department": "내과",
        "symptoms": ["두통", "발열"],
        "recommendations": "충분한 휴식과 수분 섭취를...",
        "reasoning": "두통과 발열은 감기의 대표적인 증상입니다...",
        "disclaimer": "⚠️ 이는 참고용이며...",
        "response_time": 2.3,
        "timestamp": "2024-06-12T15:30:00Z"
    }
    ```
    """
    try:
        logger.info(f"🔍 질병 진단 요청: '{request.message}'")
        
        # 질병 진단 실행
        response = service.diagnose_disease(request)
        
        logger.info(f"✅ 질병 진단 완료: {response.diagnosis} ({response.response_time:.2f}초)")
        
        return response
        
    except DiseaseValidationError as e:
        logger.warning(f"⚠️ 입력 검증 실패: {e}")
        raise HTTPException(status_code=400, detail=f"입력 검증 실패: {str(e)}")
        
    except ExaoneConnectionError as e:
        logger.error(f"❌ EXAONE 연결 오류: {e}")
        raise HTTPException(status_code=503, detail="AI 모델 서버에 연결할 수 없습니다. 잠시 후 다시 시도해주세요.")
        
    except DiseaseDiagnosisError as e:
        logger.error(f"❌ 진단 오류: {e}")
        raise HTTPException(status_code=500, detail=f"진단 처리 중 오류가 발생했습니다: {str(e)}")
        
    except Exception as e:
        logger.error(f"❌ 예상치 못한 오류: {e}")
        raise HTTPException(status_code=500, detail="서버 내부 오류가 발생했습니다. 관리자에게 문의하세요.")


@router.post("/disease/info", response_model=DiseaseResponse)
async def get_disease_info(
    request: DiseaseRequest,
    service: DiseaseService = Depends(get_initialized_disease_service)
) -> DiseaseResponse:
    """
    질병 정보 제공 API (POST 방식)
    
    **주요 기능:**
    - 질병명 기반 상세 정보 제공
    - 자연어 질문 처리
    - FAISS 벡터 검색 + RAG 검색
    - EXAONE LLM 의학 정보 생성
    
    **지원하는 입력 형태:**
    - 단순 질병명: "감기", "두통", "편두통"
    - 자연어 문장: "감기에 대해 알려주세요"
    - 질문 형태: "두통 원인이 뭔가요?"
    - 코로나19: "코로나", "COVID-19", "코로나19"
    
    **요청 예시:**
    ```json
    {
        "message": "코로나19에 대해 설명해줘",
        "context": {}
    }
    ```
    
    **응답 예시:**
    ```json
    {
        "diagnosis": "코로나19",
        "confidence": 1.0,
        "department": "내과",
        "symptoms": [],
        "recommendations": "**질병 정의**: 코로나19(COVID-19)는 SARS-CoV-2 바이러스에 의한 감염성 호흡기 질환입니다...",
        "reasoning": "'코로나19'에 대한 의료 정보입니다.",
        "disclaimer": "⚠️ 이는 참고용이며 실제 진료를 대체하지 않습니다...",
        "response_time": 1.8,
        "timestamp": "2024-06-13T10:15:30Z"
    }
    ```
    
    **기존 DiseaseResponse 형식 그대로 반환**
    """
    try:
        logger.info(f"📚 질병 정보 요청: '{request.message}'")
        
        # ✅ 질병명 추출
        disease_name = extract_disease_name_from_message(request.message)
        logger.info(f"🔍 추출된 질병명: '{disease_name}' (원문: '{request.message}')")
        
        # ✅ 기존 서비스 로직 그대로 사용
        response = service.get_disease_info(disease_name)
        
        logger.info(f"✅ 질병 정보 제공 완료: {disease_name} ({response.response_time:.2f}초)")
        
        return response
        
    except DiseaseValidationError as e:
        logger.warning(f"⚠️ 입력 검증 실패: {e}")
        raise HTTPException(status_code=400, detail=f"입력 검증 실패: {str(e)}")
        
    except ExaoneConnectionError as e:
        logger.error(f"❌ EXAONE 연결 오류: {e}")
        raise HTTPException(status_code=503, detail="AI 모델 서버에 연결할 수 없습니다.")
        
    except DiseaseDiagnosisError as e:
        logger.error(f"❌ 정보 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"정보 조회 중 오류가 발생했습니다: {str(e)}")
        
    except Exception as e:
        logger.error(f"❌ 예상치 못한 오류: {e}")
        raise HTTPException(status_code=500, detail="서버 내부 오류가 발생했습니다.")


@router.get("/disease/status")
async def get_service_status(
    service: DiseaseService = Depends(get_disease_service)  # 초기화 체크 안함
) -> Dict[str, Any]:
    """
    질병 서비스 상태 확인 API
    
    **주요 기능:**
    - 서비스 초기화 상태
    - FAISS 인덱스 로드 상태
    - 임베딩 모델 상태
    - RAG 서비스 상태
    - EXAONE 연결 상태
    
    **응답 예시:**
    ```json
    {
        "service_name": "질병 LLM API",
        "version": "1.0.0",
        "status": "healthy",
        "details": {
            "is_initialized": true,
            "embedding_service_loaded": true,
            "faiss_loader_loaded": true,
            "rag_service_initialized": true,
            "exaone_connected": true,
            "disease_metadata_count": 1500
        }
    }
    ```
    """
    try:
        logger.debug("📊 서비스 상태 확인 요청")
        
        # 서비스 상태 조회
        service_status = service.get_service_status()
        
        # 전반적인 건강 상태 판단
        is_healthy = (
            service_status.get("is_initialized", False) and
            service_status.get("embedding_service_loaded", False) and
            service_status.get("faiss_loader_loaded", False)
        )
        
        response = {
            "service_name": "질병 LLM API",
            "version": "1.0.0",
            "status": "healthy" if is_healthy else "unhealthy",
            "timestamp": service_status.get("timestamp"),
            "details": service_status
        }
        
        logger.debug(f"✅ 서비스 상태: {'정상' if is_healthy else '비정상'}")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ 상태 확인 오류: {e}")
        return {
            "service_name": "질병 LLM API",
            "version": "1.0.0",
            "status": "error",
            "error": str(e),
            "details": {}
        }


@router.get("/disease/health")
async def health_check() -> Dict[str, str]:
    """
    간단한 헬스 체크 API
    
    **주요 기능:**
    - 기본적인 API 응답 확인
    - 로드 밸런서용 헬스 체크
    
    **응답:**
    ```json
    {
        "status": "ok",
        "message": "질병 LLM API가 실행 중입니다"
    }
    ```
    """
    return {
        "status": "ok",
        "message": "질병 LLM API가 실행 중입니다"
    }


# =============================================================================
# 유틸리티 함수 - 질병명 추출 로직
# =============================================================================

import re
from typing import Optional

def extract_disease_name_from_message(message: str) -> str:
    """
    메시지에서 질병명 추출
    
    지원하는 입력 형태:
    - 단순 질병명: "감기", "두통", "편두통"
    - 자연어 문장: "감기에 대해 알려주세요"
    - 질문 형태: "두통 원인이 뭔가요?"
    - 비교 요청: "편두통과 긴장성 두통 차이"
    """
    if not message or not message.strip():
        return message
    
    message = message.strip()
    
    # 1. 단순 질병명인 경우 (1-2단어) 그대로 반환
    words = message.split()
    if len(words) <= 2:
        # 단순 조사 제거
        cleaned = re.sub(r'(이|가|은|는|을|를|의|에|로|으로|과|와|도)$', '', message)
        return cleaned if cleaned else message
    
    # 2. 주요 질병명 직접 매칭 (우선순위 높음)
    disease_keywords = [
        # 일반적인 질병
        '감기', '독감', '몸살', '발열', '인후염', '목감기',
        '위염', '장염', '급성위장염', '과민성대장증후군', '역류성식도염',
        '복통', '설사', '구토', '변비', '소화불량',
        
        # 신경계 질환
        '두통', '편두통', '긴장성두통', '군발두통',
        '신경병증', '다발성신경병증', '말초신경병증', '당뇨병성신경병증',
        '수근관증후군', '손목터널증후군', '신경압박증후군',
        '안면신경마비', '삼차신경통', '좌골신경통',
        
        # 근골격계
        '관절염', '류마티스관절염', '퇴행성관절염',
        '근육통', '허리통증', '목통증', '어깨통증', '무릎통증',
        '디스크', '추간판탈출증', '척추관협착증',
        
        # 호흡기
        '기관지염', '폐렴', '천식', '만성폐쇄성폐질환',
        '비염', '알레르기비염', '축농증', '부비동염',
        
        # 피부
        '아토피', '아토피피부염', '습진', '두드러기', '피부염',
        '건선', '여드름', '무좀',
        
        # 내분비/대사
        '당뇨병', '갑상선기능항진증', '갑상선기능저하증',
        '고혈압', '고지혈증', '대사증후군',
        
        # 감염성 질환
        '코로나19', '코로나', 'COVID-19', 'covid-19',
        
        # 기타
        '빈혈', '철결핍성빈혈', '우울증', '불안장애', '수면장애'
    ]
    
    # 긴 질병명부터 매칭 (더 구체적인 것 우선)
    sorted_diseases = sorted(disease_keywords, key=len, reverse=True)
    
    message_lower = message.lower()
    for disease in sorted_diseases:
        if disease in message_lower:
            logger.debug(f"직접 매칭된 질병명: '{disease}' (원문: '{message}')")
            return disease
    
    # 3. 패턴 매칭으로 질병명 추출
    extraction_patterns = [
        # "~에 대해" 패턴
        r'([가-힣]+(?:병|염|증|통|장애|증후군|질환)?)에\s*대해',
        r'([가-힣]+(?:병|염|증|통|장애|증후군|질환)?)\s*에\s*대해',
        
        # "~이/가 뭐/무엇" 패턴  
        r'([가-힣]+(?:병|염|증|통|장애|증후군|질환)?)\s*(?:이|가)\s*(?:뭐|무엇)',
        
        # "~의 원인/증상/치료" 패턴
        r'([가-힣]+(?:병|염|증|통|장애|증후군|질환)?)\s*(?:의|)\s*(?:원인|증상|치료|진단|예방)',
        
        # "~와/과 ~의 차이" 패턴에서 첫 번째 질병
        r'([가-힣]+(?:병|염|증|통|장애|증후군|질환)?)\s*(?:와|과)',
        
        # "~를/을 알고 싶어" 패턴  
        r'([가-힣]+(?:병|염|증|통|장애|증후군|질환)?)\s*(?:를|을)\s*(?:알고|알려)',
        
        # "~란/는 무엇" 패턴
        r'([가-힣]+(?:병|염|증|통|장애|증후군|질환)?)\s*(?:란|는)\s*(?:무엇|뭐)',
        
        # "~에 걸렸어/앓고 있어" 패턴
        r'([가-힣]+(?:병|염|증|통|장애|증후군|질환)?)\s*에\s*(?:걸렸|앓고)',
        
        # 질병 접미사 패턴
        r'([가-힣]+(?:병|염|증|통|장애|증후군|질환))',
        
        # 일반적인 질병명 패턴 (한글 2글자 이상)
        r'([가-힣]{2,})'
    ]
    
    for pattern in extraction_patterns:
        matches = re.findall(pattern, message)
        if matches:
            # 첫 번째 매치를 사용하되, 의미있는 길이인지 확인
            disease_candidate = matches[0].strip()
            
            # 너무 일반적인 단어들 필터링
            common_words = [
                '그것', '이것', '저것', '무엇', '어떤', '하나', '전부', '모든', '모두',
                '사람', '환자', '의사', '병원', '약물', '치료', '증상', '원인',
                '방법', '정보', '설명', '내용', '결과', '상태', '경우', '때문',
                '그래서', '하지만', '그런데', '그리고', '또한', '역시', '정말'
            ]
            
            if (len(disease_candidate) >= 2 and 
                disease_candidate not in common_words and
                not disease_candidate.isdigit()):
                
                logger.debug(f"패턴 매칭된 질병명: '{disease_candidate}' (원문: '{message}')")
                return disease_candidate
    
    # 4. 추출 실패시 전체 메시지 사용 (벡터 검색에서 처리)
    logger.debug(f"질병명 추출 실패, 전체 메시지 사용: '{message}'")
    return message


def validate_extracted_disease_name(disease_name: str) -> str:
    """
    추출된 질병명 검증 및 정제
    """
    if not disease_name:
        return disease_name
    
    # 조사 제거
    cleaned = re.sub(r'(이|가|은|는|을|를|의|에|로|으로|과|와|도|만|도|라도|이라도)$', '', disease_name)
    
    # 공백 정리
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    # 최소 길이 체크
    if len(cleaned) < 1:
        return disease_name
    
    return cleaned


# 메인 함수에서 사용할 때
def extract_disease_name_from_message_safe(message: str) -> str:
    """안전한 질병명 추출 (예외 처리 포함)"""
    try:
        extracted = extract_disease_name_from_message(message)
        validated = validate_extracted_disease_name(extracted)
        return validated
    except Exception as e:
        logger.warning(f"질병명 추출 중 오류: {e}, 원본 메시지 반환")
        return message