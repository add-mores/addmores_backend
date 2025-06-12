"""
간소화된 FastAPI 메인 앱 - LLM 통합 의료 챗봇 API
위치: backend/app/llm/main_llm.py

🎯 목적: 설정 파일 없이 바로 실행 가능한 버전
🚀 실행: python -m app.llm.main_llm
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import uvicorn
import os


# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 전역 서비스 인스턴스
chat_service_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 라이프사이클 관리"""
    global chat_service_instance
    
    # 애플리케이션 시작 시 초기화
    logger.info("🚀 LLM 통합 의료 챗봇 API 초기화 시작...")
    
    try:
        # 🔧 동적 import로 초기화 지연
        logger.info("🧠 LLM 서비스 초기화 중...")
        
        from app.llm.services.integrated_chat_service import IntegratedChatAPIService
        chat_service_instance = IntegratedChatAPIService()
        
        logger.info("✅ LLM 서비스 초기화 완료")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ 서비스 초기화 실패: {e}")
        logger.warning("LLM 기능 없이 서비스를 시작합니다.")
        chat_service_instance = None
        yield
    finally:
        # 애플리케이션 종료 시 정리
        logger.info("🔄 LLM 서비스 종료 중...")
        if chat_service_instance:
            try:
                chat_service_instance.cleanup()
            except:
                pass
        logger.info("✅ LLM 서비스 종료 완료")

# FastAPI 앱 생성
app = FastAPI(
    title="LLM Integrated Medical Chatbot API",
    description="""
    🏥 **통합 의료 챗봇 API v6**
    
    CLI 기반 의료 챗봇을 FastAPI로 변환한 서비스입니다.
    
    ## 🔍 주요 기능:
    - **질병 진단**: 증상 기반 스마트한 차별화 질문
    - **의약품 추천**: 질병 연계 의약품 정보 제공
    - **RAG 검색**: 6개 clean_ 파일 기반 지식 베이스
    - **세션 관리**: 대화 문맥 유지 및 세션 상태 관리
    - **의도 분류**: 사용자 메시지 의도 자동 파악
    
    ## 🚀 기술 스택:
    - **LLM**: EXAONE 3.5:7.8b
    - **임베딩**: KM-BERT (madatnlp/km-bert)
    - **벡터 검색**: FAISS 인덱스
    - **백엔드**: FastAPI
    
    ## ⚠️ 주의사항:
    이 서비스는 의료 전문가의 진단을 대체할 수 없습니다.
    """,
    version="6.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS 설정 (개발 환경 친화적)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://127.0.0.1:3000",
        "http://localhost:8000",  # 기존 API와 연동
        "http://127.0.0.1:8000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 라우터 등록
app.include_router(
    chat_router, 
    prefix="/api/v1",
    tags=["채팅"]
)

# 헬스 체크 엔드포인트
@app.get("/health", tags=["헬스체크"])
async def health_check():
    """서비스 상태 확인"""
    global chat_service_instance
    
    if not chat_service_instance:
        return {
            "status": "partial",
            "service": "LLM Integrated Medical Chatbot API",
            "version": "6.0.0",
            "timestamp": "2024-12-12T14:30:22",
            "message": "LLM 서비스 초기화 중이거나 실패했습니다.",
            "llm_available": False
        }
    
    # 서비스 상태 확인
    try:
        service_status = chat_service_instance.get_service_status()
        
        return {
            "status": "healthy",
            "service": "LLM Integrated Medical Chatbot API",
            "version": "6.0.0",
            "timestamp": chat_service_instance.get_current_timestamp(),
            "llm_available": True,
            "services": service_status
        }
    except Exception as e:
        return {
            "status": "degraded",
            "service": "LLM Integrated Medical Chatbot API",
            "version": "6.0.0",
            "message": f"서비스 상태 확인 실패: {str(e)}",
            "llm_available": False
        }

# 서비스 정보 엔드포인트
@app.get("/info", tags=["정보"])
async def service_info():
    """서비스 상세 정보"""
    global chat_service_instance
    
    base_info = {
        "service_name": "LLM Integrated Medical Chatbot API",
        "version": "6.0.0",
        "description": "CLI 기반 의료 챗봇의 FastAPI 변환 버전",
        "features": [
            "질병 진단 (스마트한 차별화 질문)",
            "의약품 추천 및 정보 제공", 
            "RAG 기반 의료 지식 검색",
            "세션 기반 대화 문맥 유지",
            "사용자 의도 자동 분류"
        ],
        "endpoints": {
            "chat": "/api/v1/chat",
            "session_reset": "/api/v1/chat/session/{session_id}/reset",
            "session_status": "/api/v1/chat/session/{session_id}/status"
        },
        "models": {
            "llm": "EXAONE 3.5:7.8b",
            "embedding": "madatnlp/km-bert",
            "vector_search": "FAISS"
        }
    }
    
    if chat_service_instance:
        try:
            service_status = chat_service_instance.get_service_status()
            base_info["llm_status"] = service_status
        except:
            base_info["llm_status"] = "unavailable"
    else:
        base_info["llm_status"] = "not_initialized"
    
    return base_info

# 간단한 테스트 엔드포인트
@app.get("/test", tags=["테스트"])
async def test_endpoint():
    """간단한 테스트 엔드포인트"""
    return {
        "message": "LLM API 서버가 정상 동작 중입니다!",
        "timestamp": "2024-12-12T14:30:22",
        "status": "ok"
    }

# 전역 서비스 인스턴스 접근 함수
def get_chat_service():
    """전역 채팅 서비스 인스턴스 반환"""
    global chat_service_instance
    if not chat_service_instance:
        raise HTTPException(
            status_code=503, 
            detail="채팅 서비스가 초기화되지 않았습니다. 잠시 후 다시 시도해주세요."
        )
    return chat_service_instance

# 개발 환경에서 직접 실행 시
if __name__ == "__main__":
    print("🚀 LLM 통합 의료 챗봇 API 서버 시작...")
    print("📍 URL: http://localhost:8001")
    print("📖 API 문서: http://localhost:8001/docs")
    print("🔄 종료: Ctrl+C")
    print("⚠️ 첫 실행 시 모델 로딩으로 인해 시간이 소요될 수 있습니다.")
    
    # 환경 변수 설정 (필요시)
    os.environ.setdefault("LLM_ENABLED", "true")
    os.environ.setdefault("EXAONE_SERVER_URL", "http://localhost:11434")
    
    uvicorn.run(
        "app.llm.main_llm:app",
        host="0.0.0.0",
        port=8001,
        reload=True,  # 개발 환경에서만
        log_level="info"
    )