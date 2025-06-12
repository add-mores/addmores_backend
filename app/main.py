"""
통합 의료 시스템 API 서버 - 기존 4개 + 신규 통합 챗봇 1개
위치: backend/app/main.py

🎯 구성:
- 기존 4개 API: 증상처리, 질병추천, 의약품추천, 병원추천 (이전 버전 호환성)
- 신규 1개 API: 통합 챗봇 (EXAONE + 스마트한 차별화 질문)

🔧 기술스택: FastAPI + Next.js + AWS EC2 + AWS RDS
"""

from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from datetime import datetime
from dotenv import load_dotenv
import asyncio

# 🔧 기존 4개 API 라우터 import (이전 버전 유지)
from app.api import insert_api     # 입력 API (api/insert)
from app.api import disease_api    # 질병 API (api/disease)
from app.api import medicine_api   # 의약품 API (api/medicine)
from app.api import hospital_api   # 병원 API (api/hospital)

# 🆕 신규 통합 챗봇 API 라우터 import
from app.api import chat           # 통합 챗봇 API (api/chat)

load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="통합 의료 시스템 API v2.0",
    description="""
🏥 **통합 의료 시스템 API v2.0**

**기존 4개 API (v1.0 호환성 유지):**
- 증상 처리 API
- 질병 추천 API  
- 의약품 추천 API
- 병원 추천 API

**🆕 신규 통합 챗봇 API (v2.0):**
- EXAONE 3.5:7.8b 기반 자연어 대화
- 스마트한 차별화 질문 시스템
- 질병 진단 + 의약품 추천 통합
- 세션 기반 대화 관리

🤖 **권장 사용법:** 새로운 프로젝트는 `/api/chat` 엔드포인트 사용
    """,
    version="2.0.0",
    docs_url="/docs",      # Swagger UI
    redoc_url="/redoc"     # ReDoc UI
)

# CORS 설정 (Next.js와 연동)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",    # Next.js 개발 서버
        "http://127.0.0.1:3000",
        "http://localhost:3001",    # 예비 포트
        "http://localhost:3002",    # 추가 포트
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# =============================================================================
# 🔧 기존 4개 API 라우터 등록 (v1.0 호환성 유지)
# =============================================================================

app.include_router(
    insert_api.router, 
    tags=["v1.0 - 증상 처리"],
    prefix="",  # /api/insert 그대로 사용
)

app.include_router(
    disease_api.router, 
    tags=["v1.0 - 질병 추천"],
    prefix="",  # /api/disease 그대로 사용
)

app.include_router(
    medicine_api.router, 
    tags=["v1.0 - 의약품 추천"],
    prefix="",  # /api/medicine 그대로 사용
)

app.include_router(
    hospital_api.router, 
    tags=["v1.0 - 병원 추천"],
    prefix="",  # /api/hospital 그대로 사용
)

# =============================================================================
# 🆕 신규 통합 챗봇 API 라우터 등록 (v2.0)
# =============================================================================

app.include_router(
    chat.router,
    tags=["v2.0 - 통합 챗봇"],
    prefix="",  # /api/chat 사용
)

# =============================================================================
# 🏠 루트 및 시스템 엔드포인트들
# =============================================================================

@app.get("/")
async def root():
    """API 서버 메인 정보"""
    return {
        "message": "통합 의료 시스템 API v2.0",
        "version": "2.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "apis": {
            "v1_legacy": {
                "description": "기존 분리된 API들 (호환성 유지)",
                "endpoints": {
                    "증상 처리": "/api/insert",
                    "질병 추천": "/api/disease", 
                    "의약품 추천": "/api/medicine",
                    "병원 추천": "/api/hospital"
                }
            },
            "v2_integrated": {
                "description": "🆕 통합 챗봇 API (권장)",
                "endpoints": {
                    "통합 채팅": "/api/chat/message",
                    "세션 관리": "/api/chat/session",
                    "시스템 상태": "/api/chat/health"
                }
            },
            "documentation": {
                "swagger_ui": "/docs",
                "redoc": "/redoc"
            }
        },
        "recommendation": "🤖 새로운 프로젝트는 /api/chat/message 사용을 권장합니다"
    }

@app.get("/health")
async def health_check():
    """전체 시스템 헬스 체크"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "services": {
            "v1_apis": {
                "symptoms": "active",
                "diseases": "active", 
                "medications": "active",
                "hospitals": "active"
            },
            "v2_chatbot": "active"
        }
    }

@app.get("/api")
async def api_overview():
    """전체 API 개요"""
    return {
        "api_version": "2.0.0",
        "total_endpoints": 5,
        "legacy_apis": [
            {
                "path": "/api/insert",
                "method": "POST", 
                "description": "사용자 증상 입력 및 긍정/부정 세그먼트 분리",
                "version": "v1.0"
            },
            {
                "path": "/api/disease",
                "method": "POST",
                "description": "증상 기반 질병 추천 (상위 5개)",
                "version": "v1.0"
            },
            {
                "path": "/api/medicine", 
                "method": "POST",
                "description": "질병명 기반 의약품 추천",
                "version": "v1.0"
            },
            {
                "path": "/api/hospital",
                "method": "POST", 
                "description": "진료과 및 위치 기반 병원 추천",
                "version": "v1.0"
            }
        ],
        "integrated_api": {
            "path": "/api/chat/message",
            "method": "POST",
            "description": "🆕 통합 의료 챗봇 - 질병 진단부터 의약품 추천까지 한 번에",
            "version": "v2.0",
            "features": [
                "EXAONE 3.5:7.8b 기반 자연어 대화",
                "스마트한 차별화 질문으로 정확한 진단",
                "질병-의약품 연계 추천",
                "세션 기반 대화 기록 관리",
                "사용자 맞춤형 안전성 필터링"
            ]
        }
    }

# =============================================================================
# 🚀 애플리케이션 이벤트 핸들러
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 실행되는 함수"""
    logger.info("🚀 통합 의료 시스템 API v2.0 서버가 시작되었습니다.")
    logger.info("📖 API 문서: http://localhost:8000/docs")
    logger.info("🔗 총 5개 엔드포인트:")
    logger.info("   📊 기존 API (v1.0 호환성):")
    logger.info("      - 증상 처리: POST /api/insert")
    logger.info("      - 질병 추천: POST /api/disease")
    logger.info("      - 의약품 추천: POST /api/medicine")
    logger.info("      - 병원 추천: POST /api/hospital")
    logger.info("   🤖 신규 통합 챗봇 (v2.0):")
    logger.info("      - 통합 채팅: POST /api/chat/message")
    logger.info("   💡 권장: 새 프로젝트는 /api/chat/message 사용")
    
    # 🚀 백그라운드에서 챗봇 서비스 미리 로딩 (옵션)
    async def preload_chatbot():
        try:
            logger.info("🔄 백그라운드에서 챗봇 서비스 사전 로딩 중...")
            # 첫 번째 요청 시 대기시간 단축을 위한 사전 로딩
            # 실제 구현은 chat.py의 get_chat_service() 함수에서 처리
        except Exception as e:
            logger.warning(f"⚠️ 백그라운드 로딩 실패 (정상 작동에는 영향 없음): {e}")
    
    # 백그라운드 태스크로 실행
    asyncio.create_task(preload_chatbot())

@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 실행되는 함수"""
    logger.info("🛑 통합 의료 시스템 API v2.0 서버가 종료됩니다.")

# =============================================================================
# 🔧 전역 예외 처리
# =============================================================================

@app.exception_handler(500)
async def internal_server_error_handler(request, exc):
    """서버 내부 오류 처리"""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "서버에서 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
            "timestamp": datetime.now().isoformat(),
            "version": "2.0.0"
        }
    )

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """404 에러 처리"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": f"요청한 경로 '{request.url.path}'를 찾을 수 없습니다.",
            "available_endpoints": {
                "v1_legacy": ["/api/insert", "/api/disease", "/api/medicine", "/api/hospital"],
                "v2_integrated": ["/api/chat/message", "/api/chat/session", "/api/chat/health"]
            },
            "recommendation": "새로운 기능은 /api/chat/message를 사용해보세요",
            "timestamp": datetime.now().isoformat()
        }
    )

# =============================================================================
# 🔧 개발용 정보 (배포 시 제거)
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # 개발 모드
        log_level="info"
    )