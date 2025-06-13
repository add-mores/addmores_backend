# app/main.py
# 디렉토리: backend/app/

"""
FastAPI 메인 애플리케이션
- 증상 기반 질병 및 의약품 추천 시스템 통합 API
- 멀티 레포 구조의 백엔드 엔트리포인트
- 기술스택: FastAPI, AWS EC2, AWS RDS
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from datetime import datetime
from dotenv import load_dotenv

# ========== 각 API 라우터 import ==========
from app.api import insert_api     # 입력 API (api/insert) - 증상 처리
from app.api import disease_api    # 질병 API (api/disease) - 질병 예측
from app.api import medicine_api   # 의약품 API (api/medicine) - 의약품 추천
from app.api import hospital_api   # 병원 API (api/hospital) - 병원 추천


# ========== LLM API 라우터 import ==========
from app.api import medi_llm_api   # 의약품 LLM API (llm/medicine)
from app.api import hospi_llm_api  # 병원 LLM API (llm/hospital)

# ========== 🆕 LLM 질병 API (조건부 import) ==========
LLM_DISEASE_AVAILABLE = False
try:
    from app.api.llm_disease_api import router as disease_llm_router
    from app.services.disease_faiss_loader import initialize_faiss_loader
    from app.services.disease_embedding_service import initialize_embedding_service
    from app.services.disease_rag_service import initialize_rag_service
    from app.services.disease_service import initialize_disease_service
    LLM_DISEASE_AVAILABLE = True
    print("✅ LLM 질병 API 모듈 로드 성공")
except ImportError as e:
    print(f"⚠️ LLM 질병 API 모듈을 찾을 수 없습니다: {e}")
    LLM_DISEASE_AVAILABLE = False

# 환경변수 로드
load_dotenv()

# ========== 로깅 설정 ==========
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== FastAPI 앱 생성 ==========
app = FastAPI(
    title="증상 기반 질병 및 의약품 추천 시스템",
    description="사용자의 증상을 분석하여 질병과 의약품, 병원을 추천하는 통합 API",
    version="1.0.0",
    docs_url="/docs",      # Swagger UI
    redoc_url="/redoc"     # ReDoc UI
)

# ========== CORS 설정 (Next.js와 연동) ==========
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",    # Next.js 개발 서버
        "http://127.0.0.1:3000",
        "http://localhost:3001",    # 예비 포트
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# ========== API 라우터 등록 ==========

# 1️⃣ 증상 처리 API
app.include_router(
    insert_api.router, 
    tags=["증상 처리"],
    prefix="",  # /api/insert 그대로 사용
)

# 2️⃣ 질병 예측 API (기본)
app.include_router(
    disease_api.router, 
    tags=["질병 예측"],
    prefix="",  # /api/disease 그대로 사용
)
app.include_router(
    medicine_api.router, 
    tags=["의약품 추천"],
    prefix="",  # /api/medicine 그대로 사용
)

# 4️⃣ 의약품 추천 API (LLM)
app.include_router(
    medi_llm_api.router, 
    tags=["의약품 추천 LLM"],
    prefix="",  # /llm/medicine 그대로 사용
)

# 5️⃣ 병원 추천 API (기본)
app.include_router(
    medi_llm_api.router, 
    tags=["의약품 추천 LLM"],
    prefix="",  # /llm/medicine 그대로 사용
)

app.include_router(
    hospital_api.router, 
    tags=["병원 추천"],
    prefix="",  # /api/hospital 그대로 사용
)

app.include_router(
    hospi_llm_api.router, 
    tags=["병원 추천 LLM"],
    prefix="",  # /llm/hospital 그대로 사용
)

if LLM_DISEASE_AVAILABLE:
    app.include_router(
        disease_llm_router,
        tags=["AI 질병 진단"],
        prefix="",  # /api/llm/disease
    )
    logger.info("✅ LLM 질병 API 라우터 등록 완료")

# ========== 루트 엔드포인트 ==========

@app.get("/")
async def root():
    """API 서버 상태 확인 및 사용 가능한 엔드포인트 안내"""
    
    # 기본 엔드포인트 정의
    endpoints = {
        "증상 처리": "/api/insert",
        "질병 추천": "/api/disease", 
        "의약품 추천": "/api/medicine",
        "의약품 추천 LLM": "/llm/medicine",
        "병원 추천": "/api/hospital",
        "병원 추천 LLM": "/llm/hospital",
        "API 문서": "/docs",
        "ReDoc 문서": "/redoc"
    }
    
    # 🆕 LLM 질병 API 엔드포인트 추가 (조건부)
    if LLM_DISEASE_AVAILABLE:
        endpoints["AI 질병 진단"] = "/api/llm/disease"
        endpoints["AI 질병 정보"] = "/api/llm/disease/info"
    
    return {
        "message": "증상 기반 질병 및 의약품 추천 시스템 API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),

        "endpoints": {
            "증상 처리": "/api/insert",
            "질병 추천": "/api/disease", 
            "질병 추천 llm": "/llm/disease",
            "의약품 추천": "/api/medicine",
            "의약품 추천 llm": "/llm/medicine",
            "병원 추천": "/api/hospital",
            "병원 추천 llm": "/llm/hospital",
            "API 문서": "/docs",
            "ReDoc 문서": "/redoc"
        }
    }

# ========== 전체 시스템 헬스 체크 ==========
@app.get("/health")
async def health_check():
    """전체 시스템 헬스 체크"""
    
    # 기본 서비스 상태
    services = {
        "symptoms": "active",      # 증상 처리
        "diseases": "active",      # 질병 예측
        "medications": "active",   # 의약품 추천
        "hospitals": "active"      # 병원 추천
    }
    
    # 🆕 LLM 질병 API 상태 추가 (조건부)
    if LLM_DISEASE_AVAILABLE:
        services["ai_diagnosis"] = "active"
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": services
    }

# ========== 전체 API 목록 ==========
@app.get("/api")
async def api_list():
    """사용 가능한 모든 API 엔드포인트 목록 및 사용법"""
    endpoints = [
        {
            "path": "/api/insert",
            "method": "POST",
            "description": "사용자 증상 입력 및 긍정/부정 세그먼트 분리",
            "input": "{ text: '증상 설명' }",
            "output": "{ original_text, positive, negative }"
        },
        {
            "path": "/api/disease",
            "method": "POST",
            "description": "증상 기반 질병 추천 (상위 5개)",
            "input": "입력 API 결과",
            "output": "{ diseases, departments, disease_names }"
        },
        {
            "path": "/api/medicine",
            "method": "POST",
            "description": "증상 기반 의약품 추천",
            "input": "{ disease_names: [...] }",
            "output": "{ medications }"
        },
        {
            "path": "/llm/medicine",
            "method": "POST",
            "description": "EXAONE LLM 기반 의약품 상담 및 추천",
            "input": {
                "query": "string (자연어 증상 설명 또는 의약품 질문)",
                "age_group": "string (선택사항: '소아', '청소년', '성인', '노인', 기본값: '성인')",
                "is_pregnant": "boolean (선택사항: 임신 여부, 기본값: false)",
                "chronic_conditions": "array (선택사항: 만성질환 목록, 기본값: [])"
            },
            "output": {
                "response": "string (EXAONE LLM이 생성한 자연어 의약품 상담 응답)",
                "recommended_medications": [
                    {
                        "name": "string (의약품명)",
                        "source": "string (추천 출처)",
                        "response_context": "string (응답 컨텍스트)"
                    }
                ],
                "warning": "string (안전 경고 메시지)",
                "session_context": "object (세션 컨텍스트 정보)"
            }
        },
        {
            "path": "/api/hospital",
            "method": "POST",
            "description": "진료과 및 위치 기반 병원 추천",
            "input": "{ departments: [...], location: {...} }",
            "output": "{ hospitals }"
        },
        {
            "path": "/llm/hospital",
            "method": "POST",
            "description": "진료과 및 위치 기반 LLM 병원 추천",
            "input": {
                "address": "string (도로명 주소 또는 지번 주소)",
                "symptom": "string (자연어 증상 설명)",
                "radius": "number (단위: km, 기본값: 1.0)"
            },
            "output": {
                "predicted_deps": [
                    {"department": "...", "score": "..."}
                ],
                "llm_summary": [
                    {
                        "hos_nm": "...",
                        "reason": "...",
                        "add": "...",
                        "deps": ["..."],
                        "distance": "...",
                        "opening_hours": "...",
                        "map_url": "..."
                    }
                ]
            }
        }
    ]

    if LLM_DISEASE_AVAILABLE:
        endpoints.extend([
            {
                "path": "/api/llm/disease",
                "method": "POST",
                "description": "EXAONE LLM + FAISS RAG 기반 AI 질병 진단",
                "input_example": {"message": "머리가 아프고 열이 나요", "context": {}},
                "output_example": {
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
            },
            {
                "path": "/api/llm/disease/info",
                "method": "POST",
                "description": "EXAONE LLM 기반 질병 정보 상세 제공",
                "input_example": {"message": "코로나19에 대해 설명해줘", "context": {}},
                "output_example": {
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
            }
        ])

    return {"available_endpoints": endpoints}

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 실행되는 함수 - 서비스 초기화"""
    
    logger.info("🚀 증상 기반 질병 및 의약품 추천 시스템 API 서버가 시작되었습니다.")
    logger.info("📖 API 문서: http://localhost:8000/docs")
    logger.info("🔗 기본 엔드포인트:")
    logger.info("   - 증상 처리: POST /api/insert")
    logger.info("   - 질병 추천: POST /api/disease")
    logger.info("   - 의약품 추천: POST /api/medicine")
    logger.info("   - 병원 추천: POST /api/hospital")
    logger.info("   - 의약품 LLM: POST /llm/medicine")
    logger.info("   - 병원 LLM: POST /llm/hospital")
    
    # ========== 🆕 LLM 질병 서비스 초기화 (조건부) ==========
    if LLM_DISEASE_AVAILABLE:
        logger.info("   - AI 질병 진단: POST /api/llm/disease")
        logger.info("   - AI 질병 정보: POST /api/llm/disease/info")
        
        logger.info("🔄 LLM 질병 서비스 초기화 시작...")
        initialization_success = True
        
        try:
            # 1️⃣ 임베딩 서비스 초기화
            logger.info("1️⃣ 임베딩 서비스 초기화...")
            if initialize_embedding_service():
                logger.info("✅ 임베딩 서비스 완료")
            else:
                logger.warning("⚠️ 임베딩 서비스 실패")
                initialization_success = False
        except Exception as e:
            logger.warning(f"⚠️ 임베딩 오류: {e}")
            initialization_success = False
        
        if initialization_success:
            try:
                # 2️⃣ FAISS 인덱스 초기화
                logger.info("2️⃣ FAISS 인덱스 초기화...")
                if initialize_faiss_loader():
                    logger.info("✅ FAISS 인덱스 완료")
                else:
                    logger.warning("⚠️ FAISS 인덱스 실패")
                    initialization_success = False
            except Exception as e:
                logger.warning(f"⚠️ FAISS 오류: {e}")
                initialization_success = False
        
        if initialization_success:
            try:
                # 3️⃣ RAG 서비스 초기화
                logger.info("3️⃣ RAG 서비스 초기화...")
                if initialize_rag_service():
                    logger.info("✅ RAG 서비스 완료")
                else:
                    logger.warning("⚠️ RAG 서비스 실패")
                    initialization_success = False
            except Exception as e:
                logger.warning(f"⚠️ RAG 오류: {e}")
                initialization_success = False
        
        if initialization_success:
            try:
                # 4️⃣ 질병 진단 서비스 초기화
                logger.info("4️⃣ 질병 진단 서비스 초기화...")
                if initialize_disease_service():
                    logger.info("✅ 질병 진단 서비스 완료")
                else:
                    logger.warning("⚠️ 질병 진단 서비스 실패")
                    initialization_success = False
            except Exception as e:
                logger.warning(f"⚠️ 질병 진단 오류: {e}")
                initialization_success = False
        
        # 초기화 결과 로깅
        if initialization_success:
            logger.info("🎉 LLM 질병 서비스 초기화 완료!")
        else:
            logger.warning("⚠️ LLM 질병 서비스 초기화 실패 - 기존 API는 정상 동작")
    else:
        logger.info("ℹ️ LLM 질병 API 모듈 없음 - 해당 기능 비활성화")

# ========== 애플리케이션 종료 이벤트 ==========  
@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 실행되는 함수"""
    logger.info("🛑 API 서버가 종료됩니다.")

# ========== 예외 처리 (전역) ==========
@app.exception_handler(500)
async def internal_server_error_handler(request, exc):
    """서버 내부 오류 처리"""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "서버에서 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """404 에러 처리"""
    
    # 기본 사용가능한 엔드포인트
    available_endpoints = [
        "/api/insert", "/api/disease", "/api/medicine", "/api/hospital",
        "/llm/medicine", "/llm/hospital"
    ]
    
    # 🆕 LLM 질병 API 엔드포인트 추가 (조건부)
    if LLM_DISEASE_AVAILABLE:
        available_endpoints.extend(["/api/llm/disease", "/api/llm/disease/info"])
    
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": f"요청한 경로 '{request.url.path}'를 찾을 수 없습니다.",
            "available_endpoints": available_endpoints,
            "timestamp": datetime.now().isoformat()
        }
    )

# ========== 개발용 실행 설정 ==========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # 개발 모드에서만 사용
        log_level="info"
    )