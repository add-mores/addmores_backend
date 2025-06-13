# app/main.py
# 디렉토리: backend/app/

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from datetime import datetime
from dotenv import load_dotenv

# 각 API 라우터 import
from app.api import insert_api     # 입력 API (api/insert)
from app.api import disease_api     # 질병 API (api/disease)
from app.api import medicine_api  # 의약품 API (api/medicine)
from app.api import hospital_api    # 병원 API (api/hospital)

#from app.api import dis_llm_api    # 질병 llm API (llm/disease) - 기존 주석 유지
from app.api import medi_llm_api    # 의약품 llm API (llm/medicine)
from app.api import hospi_llm_api    # 병원 llm API (llm/hospital) - 팀원 코드 유지

# ========== 🆕 추가: LLM 질병 API (조건부 import) ==========
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

load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="증상 기반 질병 및 의약품 추천 시스템",
    description="사용자의 증상을 분석하여 질병과 의약품, 병원을 추천하는 통합 API",
    version="1.0.0",
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
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE","OPTIONS"],
    allow_headers=["*"],
)


# 각 API 라우터 등록
app.include_router(
    insert_api.router, 
    tags=["증상 처리"],
    prefix="",  # /api/insert 그대로 사용
)

app.include_router(
    disease_api.router, 
    tags=["질병 예측"],
    prefix="",  # /api/disease 그대로 사용
)

# app.include_router(
#     dis_llm_api.router, 
#     tags=["질병 예측 LLM"],
#     prefix="",  # /llm/disease 그대로 사용
# )

app.include_router(
    medicine_api.router, 
    tags=["의약품 추천"],
    prefix="",  # /api/medicine 그대로 사용
)

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

# ========== 🆕 추가: LLM 질병 API 라우터 (조건부 등록) ==========
if LLM_DISEASE_AVAILABLE:
    app.include_router(
        disease_llm_router,
        tags=["AI 질병 진단"],
        prefix="",  # /api/llm/disease
    )
    logger.info("✅ LLM 질병 API 라우터 등록 완료")


# 루트 엔드포인트
@app.get("/")
async def root():
    """API 서버 상태 확인"""
    endpoints = {
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
    
    # 🆕 LLM 질병 API 추가 (조건부)
    if LLM_DISEASE_AVAILABLE:
        endpoints["AI 질병 진단"] = "/api/llm/disease"
    
    return {
        "message": "증상 기반 질병 및 의약품 추천 시스템 API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "endpoints": endpoints
    }

# 전체 시스템 헬스 체크
@app.get("/health")
async def health_check():
    """전체 시스템 헬스 체크"""
    services = {
        "symptoms": "active",
        "diseases": "active", 
        "medications": "active",
        "hospitals": "active"
    }
    
    # 🆕 LLM 질병 API 상태 추가 (조건부)
    if LLM_DISEASE_AVAILABLE:
        services["ai_diagnosis"] = "active"
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": services
    }

# 전체 API 목록
@app.get("/api")
async def api_list():
    """사용 가능한 모든 API 엔드포인트 목록"""
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
            "description": "증상 기반 LLM 의약품 추천",
            "input": "{ query: '증상 텍스트', age_group?: '성인', is_pregnant?: bool, chronic_conditions?: [...] }",
            "output": "{ recommended_diseases: [...], explanation: 'LLM 설명 텍스트' }"
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
                    {
                        "department": "...",
                        "score": "..."
                    }
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
    
    # 🆕 LLM 질병 API 엔드포인트 추가 (조건부)
    if LLM_DISEASE_AVAILABLE:
        endpoints.extend([
            {
                "path": "/api/llm/disease",
                "method": "POST",
                "description": "AI 기반 질병 진단 (FAISS + EXAONE LLM)",
                "input": "{ message: '증상 설명' }",
                "output": "{ diagnosis, confidence, department, symptoms, recommendations }"
            },
            {
                "path": "/api/llm/disease/info",
                "method": "POST",
                "description": "AI 기반 질병 정보 제공",
                "input": "{ message: '질병명 또는 질병 관련 질문' }",
                "output": "{ diagnosis, recommendations, department }"
            }
        ])
    
    return {
        "available_endpoints": endpoints
    }

# 애플리케이션 시작 이벤트
@app.on_event("startup")
async def startup_event():
    """서버 시작 시 실행되는 함수"""
    logger.info("🚀 증상 기반 질병 및 의약품 추천 시스템 API 서버가 시작되었습니다.")
    logger.info("📖 API 문서: http://localhost:8000/docs")
    logger.info("🔗 엔드포인트:")
    logger.info("   - 증상 처리: POST /api/insert")
    logger.info("   - 질병 추천: POST /api/disease")
    logger.info("   - 의약품 추천: POST /api/medicine")
    logger.info("   - 병원 추천: POST /api/hospital")
    
    # ========== 🆕 추가: LLM 질병 서비스 초기화 (조건부) ==========
    if LLM_DISEASE_AVAILABLE:
        logger.info("   - AI 질병 진단: POST /api/llm/disease")
        logger.info("   - AI 질병 정보: POST /api/llm/disease/info")
        
        logger.info("🔄 LLM 질병 서비스 초기화 시작...")
        initialization_success = True
        
        try:
            # 1. 임베딩 서비스 초기화
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
                # 2. FAISS 인덱스 초기화
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
                # 3. RAG 서비스 초기화
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
                # 4. 질병 진단 서비스 초기화
                logger.info("4️⃣ 질병 진단 서비스 초기화...")
                if initialize_disease_service():
                    logger.info("✅ 질병 진단 서비스 완료")
                else:
                    logger.warning("⚠️ 질병 진단 서비스 실패")
                    initialization_success = False
            except Exception as e:
                logger.warning(f"⚠️ 질병 진단 오류: {e}")
                initialization_success = False
        
        if initialization_success:
            logger.info("🎉 LLM 질병 서비스 초기화 완료!")
        else:
            logger.warning("⚠️ LLM 질병 서비스 초기화 실패 - 기존 API는 정상 동작")
    else:
        logger.info("ℹ️ LLM 질병 API 모듈 없음 - 해당 기능 비활성화")

from fastapi.responses import JSONResponse
from datetime import datetime

# 애플리케이션 종료 이벤트  
@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 실행되는 함수"""
    logger.info("🛑 API 서버가 종료됩니다.")

# 예외 처리 (전역)
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
    available_endpoints = ["/api/insert", "/api/disease", "/api/medicine", "/api/hospital"]
    
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


# 개발용 정보 (배포 시 제거)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # 개발 모드
        log_level="info"
    )