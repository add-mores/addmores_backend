"""
통합 채팅 API 라우터 - 원본 버전 (클래스명 확인됨)
위치: backend/app/api/chat.py

🎯 목적: 
- 기존 4개 API와는 별개의 새로운 통합 챗봇 API 제공
- 대화형 방식으로 질병 진단부터 의약품 추천까지 한 번에 처리
- 기존 API들은 그대로 유지 (호환성)

📋 기능: 
- EXAONE 기반 자연어 대화
- 스마트한 차별화 질문으로 정확한 진단
- 질병 진단 + 의약품 추천 자동 연계
- 세션 기반 대화 기록 관리
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import logging
import traceback
from datetime import datetime
import asyncio

# 통합 서비스 import
from app.llm.services.integrated_chat_service import IntegratedChatAPIService
from app.llm.api.chat_models import ChatRequest, ChatResponse

# 로깅 설정
logger = logging.getLogger(__name__)

# API 라우터 초기화
router = APIRouter(prefix="/api/chat", tags=["통합 챗봇"])

# 🔧 전역 서비스 인스턴스 (싱글톤 패턴)
_chat_service: Optional[IntegratedChatAPIService] = None
_service_initializing = False

async def get_chat_service() -> IntegratedChatAPIService:
    """
    통합 채팅 서비스 인스턴스 반환 (지연 초기화)
    
    📋 기능:
    - 서비스 싱글톤 관리
    - 초기화 중복 방지
    - 오류 상황 핸들링
    """
    global _chat_service, _service_initializing
    
    if _chat_service is not None:
        return _chat_service
    
    if _service_initializing:
        # 다른 요청이 초기화 중인 경우 대기
        while _service_initializing:
            await asyncio.sleep(0.1)
        if _chat_service is not None:
            return _chat_service
    
    try:
        _service_initializing = True
        logger.info("🚀 통합 채팅 서비스 초기화 시작...")
        
        # 서비스 초기화 (시간이 걸릴 수 있음)
        _chat_service = IntegratedChatAPIService()
        
        logger.info("✅ 통합 채팅 서비스 초기화 완료!")
        return _chat_service
        
    except Exception as e:
        logger.error(f"❌ 서비스 초기화 실패: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"서비스 초기화 실패: {str(e)}"
        )
    finally:
        _service_initializing = False

# =============================================================================
# 🎯 메인 채팅 엔드포인트
# =============================================================================

@router.post("/message", response_model=ChatResponse)
async def process_chat_message(request: ChatRequest) -> ChatResponse:
    """
    🤖 통합 채팅 메시지 처리
    
    📋 기능:
    - 사용자 메시지 의도 자동 분류
    - 질병 진단 (스마트한 차별화 질문 포함)
    - 의약품 추천
    - 질병/의약품 정보 검색
    - 세션 기반 대화 관리
    
    🔄 처리 플로우:
    1. 사용자 메시지 수신
    2. 의도 분류 (질병진단/의약품추천/정보검색 등)
    3. 해당 서비스로 라우팅
    4. 자연어 응답 생성 (EXAONE)
    5. 세션 컨텍스트 업데이트
    """
    try:
        # 서비스 인스턴스 가져오기
        chat_service = await get_chat_service()
        
        # 요청 로깅
        logger.info(f"💬 채팅 메시지 처리: session={request.session_id}, message={request.message[:50]}...")
        
        # 통합 서비스로 메시지 처리
        response = await chat_service.process_chat_message(request)
        
        # 응답 로깅
        logger.info(f"✅ 처리 완료: intent={response.intent}, session={response.session_id}")
        
        return response
        
    except HTTPException:
        # FastAPI HTTPException은 그대로 전달
        raise
    except Exception as e:
        # 예상치 못한 오류 처리
        logger.error(f"❌ 채팅 메시지 처리 중 오류: {e}")
        logger.error(traceback.format_exc())
        
        # 사용자 친화적 오류 응답
        error_response = ChatResponse(
            response="⚠️ 죄송합니다. 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
            intent="error",
            session_id=request.session_id or "error_session",
            status="error",
            error_code="INTERNAL_ERROR",
            error_message=str(e)
        )
        
        return error_response

# =============================================================================
# 🔧 세션 관리 엔드포인트들
# =============================================================================

@router.post("/session/reset/{session_id}")
async def reset_session(session_id: str):
    """
    🔄 세션 초기화
    
    📋 기능:
    - 특정 세션의 대화 기록 및 컨텍스트 초기화
    - 질병 진단 정보, 추천 의약품 등 모든 상태 리셋
    """
    try:
        chat_service = await get_chat_service()
        success = await chat_service.reset_session(session_id)
        
        if success:
            logger.info(f"🔄 세션 초기화 성공: {session_id}")
            return {
                "success": True,
                "message": f"세션 {session_id}이 초기화되었습니다.",
                "timestamp": datetime.now().isoformat()
            }
        else:
            logger.warning(f"⚠️ 세션을 찾을 수 없음: {session_id}")
            return {
                "success": False,
                "message": f"세션 {session_id}을 찾을 수 없습니다.",
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"❌ 세션 초기화 오류: {e}")
        raise HTTPException(status_code=500, detail=f"세션 초기화 실패: {str(e)}")

@router.get("/session/status/{session_id}")
async def get_session_status(session_id: str):
    """
    📊 세션 상태 조회
    
    📋 반환 정보:
    - 세션 존재 여부
    - 생성 시간, 마지막 활동 시간
    - 메시지 수, 진단된 질병, 추천된 의약품 등
    """
    try:
        chat_service = await get_chat_service()
        session_info = await chat_service.get_session_status(session_id)
        
        logger.info(f"📊 세션 상태 조회: {session_id}")
        return {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            **session_info
        }
        
    except Exception as e:
        logger.error(f"❌ 세션 상태 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"세션 상태 조회 실패: {str(e)}")

@router.get("/session/active")
async def get_active_sessions():
    """
    📋 활성 세션 목록 조회
    
    📋 기능:
    - 현재 활성화된 모든 세션 정보
    - 세션별 통계 (메시지 수, 진단 여부 등)
    - 전체 시스템 통계
    """
    try:
        chat_service = await get_chat_service()
        sessions_info = await chat_service.get_active_sessions()
        
        logger.info(f"📋 활성 세션 조회: {sessions_info.get('total_sessions', 0)}개")
        return {
            "timestamp": datetime.now().isoformat(),
            **sessions_info
        }
        
    except Exception as e:
        logger.error(f"❌ 활성 세션 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"활성 세션 조회 실패: {str(e)}")

# =============================================================================
# 🔧 시스템 상태 및 진단 엔드포인트들
# =============================================================================

@router.get("/health")
async def health_check():
    """
    🏥 시스템 건강 상태 체크
    
    📋 확인 항목:
    - 서비스 초기화 상태
    - 각 서비스 컴포넌트 상태
    - 메모리 사용량, 활성 세션 수 등
    """
    try:
        if _chat_service is None:
            return {
                "status": "initializing",
                "message": "서비스가 아직 초기화되지 않았습니다.",
                "timestamp": datetime.now().isoformat()
            }
        
        # 서비스 상태 정보 가져오기
        service_status = _chat_service.get_service_status()
        
        overall_status = "healthy" if service_status.get("initialized", False) else "unhealthy"
        
        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "service_details": service_status
        }
        
    except Exception as e:
        logger.error(f"❌ 헬스 체크 오류: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@router.get("/status")
async def get_system_status():
    """
    📊 전체 시스템 상태 정보
    
    📋 상세 정보:
    - 서비스 컴포넌트 별 상태
    - 로딩된 데이터 통계
    - 처리 통계 (총 메시지 수, 성공/실패률 등)
    """
    try:
        basic_info = {
            "service_name": "통합 의료 챗봇 API",
            "version": "6.0",
            "status": "running" if _chat_service else "not_initialized",
            "timestamp": datetime.now().isoformat()
        }
        
        if _chat_service:
            service_status = _chat_service.get_service_status()
            basic_info["details"] = service_status
        else:
            basic_info["details"] = {
                "initialized": False,
                "message": "서비스가 초기화되지 않았습니다."
            }
        
        return basic_info
        
    except Exception as e:
        logger.error(f"❌ 시스템 상태 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"시스템 상태 조회 실패: {str(e)}")

# =============================================================================
# 🔧 개발/디버깅용 엔드포인트들 (배포 시 제거 가능)
# =============================================================================

@router.post("/debug/process", include_in_schema=False)
async def debug_process_message(
    message: str,
    session_id: Optional[str] = None,
    include_debug_info: bool = True
):
    """
    🐛 디버깅용 메시지 처리 엔드포인트
    
    📋 추가 정보:
    - 의도 분류 과정
    - 각 서비스 처리 단계
    - 내부 상태 변화
    """
    try:
        request = ChatRequest(
            message=message,
            session_id=session_id
        )
        
        chat_service = await get_chat_service()
        response = await chat_service.process_chat_message(request)
        
        if include_debug_info:
            # 디버깅 정보 추가
            debug_info = {
                "service_stats": chat_service.get_service_status(),
                "processing_time": datetime.now().isoformat()
            }
            
            # response에 debug_info 추가 (필드가 있는 경우)
            response_dict = response.dict()
            response_dict["debug_info"] = debug_info
            
            return response_dict
        
        return response
        
    except Exception as e:
        logger.error(f"❌ 디버그 처리 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# 🚀 라우터 이벤트 핸들러
# =============================================================================

@router.on_event("startup")
async def startup_event():
    """라우터 시작 이벤트"""
    logger.info("🤖 통합 채팅 API 라우터 시작됨")

@router.on_event("shutdown") 
async def shutdown_event():
    """라우터 종료 이벤트"""
    global _chat_service
    if _chat_service:
        _chat_service.cleanup()
    logger.info("🤖 통합 채팅 API 라우터 종료됨")