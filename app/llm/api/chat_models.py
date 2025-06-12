"""
통합 채팅 API 모델 정의 + 라우터 (Pydantic V2 호환)
위치: backend/app/llm/api/chat_models.py

🎯 목적: 채팅 API의 요청/응답 데이터 모델 정의 + FastAPI 라우터
📋 기능: Pydantic 모델로 API 스키마 정의 및 데이터 검증 + API 엔드포인트
🔧 수정: 
- Pydantic V2 호환성 (schema_extra → json_schema_extra)
- FastAPI 라우터 추가하여 완전한 API 모듈로 구성
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum
import logging

# 로깅 설정
logger = logging.getLogger(__name__)

# =============================================================================
# 🔧 기본 타입 정의
# =============================================================================

class IntentType(str, Enum):
    """의도 분류 타입"""
    DISEASE_DIAGNOSIS = "disease_diagnosis"          # 질병 진단
    MEDICATION_RECOMMEND = "medication_recommend"    # 의약품 추천  
    DISEASE_INFO = "disease_info"                   # 질병 정보 검색
    MEDICATION_INFO = "medication_info"             # 의약품 정보 검색
    DISEASE_TO_MEDICATION = "disease_to_medication" # 질병-의약품 연계
    SESSION_RESET = "session_reset"                 # 세션 초기화
    GENERAL = "general"                            # 일반 대화
    ERROR = "error"                                # 오류 상황

class UserProfile(BaseModel):
    """사용자 프로필 정보"""
    age_group: Optional[str] = Field("성인", description="연령대 (소아/성인/고령자)")
    is_pregnant: Optional[bool] = Field(False, description="임신 여부")
    chronic_conditions: Optional[List[str]] = Field([], description="만성질환 목록")
    allergies: Optional[List[str]] = Field([], description="알레르기 목록")
    
    class Config:
        json_schema_extra = {
            "example": {
                "age_group": "성인",
                "is_pregnant": False,
                "chronic_conditions": ["고혈압"],
                "allergies": ["페니실린"]
            }
        }

class QuestioningState(BaseModel):
    """차별화 질문 상태"""
    is_questioning: bool = Field(False, description="질문 모드 활성 여부")
    current_question: Optional[str] = Field(None, description="현재 질문")
    question_count: int = Field(0, description="질문 개수")
    
    class Config:
        json_schema_extra = {
            "example": {
                "is_questioning": True,
                "current_question": "기침이나 가래 증상이 있으신가요?",
                "question_count": 1
            }
        }

class SessionContext(BaseModel):
    """세션 컨텍스트 정보"""
    last_disease: Optional[str] = Field(None, description="최근 진단된 질병")
    symptoms: Optional[List[str]] = Field([], description="감지된 증상들")
    mentioned_symptoms: Optional[List[str]] = Field([], description="사용자가 언급한 증상들")
    initial_symptoms_text: Optional[str] = Field(None, description="초기 증상 텍스트")
    questioning_state: Optional[QuestioningState] = Field(QuestioningState(), description="질문 상태")
    medications: Optional[List[Dict[str, Any]]] = Field([], description="추천된 의약품들")
    diagnosis_time: Optional[str] = Field(None, description="진단 시간")
    
    class Config:
        json_schema_extra = {
            "example": {
                "last_disease": "감기",
                "symptoms": ["두통", "발열", "기침"],
                "mentioned_symptoms": ["머리아픔", "열"],
                "initial_symptoms_text": "머리가 아프고 열이 나요",
                "questioning_state": {
                    "is_questioning": False,
                    "current_question": None,
                    "question_count": 2
                },
                "medications": [
                    {"name": "타이레놀", "effect": "해열진통"}
                ],
                "diagnosis_time": "2024-01-15T10:30:00"
            }
        }

# =============================================================================
# 🔄 요청 모델
# =============================================================================

class ChatRequest(BaseModel):
    """채팅 요청 모델"""
    message: str = Field(..., description="사용자 메시지", min_length=1, max_length=1000)
    session_id: Optional[str] = Field(None, description="세션 ID (미제공시 자동 생성)")
    user_profile: Optional[UserProfile] = Field(None, description="사용자 프로필 정보")
    
    @validator('message')
    def validate_message(cls, v):
        """메시지 검증"""
        if not v or not v.strip():
            raise ValueError('메시지는 공백일 수 없습니다')
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "머리가 아프고 열이 나는데 기침은 없어요",
                "session_id": "session_123456",
                "user_profile": {
                    "age_group": "성인",
                    "is_pregnant": False,
                    "chronic_conditions": [],
                    "allergies": []
                }
            }
        }

# =============================================================================
# 🔄 응답 모델
# =============================================================================

class ChatResponse(BaseModel):
    """채팅 응답 모델"""
    response: str = Field(..., description="챗봇 응답 메시지")
    intent: IntentType = Field(..., description="분류된 의도")
    session_id: str = Field(..., description="세션 ID")
    context: Optional[SessionContext] = Field(None, description="세션 컨텍스트")
    status: str = Field("success", description="처리 상태 (success/error)")
    
    # 오류 정보 (선택적)
    error_code: Optional[str] = Field(None, description="오류 코드")
    error_message: Optional[str] = Field(None, description="오류 메시지")
    
    # 메타 정보 (선택적)
    processing_time_ms: Optional[int] = Field(None, description="처리 시간 (밀리초)")
    timestamp: Optional[str] = Field(None, description="응답 생성 시간")
    
    class Config:
        json_schema_extra = {
            "example": {
                "response": "증상을 분석한 결과 감기의 가능성이 높습니다. 기침이나 가래 증상이 있으신가요?",
                "intent": "disease_diagnosis",
                "session_id": "session_123456",
                "context": {
                    "last_disease": "감기",
                    "symptoms": ["두통", "발열"],
                    "questioning_state": {
                        "is_questioning": True,
                        "current_question": "기침이나 가래 증상이 있으신가요?",
                        "question_count": 1
                    }
                },
                "status": "success",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }

class SessionResetResponse(BaseModel):
    """세션 초기화 응답"""
    message: str = Field(..., description="초기화 메시지")
    session_id: str = Field(..., description="초기화된 세션 ID")
    status: str = Field(default="success", description="상태")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="초기화 시간")

class SessionStatusResponse(BaseModel):
    """세션 상태 응답"""
    session_id: str = Field(..., description="세션 ID")
    exists: bool = Field(..., description="세션 존재 여부")
    context: Optional[SessionContext] = Field(None, description="세션 컨텍스트 (존재하는 경우)")
    created_at: Optional[str] = Field(None, description="세션 생성 시간")
    last_activity: Optional[str] = Field(None, description="마지막 활동 시간")
    message_count: int = Field(default=0, description="메시지 수")
    status: str = Field(default="success", description="상태")

class SimpleResponse(BaseModel):
    """간단한 응답 모델 (기존 호환성용)"""
    message: str
    status: str = "success"

# =============================================================================
# 🔧 유틸리티 함수들
# =============================================================================

def create_error_response(
    message: str,
    session_id: str = "unknown",
    error_code: str = "UNKNOWN_ERROR",
    intent: IntentType = IntentType.ERROR
) -> ChatResponse:
    """표준화된 오류 응답 생성"""
    return ChatResponse(
        response=f"⚠️ {message}",
        intent=intent,
        session_id=session_id,
        status="error",
        error_code=error_code,
        error_message=message,
        timestamp=datetime.now().isoformat()
    )

def create_success_response(
    response_text: str,
    intent: IntentType,
    session_id: str,
    context: Optional[SessionContext] = None
) -> ChatResponse:
    """표준화된 성공 응답 생성"""
    return ChatResponse(
        response=response_text,
        intent=intent,
        session_id=session_id,
        context=context,
        status="success",
        timestamp=datetime.now().isoformat()
    )

# =============================================================================
# 🔧 의존성 주입 함수
# =============================================================================

def get_chat_service():
    """채팅 서비스 의존성 주입"""
    try:
        from app.llm.main_llm import get_chat_service
        return get_chat_service()
    except ImportError:
        # main_llm.py가 없는 경우 임시 더미 서비스 반환
        logger.warning("⚠️ main_llm.py를 찾을 수 없습니다. 더미 서비스를 사용합니다.")
        return DummyChatService()

class DummyChatService:
    """임시 더미 채팅 서비스 (개발용)"""
    
    async def process_chat_message(self, request: ChatRequest) -> ChatResponse:
        """더미 메시지 처리"""
        return create_success_response(
            response_text=f"🤖 더미 응답: '{request.message}' 메시지를 받았습니다. 실제 서비스는 아직 로딩 중입니다.",
            intent=IntentType.GENERAL,
            session_id=request.session_id or "dummy_session",
            context=SessionContext()
        )
    
    async def reset_session(self, session_id: str) -> bool:
        """더미 세션 리셋"""
        return True
    
    async def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """더미 세션 상태"""
        return {
            "session_id": session_id,
            "exists": False,
            "context": None,
            "created_at": None,
            "last_activity": None,
            "message_count": 0,
            "status": "dummy"
        }
    
    async def get_active_sessions(self) -> Dict[str, Any]:
        """더미 활성 세션"""
        return {"total_sessions": 0, "sessions": []}

# =============================================================================
# 📡 FastAPI 라우터 정의
# =============================================================================

# 라우터 생성
router = APIRouter()

@router.post("/chat", response_model=ChatResponse, summary="채팅 메시지 처리")
async def chat_message(
    request: ChatRequest,
    chat_service=Depends(get_chat_service)
) -> ChatResponse:
    """
    ## 🏥 통합 의료 챗봇 채팅 API
    
    사용자의 메시지를 처리하고 적절한 의료 정보를 제공합니다.
    
    ### 🔍 지원 기능:
    - **질병 진단**: 증상 기반 질병 예측 및 차별화 질문
    - **의약품 추천**: 질병 연계 의약품 정보 제공
    - **의료 정보 검색**: RAG 기반 의료 지식 검색
    - **세션 관리**: 대화 문맥 유지
    
    ### 📝 사용 예시:
    - "머리가 아프고 열이 나요" → 질병 진단 + 차별화 질문
    - "어떤 약을 먹어야 해요?" → 의약품 추천
    - "감기에 대해 알려줘" → 의료 정보 검색
    - "처음으로" → 세션 초기화
    
    ### ⚠️ 주의사항:
    이 서비스는 의료 전문가의 진단을 대체할 수 없습니다.
    """
    try:
        logger.info(f"📨 채팅 요청 수신: session_id={request.session_id}, message_length={len(request.message)}")
        
        # 메시지 길이 검증
        if len(request.message.strip()) == 0:
            raise HTTPException(
                status_code=400,
                detail="메시지가 비어있습니다. 증상이나 질문을 입력해주세요."
            )
        
        # 채팅 서비스를 통해 메시지 처리
        response_data = await chat_service.process_chat_message(request)
        
        logger.info(f"✅ 채팅 응답 생성 완료: session_id={response_data.session_id}, intent={response_data.intent}")
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 채팅 처리 중 오류 발생: {str(e)}", exc_info=True)
        
        # 오류 응답 생성
        return create_error_response(
            message="처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
            session_id=request.session_id or "error_session",
            error_code="INTERNAL_SERVER_ERROR"
        )

@router.delete("/chat/session/{session_id}/reset", response_model=SessionResetResponse, summary="세션 초기화")
async def reset_session(
    session_id: str,
    chat_service=Depends(get_chat_service)
) -> SessionResetResponse:
    """
    ## 🔄 세션 초기화
    
    지정된 세션의 모든 대화 기록과 컨텍스트를 초기화합니다.
    """
    try:
        logger.info(f"🔄 세션 초기화 요청: session_id={session_id}")
        
        success = await chat_service.reset_session(session_id)
        
        if success:
            logger.info(f"✅ 세션 초기화 완료: session_id={session_id}")
            return SessionResetResponse(
                message="세션이 성공적으로 초기화되었습니다.",
                session_id=session_id
            )
        else:
            logger.warning(f"⚠️ 세션 초기화 실패: session_id={session_id}")
            return SessionResetResponse(
                message="세션 초기화에 실패했습니다. 세션이 존재하지 않을 수 있습니다.",
                session_id=session_id,
                status="warning"
            )
            
    except Exception as e:
        logger.error(f"❌ 세션 초기화 중 오류: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"세션 초기화 중 오류가 발생했습니다: {str(e)}"
        )

@router.get("/chat/session/{session_id}/status", response_model=SessionStatusResponse, summary="세션 상태 조회")
async def get_session_status(
    session_id: str,
    chat_service=Depends(get_chat_service)
) -> SessionStatusResponse:
    """
    ## 📊 세션 상태 조회
    
    지정된 세션의 현재 상태와 컨텍스트 정보를 조회합니다.
    """
    try:
        logger.info(f"📊 세션 상태 조회: session_id={session_id}")
        
        session_info = await chat_service.get_session_status(session_id)
        
        logger.info(f"✅ 세션 상태 조회 완료: session_id={session_id}, exists={session_info['exists']}")
        
        return SessionStatusResponse(**session_info)
        
    except Exception as e:
        logger.error(f"❌ 세션 상태 조회 중 오류: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"세션 상태 조회 중 오류가 발생했습니다: {str(e)}"
        )

@router.get("/chat/sessions", summary="활성 세션 목록 조회")
async def get_active_sessions(
    chat_service=Depends(get_chat_service)
) -> Dict[str, Any]:
    """
    ## 📋 활성 세션 목록 조회
    
    현재 활성 상태인 모든 세션의 목록을 조회합니다.
    """
    try:
        logger.info("📋 활성 세션 목록 조회")
        
        sessions_info = await chat_service.get_active_sessions()
        
        logger.info(f"✅ 활성 세션 조회 완료: count={sessions_info['total_sessions']}")
        
        return sessions_info
        
    except Exception as e:
        logger.error(f"❌ 활성 세션 조회 중 오류: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"활성 세션 조회 중 오류가 발생했습니다: {str(e)}"
        )

# =============================================================================
# 📦 Export 정의
# =============================================================================

# 🔧 기존 서비스와의 호환성을 위한 추가 import
__all__ = [
    # 모델들
    'ChatRequest', 'ChatResponse', 'SessionContext', 'QuestioningState', 
    'UserProfile', 'IntentType', 'SimpleResponse', 'SessionResetResponse', 'SessionStatusResponse',
    
    # 유틸리티 함수들
    'create_error_response', 'create_success_response',
    
    # 라우터
    'router'
]