# backend/app/llm/api/__init__.py
"""
LLM API 엔드포인트 모듈

📡 구조:
- chat_models.py: Pydantic 모델 정의 (ChatRequest, ChatResponse 등)
- chat_api.py: FastAPI 라우터 정의 (실제 엔드포인트)

🔧 수정사항:
- 모델은 chat_models.py에서, 라우터는 chat_api.py에서 import
- integrated_chat_service.py에서 모델들에 접근 가능하도록 구성
"""

# 📋 모델들 import (chat_models.py에서)
from .chat_models import (
    ChatRequest,
    ChatResponse, 
    SessionContext,
    QuestioningState,
    UserProfile,
    IntentType,
    SimpleResponse,
    create_error_response,
    create_success_response
)

# 📡 라우터 import (chat_models.py에서)
from .chat_models import router as chat_router

# 📦 외부에서 사용 가능한 모든 요소들
__all__ = [
    # 라우터
    "chat_router",
    
    # 모델 클래스들
    "ChatRequest",
    "ChatResponse",
    "SessionContext", 
    "QuestioningState",
    "UserProfile",
    "IntentType",
    "SimpleResponse",
    
    # 유틸리티 함수들
    "create_error_response",
    "create_success_response"
]