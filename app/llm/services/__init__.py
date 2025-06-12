# =============================================================================
# 3. backend/app/llm/services/__init__.py
# =============================================================================
"""
LLM 서비스 모듈

🔧 서비스들:
- integrated_chat_service.py: 통합 채팅 서비스 (메인)
- session_manager.py: 세션 관리
- intent_classifier.py: 의도 분류
- disease_service.py: 질병 진단 서비스
- medication_service.py: 의약품 서비스
- embedding_service.py: 임베딩 및 RAG 서비스
"""

# 주요 서비스 import
from .integrated_chat_service import IntegratedChatAPIService
from .session_manager import SessionManager, IntegratedSession
from .intent_classifier import EnhancedIntentClassifier
from .disease_service import EnhancedDiseaseService
from .medication_service import MedicationService
from .embedding_service import EmbeddingModel, RAGIndexManager

__all__ = [
    "IntegratedChatAPIService",
    "SessionManager", 
    "IntegratedSession",
    "EnhancedIntentClassifier",
    "EnhancedDiseaseService",
    "MedicationService",
    "EmbeddingModel",
    "RAGIndexManager"
]