"""
통합 채팅 서비스 - CLI에서 FastAPI 서비스로 변환
위치: backend/app/llm/services/integrated_chat_service.py

🎯 목적: OptimizedIntegratedChatServiceV6 CLI 로직을 FastAPI 서비스로 변환
📋 기능: CLI의 모든 기능을 API 형태로 제공 (100% 로직 보존)
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import uuid
import traceback

# 내부 모듈 imports
from app.llm.services.session_manager import SessionManager, IntegratedSession
from app.llm.services.disease_service import EnhancedDiseaseService
from app.llm.services.medication_service import MedicationService
from app.llm.services.intent_classifier import EnhancedIntentClassifier
from app.llm.services.embedding_service import EmbeddingModel, RAGIndexManager
from app.llm.api import ChatRequest, ChatResponse, SessionContext, QuestioningState

# 로깅 설정
logger = logging.getLogger(__name__)

class IntegratedChatAPIService:
    """
    통합 채팅 API 서비스
    
    🔄 CLI → API 변환:
    - OptimizedIntegratedChatServiceV6의 모든 로직 보존
    - FastAPI 환경에 맞게 비동기 처리 적용
    - 세션 관리 메모리 기반으로 구현
    - JSON 응답 형태로 변환
    """
    
    def __init__(self):
        """서비스 초기화"""
        logger.info("🚀 통합 채팅 서비스 초기화 시작...")
        
        # 🔧 핵심 서비스 컴포넌트들
        self.session_manager = None
        self.disease_service = None
        self.medication_service = None
        self.intent_classifier = None
        self.embedding_model = None
        self.rag_manager = None
        
        # 🔄 서비스 상태
        self.is_initialized = False
        self.initialization_error = None
        
        # 📊 통계 정보
        self.stats = {
            "total_messages": 0,
            "successful_responses": 0,
            "error_responses": 0,
            "sessions_created": 0,
            "start_time": datetime.now()
        }
        
        # 🚀 서비스 초기화 실행
        self._initialize_services()
    
    def _initialize_services(self):
        """모든 서비스 컴포넌트 초기화 (CLI 로직 기반)"""
        try:
            logger.info("🔄 서비스 컴포넌트 초기화 중...")
            
            # 1️⃣ 임베딩 모델 초기화 (KM-BERT)
            logger.info("📚 임베딩 모델 로딩...")
            self.embedding_model = EmbeddingModel()
            
            # 2️⃣ RAG 매니저 초기화 (6개 clean_ 파일 + FAISS)
            logger.info("🔍 RAG 인덱스 매니저 초기화...")
            self.rag_manager = RAGIndexManager(self.embedding_model)
            self.rag_manager.load_rag_data()  # CLI와 동일한 로직
            
            # 3️⃣ 의도 분류기 초기화
            logger.info("🧠 의도 분류기 초기화...")
            self.intent_classifier = EnhancedIntentClassifier(self.embedding_model)
            
            # 4️⃣ 질병 서비스 초기화
            logger.info("🏥 질병 진단 서비스 초기화...")
            self.disease_service = EnhancedDiseaseService(
                self.embedding_model, 
                self.rag_manager
            )
            
            # 5️⃣ 의약품 서비스 초기화
            logger.info("💊 의약품 서비스 초기화...")
            self.medication_service = MedicationService(
                self.embedding_model,
                self.rag_manager
            )
            
            # 6️⃣ 세션 매니저 초기화 (메모리 기반)
            logger.info("📝 세션 매니저 초기화...")
            self.session_manager = SessionManager()
            
            # ✅ 초기화 완료
            self.is_initialized = True
            logger.info("✅ 모든 서비스 컴포넌트 초기화 완료!")
            
            # 📊 초기화 상태 로깅
            self._log_initialization_status()
            
        except Exception as e:
            self.initialization_error = str(e)
            logger.error(f"❌ 서비스 초기화 실패: {e}")
            logger.error(traceback.format_exc())
            raise e
    
    def _log_initialization_status(self):
        """초기화 상태 로깅"""
        status_info = {
            "embedding_model": "✅ KM-BERT 로딩 완료",
            "rag_manager": f"✅ RAG 인덱스 완료 (Q&A: {len(self.rag_manager.qa_documents)}, Medical: {len(self.rag_manager.medical_documents)})",
            "disease_service": "✅ 질병 서비스 준비",
            "medication_service": "✅ 의약품 서비스 준비",
            "intent_classifier": "✅ 의도 분류기 준비",
            "session_manager": "✅ 세션 매니저 준비"
        }
        
        for component, status in status_info.items():
            logger.info(f"   {status}")
    
    async def process_chat_message(self, request: ChatRequest) -> ChatResponse:
        """
        채팅 메시지 처리 (CLI의 process_message 메서드와 동일한 로직)
        
        Args:
            request: 채팅 요청 데이터
            
        Returns:
            ChatResponse: 처리된 응답 데이터
        """
        if not self.is_initialized:
            raise Exception(f"서비스가 초기화되지 않았습니다: {self.initialization_error}")
        
        # 📊 통계 업데이트
        self.stats["total_messages"] += 1
        
        try:
            # 1️⃣ 세션 관리 (CLI의 IntegratedSession과 동일)
            session = self._get_or_create_session(request.session_id)
            
            # 2️⃣ 사용자 프로필 적용
            if request.user_profile:
                session.set_user_profile(request.user_profile.dict())
            
            # 3️⃣ 메시지 처리 (CLI 로직 완전 동일)
            response_text = await self._process_message_logic(request.message, session)
            
            # 4️⃣ 의도 분류 (마지막 처리된 의도 가져오기)
            current_intent = session.context.get("last_intent", "general")
            
            # 5️⃣ 응답 데이터 구성
            response = ChatResponse(
                response=response_text,
                intent=current_intent,
                session_id=session.session_id,
                context=self._build_session_context(session),
                status="success"
            )
            
            # 📊 성공 통계 업데이트
            self.stats["successful_responses"] += 1
            
            logger.info(f"✅ 메시지 처리 완료: session={session.session_id}, intent={current_intent}")
            
            return response
            
        except Exception as e:
            # 📊 오류 통계 업데이트
            self.stats["error_responses"] += 1
            
            logger.error(f"❌ 메시지 처리 중 오류: {str(e)}")
            logger.error(traceback.format_exc())
            
            # 오류 응답 생성
            error_response = ChatResponse(
                response="⚠️ 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
                intent="error",
                session_id=request.session_id or "error_session",
                context=SessionContext(),
                status="error",
                error_code="PROCESSING_ERROR",
                error_message=str(e)
            )
            
            return error_response
    
    def _get_or_create_session(self, session_id: Optional[str]) -> IntegratedSession:
        """세션 가져오기 또는 생성"""
        if session_id:
            session = self.session_manager.get_session(session_id)
        else:
            session = self.session_manager.create_session()
            self.stats["sessions_created"] += 1
            
        return session
    
    async def _process_message_logic(self, message: str, session: IntegratedSession) -> str:
        """
        메시지 처리 로직 (CLI의 OptimizedIntegratedChatServiceV6.process_message와 동일)
        """
        # 📝 세션에 메시지 추가
        session.add_message("user", message)
        
        # 🔍 사용자 의도 분류
        intent = self.intent_classifier.classify_intent(message, session)
        session.context["last_intent"] = intent
        
        # 🔄 의도별 처리 분기 (CLI와 동일한 로직)
        if intent == "disease_diagnosis":
            response = await self._handle_disease_diagnosis(message, session)
        elif intent == "medication_recommend":
            response = await self._handle_medication_recommend(message, session)
        elif intent == "disease_info":
            response = await self._handle_disease_info(message, session)
        elif intent == "medication_info":
            response = await self._handle_medication_info(message, session)
        elif intent == "disease_to_medication":
            response = await self._handle_disease_to_medication(message, session)
        elif intent == "session_reset":
            response = self._handle_session_reset(session)
        else:
            response = self._handle_general_message(message)
        
        # 📝 봇 응답을 세션에 추가
        session.add_message("bot", response)
        
        return response
    
    async def _handle_disease_diagnosis(self, message: str, session: IntegratedSession) -> str:
        """질병 진단 처리 (CLI의 EnhancedDiseaseService와 동일)"""
        return await asyncio.to_thread(
            self.disease_service.process_disease_diagnosis, 
            message, 
            session
        )
    
    async def _handle_medication_recommend(self, message: str, session: IntegratedSession) -> str:
        """의약품 추천 처리"""
        return await asyncio.to_thread(
            self.medication_service.process_medication_query,
            message,
            session
        )
    
    async def _handle_disease_info(self, message: str, session: IntegratedSession) -> str:
        """질병 정보 검색 처리"""
        return await asyncio.to_thread(
            self.disease_service.search_disease_info,
            message,
            session
        )
    
    async def _handle_medication_info(self, message: str, session: IntegratedSession) -> str:
        """의약품 정보 검색 처리"""
        return await asyncio.to_thread(
            self.medication_service.search_medication_info,
            message,
            session
        )
    
    async def _handle_disease_to_medication(self, message: str, session: IntegratedSession) -> str:
        """질병-의약품 연계 처리 (새 기능)"""
        # 세션에서 최근 진단 정보 가져오기
        recent_diagnosis = session.get_recent_diagnosis()
        
        if recent_diagnosis:
            # 진단된 질병 기반으로 의약품 추천
            return await asyncio.to_thread(
                self.medication_service.recommend_by_disease,
                recent_diagnosis,
                session
            )
        else:
            return "먼저 증상을 말씀해주시면 질병을 진단한 후 적절한 의약품을 추천해드리겠습니다."
    
    def _handle_session_reset(self, session: IntegratedSession) -> str:
        """세션 초기화 처리"""
        session.reset()
        return "🔄 대화가 초기화되었습니다. 새로운 증상이나 질문을 말씀해주세요."
    
    def _handle_general_message(self, message: str) -> str:
        """일반 메시지 처리 (CLI와 동일)"""
        greetings = ["안녕", "hello", "hi", "안녕하세요"]
        thanks = ["감사", "고마워", "thank"]
        
        message_lower = message.lower()
        
        if any(greet in message_lower for greet in greetings):
            return """안녕하세요! 통합 의료 챗봇 API v6입니다. 

🔍 **이용 가능한 기능**:
• 증상 설명 → 질병 진단 (스마트한 차별화 질문)
• 질병 진단 후 "어떤 약?" → 의약품 추천  
• 질병 정보 검색 (RAG 기반)
• 의약품 정보 검색
• "처음으로" → 세션 초기화

🏥 **기술 스택**:
• EXAONE 3.5:7.8b + KM-BERT + FAISS
• FastAPI 기반 API 서비스

어떤 증상이 있으신가요?"""
        elif any(thank in message_lower for thank in thanks):
            return "도움이 되셨다니 기쁩니다! 다른 궁금한 점이 있으시면 언제든 말씀해주세요."
        else:
            return "죄송합니다. 잘 이해하지 못했습니다. 증상을 설명해주시거나 질병/의약품에 대해 구체적으로 문의해주세요."
    
    def _build_session_context(self, session: IntegratedSession) -> SessionContext:
        """세션 컨텍스트 구성 (API 응답용)"""
        questioning_state = QuestioningState(
            is_questioning=session.context.get("questioning_state", {}).get("is_questioning", False),
            current_question=session.context.get("questioning_state", {}).get("current_question"),
            question_count=session.context.get("questioning_state", {}).get("question_count", 0)
        )
        
        return SessionContext(
            last_disease=session.get_recent_diagnosis(),
            symptoms=session.context.get("symptoms", []),
            mentioned_symptoms=session.context.get("mentioned_symptoms", []),
            initial_symptoms_text=session.context.get("initial_symptoms_text"),
            questioning_state=questioning_state,
            medications=session.context.get("medications", []),
            diagnosis_time=session.context.get("diagnosis_time")
        )
    
    # =============================================================================
    # 🔧 세션 관리 API 메서드들
    # =============================================================================
    
    async def reset_session(self, session_id: str) -> bool:
        """세션 초기화"""
        return self.session_manager.reset_session(session_id)
    
    async def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """세션 상태 조회"""
        session_info = self.session_manager.get_session_info(session_id)
        
        if session_info["exists"]:
            session = self.session_manager.get_session(session_id)
            session_info["context"] = self._build_session_context(session)
        
        return session_info
    
    async def get_active_sessions(self) -> Dict[str, Any]:
        """활성 세션 목록 조회"""
        return self.session_manager.get_active_sessions_info()
    
    # =============================================================================
    # 🔧 서비스 상태 관리 메서드들
    # =============================================================================
    
    def get_service_status(self) -> Dict[str, Any]:
        """서비스 상태 정보 반환"""
        return {
            "initialized": self.is_initialized,
            "initialization_error": self.initialization_error,
            "embedding_model_ready": self.embedding_model is not None,
            "rag_manager_ready": self.rag_manager is not None,
            "disease_service_ready": self.disease_service is not None,
            "medication_service_ready": self.medication_service is not None,
            "session_manager_ready": self.session_manager is not None,
            "active_sessions": len(self.session_manager.sessions) if self.session_manager else 0,
            "stats": self.stats
        }
    
    def get_current_timestamp(self) -> str:
        """현재 타임스탬프 반환"""
        return datetime.now().isoformat()
    
    def cleanup(self):
        """서비스 정리"""
        logger.info("🔄 서비스 정리 중...")
        
        if self.session_manager:
            self.session_manager.cleanup()
        
        # 기타 정리 작업들...
        
        logger.info("✅ 서비스 정리 완료")