"""
세션 매니저 - CLI의 IntegratedSession을 FastAPI 환경에 맞게 확장
위치: backend/app/llm/services/session_manager.py

🎯 목적: CLI의 세션 로직을 메모리 기반 API 세션으로 확장
📋 기능: 세션 생성/관리/정리 + CLI 세션 로직 100% 보존
"""

import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import threading
import time
from dataclasses import dataclass, field

# 로깅 설정
logger = logging.getLogger(__name__)

@dataclass
class UserProfile:
    """사용자 프로필 정보"""
    age_group: str = "성인"
    is_pregnant: bool = False
    chronic_conditions: List[str] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)

class IntegratedSession:
    """
    통합 세션 클래스 (CLI 코드 완전 동일)
    
    🔄 CLI에서 사용하던 IntegratedSession과 100% 동일한 로직
    📝 대화 기록, 컨텍스트, 질병 진단 정보 등 모든 상태 관리
    """
    
    def __init__(self, session_id: Optional[str] = None):
        """세션 초기화"""
        self.session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        
        # 📝 대화 기록 (CLI와 동일)
        self.history: List[Dict[str, Any]] = []
        
        # 🧠 세션 컨텍스트 (CLI와 동일)
        self.context: Dict[str, Any] = {
            "symptoms": [],
            "mentioned_symptoms": [],
            "initial_symptoms_text": None,
            "last_disease": None,
            "diagnosis_time": None,
            "questioning_state": {
                "is_questioning": False,
                "current_question": None,
                "question_count": 0,
                "symptoms_mentioned": set()
            },
            "medications": [],
            "last_intent": "general",
            "user_profile": {}
        }
        
        # 📊 세션 통계
        self.message_count = 0
        
        logger.info(f"📝 새 세션 생성: {self.session_id}")
    
    def add_message(self, role: str, content: str):
        """메시지 추가 (CLI와 동일)"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        self.history.append(message)
        self.message_count += 1
        self.last_activity = datetime.now()
        
        logger.debug(f"💬 메시지 추가: {self.session_id} | {role}: {content[:50]}...")
    
    def get_recent_diagnosis(self) -> Optional[str]:
        """최근 진단 결과 반환 (CLI와 동일)"""
        diagnosis_time = self.context.get("diagnosis_time")
        
        if diagnosis_time and self.context.get("last_disease"):
            # 30분 이내 진단만 유효
            if isinstance(diagnosis_time, str):
                diagnosis_time = datetime.fromisoformat(diagnosis_time)
            
            if datetime.now() - diagnosis_time < timedelta(minutes=30):
                return self.context.get("last_disease")
        
        return None
    
    def set_diagnosis(self, disease: str):
        """질병 진단 설정 (CLI와 동일)"""
        self.context["last_disease"] = disease
        self.context["diagnosis_time"] = datetime.now().isoformat()
        
        logger.info(f"🏥 질병 진단 설정: {self.session_id} | {disease}")
    
    def add_symptom(self, symptom: str):
        """증상 추가 (CLI와 동일)"""
        if symptom not in self.context["symptoms"]:
            self.context["symptoms"].append(symptom)
        
        if symptom not in self.context["mentioned_symptoms"]:
            self.context["mentioned_symptoms"].append(symptom)
        
        # 차별화 질문용 증상 기록
        self.context["questioning_state"]["symptoms_mentioned"].add(symptom.lower())
        
        logger.debug(f"🔍 증상 추가: {self.session_id} | {symptom}")
    
    def set_initial_symptoms(self, symptoms_text: str):
        """초기 증상 텍스트 설정 (CLI와 동일)"""
        self.context["initial_symptoms_text"] = symptoms_text
        logger.debug(f"📝 초기 증상 설정: {self.session_id} | {symptoms_text[:50]}...")
    
    def start_questioning(self, question: str):
        """질문 모드 시작 (CLI와 동일)"""
        self.context["questioning_state"].update({
            "is_questioning": True,
            "current_question": question,
            "question_count": self.context["questioning_state"]["question_count"] + 1
        })
        
        logger.info(f"❓ 질문 모드 시작: {self.session_id} | Q{self.context['questioning_state']['question_count']}")
    
    def stop_questioning(self):
        """질문 모드 종료 (CLI와 동일)"""
        self.context["questioning_state"].update({
            "is_questioning": False,
            "current_question": None
        })
        
        logger.info(f"✅ 질문 모드 종료: {self.session_id}")
    
    def add_medication(self, medication_info: Dict[str, Any]):
        """의약품 정보 추가 (CLI와 동일)"""
        if "medications" not in self.context:
            self.context["medications"] = []
        
        self.context["medications"].append(medication_info)
        
        logger.debug(f"💊 의약품 추가: {self.session_id} | {medication_info.get('name', 'Unknown')}")
    
    def set_user_profile(self, profile_data: Dict[str, Any]):
        """사용자 프로필 설정"""
        self.context["user_profile"] = profile_data
        
        logger.debug(f"👤 사용자 프로필 설정: {self.session_id}")
    
    def get_conversation_summary(self) -> str:
        """대화 요약 생성 (CLI와 동일)"""
        if not self.history:
            return "대화 기록 없음"
        
        recent_messages = self.history[-5:]  # 최근 5개 메시지
        summary_parts = []
        
        for msg in recent_messages:
            role = "사용자" if msg["role"] == "user" else "챗봇"
            content = msg["content"][:100] + ("..." if len(msg["content"]) > 100 else "")
            summary_parts.append(f"{role}: {content}")
        
        return "\n".join(summary_parts)
    
    def reset(self):
        """세션 초기화 (CLI와 동일)"""
        logger.info(f"🔄 세션 초기화: {self.session_id}")
        
        # 기록은 유지, 컨텍스트만 초기화
        self.context = {
            "symptoms": [],
            "mentioned_symptoms": [],
            "initial_symptoms_text": None,
            "last_disease": None,
            "diagnosis_time": None,
            "questioning_state": {
                "is_questioning": False,
                "current_question": None,
                "question_count": 0,
                "symptoms_mentioned": set()
            },
            "medications": [],
            "last_intent": "general",
            "user_profile": self.context.get("user_profile", {})  # 사용자 프로필은 유지
        }
        
        self.last_activity = datetime.now()
    
    def is_expired(self, ttl_minutes: int = 30) -> bool:
        """세션 만료 확인"""
        return datetime.now() - self.last_activity > timedelta(minutes=ttl_minutes)
    
    def to_dict(self) -> Dict[str, Any]:
        """세션 정보를 딕셔너리로 변환"""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "message_count": self.message_count,
            "context": self.context,
            "recent_diagnosis": self.get_recent_diagnosis(),
            "is_questioning": self.context["questioning_state"]["is_questioning"]
        }

class SessionManager:
    """
    세션 매니저 - 메모리 기반 세션 관리
    
    🧠 기능:
    - 세션 생성/조회/삭제
    - 자동 만료 세션 정리
    - 메모리 사용량 모니터링
    - 통계 정보 제공
    """
    
    def __init__(self, session_ttl_minutes: int = 30, cleanup_interval_minutes: int = 5):
        """세션 매니저 초기화"""
        self.sessions: Dict[str, IntegratedSession] = {}
        self.session_ttl_minutes = session_ttl_minutes
        self.cleanup_interval_minutes = cleanup_interval_minutes
        
        # 🔄 자동 정리 스레드
        self._cleanup_thread = None
        self._stop_cleanup = False
        
        # 📊 통계 정보
        self.stats = {
            "total_sessions_created": 0,
            "total_sessions_expired": 0,
            "total_messages_processed": 0,
            "start_time": datetime.now()
        }
        
        self._lock = threading.Lock()
        
        # 자동 정리 시작
        self._start_cleanup_thread()
        
        logger.info(f"📝 세션 매니저 초기화 완료 (TTL: {session_ttl_minutes}분, 정리 간격: {cleanup_interval_minutes}분)")
    
    def create_session(self, session_id: Optional[str] = None) -> IntegratedSession:
        """새 세션 생성"""
        with self._lock:
            session = IntegratedSession(session_id)
            self.sessions[session.session_id] = session
            self.stats["total_sessions_created"] += 1
            
            logger.info(f"📝 세션 생성: {session.session_id} (총 {len(self.sessions)}개 활성)")
            
            return session
    
    def get_session(self, session_id: str) -> IntegratedSession:
        """세션 조회 (없으면 새로 생성)"""
        with self._lock:
            if session_id in self.sessions:
                session = self.sessions[session_id]
                # 활동 시간 업데이트
                session.last_activity = datetime.now()
                return session
            else:
                # 새 세션 생성
                return self.create_session(session_id)
    
    def session_exists(self, session_id: str) -> bool:
        """세션 존재 확인"""
        with self._lock:
            return session_id in self.sessions
    
    def reset_session(self, session_id: str) -> bool:
        """세션 초기화"""
        with self._lock:
            if session_id in self.sessions:
                self.sessions[session_id].reset()
                logger.info(f"🔄 세션 초기화: {session_id}")
                return True
            return False
    
    def delete_session(self, session_id: str) -> bool:
        """세션 삭제"""
        with self._lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                logger.info(f"🗑️ 세션 삭제: {session_id}")
                return True
            return False
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """세션 정보 조회"""
        with self._lock:
            if session_id in self.sessions:
                session = self.sessions[session_id]
                return {
                    "session_id": session_id,
                    "exists": True,
                    "created_at": session.created_at.isoformat(),
                    "last_activity": session.last_activity.isoformat(),
                    "message_count": session.message_count,
                    "is_expired": session.is_expired(self.session_ttl_minutes)
                }
            else:
                return {
                    "session_id": session_id,
                    "exists": False,
                    "created_at": None,
                    "last_activity": None,
                    "message_count": 0,
                    "is_expired": True
                }
    
    def get_active_sessions_info(self) -> Dict[str, Any]:
        """활성 세션 목록 정보"""
        with self._lock:
            active_sessions = []
            total_messages = 0
            
            for session_id, session in self.sessions.items():
                if not session.is_expired(self.session_ttl_minutes):
                    session_info = {
                        "session_id": session_id,
                        "created_at": session.created_at.isoformat(),
                        "last_activity": session.last_activity.isoformat(),
                        "message_count": session.message_count,
                        "has_diagnosis": session.get_recent_diagnosis() is not None,
                        "is_questioning": session.context["questioning_state"]["is_questioning"]
                    }
                    active_sessions.append(session_info)
                    total_messages += session.message_count
            
            return {
                "total_sessions": len(active_sessions),
                "total_messages": total_messages,
                "sessions": active_sessions,
                "stats": self.stats,
                "memory_usage": {
                    "stored_sessions": len(self.sessions),
                    "active_sessions": len(active_sessions)
                }
            }
    
    def cleanup_expired_sessions(self) -> int:
        """만료된 세션 정리"""
        with self._lock:
            expired_sessions = []
            
            for session_id, session in self.sessions.items():
                if session.is_expired(self.session_ttl_minutes):
                    expired_sessions.append(session_id)
            
            # 만료된 세션 삭제
            for session_id in expired_sessions:
                del self.sessions[session_id]
                self.stats["total_sessions_expired"] += 1
            
            if expired_sessions:
                logger.info(f"🗑️ 만료된 세션 {len(expired_sessions)}개 정리 완료")
            
            return len(expired_sessions)
    
    def _start_cleanup_thread(self):
        """자동 정리 스레드 시작"""
        def cleanup_worker():
            while not self._stop_cleanup:
                try:
                    self.cleanup_expired_sessions()
                    time.sleep(self.cleanup_interval_minutes * 60)  # 분 → 초 변환
                except Exception as e:
                    logger.error(f"❌ 세션 정리 중 오류: {e}")
                    time.sleep(60)  # 1분 대기 후 재시도
        
        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()
        
        logger.info("🔄 자동 세션 정리 스레드 시작")
    
    def cleanup(self):
        """매니저 정리"""
        self._stop_cleanup = True
        
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)
        
        # 모든 세션 정리
        with self._lock:
            self.sessions.clear()
        
        logger.info("🔄 세션 매니저 정리 완료")
    
    def get_stats(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        with self._lock:
            current_time = datetime.now()
            uptime = current_time - self.stats["start_time"]
            
            return {
                **self.stats,
                "current_active_sessions": len(self.sessions),
                "uptime_minutes": int(uptime.total_seconds() / 60),
                "session_ttl_minutes": self.session_ttl_minutes,
                "cleanup_interval_minutes": self.cleanup_interval_minutes
            }