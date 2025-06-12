"""
의도 분류기 - CLI의 EnhancedIntentClassifier 완전 동일 로직
위치: backend/app/llm/services/intent_classifier.py

🎯 목적: 사용자 메시지의 의도를 정확하게 분류
📋 기능: 질병진단, 의약품추천, 정보검색, 질병-의약품 연계 등
"""

import re
import logging
from typing import Dict, List, Set, Optional
from dataclasses import dataclass

# 내부 모듈 imports
from app.llm.services.session_manager import IntegratedSession

# 로깅 설정
logger = logging.getLogger(__name__)

@dataclass
class IntentResult:
    """의도 분류 결과"""
    intent: str
    confidence: float
    reasoning: str
    keywords_matched: List[str]

class EnhancedIntentClassifier:
    """
    향상된 의도 분류기 (CLI 코드 완전 동일)
    
    🧠 분류 가능한 의도들:
    - disease_diagnosis: 질병 진단 (증상 → 질병)
    - medication_recommend: 의약품 추천
    - disease_info: 질병 정보 검색
    - medication_info: 의약품 정보 검색
    - disease_to_medication: 질병-의약품 연계 (새 기능)
    - session_reset: 세션 초기화
    - general: 일반 대화
    """
    
    def __init__(self, embedding_model):
        """의도 분류기 초기화"""
        self.embedding_model = embedding_model
        
        # 🎯 의도별 키워드 패턴 (CLI와 동일)
        self.intent_patterns = {
            "disease_diagnosis": {
                "symptom_keywords": [
                    "아프", "아픈", "아파", "아픔", "통증", "쑤시", "따끔", "쓰림",
                    "열", "발열", "체온", "뜨거움", "오한", "춥",
                    "기침", "가래", "콧물", "코막힘", "재채기",
                    "두통", "머리", "어지럼", "현기증",
                    "복통", "배", "설사", "변비", "구토", "메스꺼움",
                    "피로", "힘들", "지침", "무기력",
                    "가려움", "발진", "부종", "붓",
                    "숨", "호흡", "숨쉬기", "답답",
                    "목", "인후", "목구멍", "삼키기"
                ],
                "patterns": [
                    r"([가-힣]+)\s*(?:아|픈|아픈|아파|통증|쑤시|따끔)",
                    r"(열|발열|체온|오한|춥).*(?:나|남|있|해|올라)",
                    r"(기침|가래|콧물|재채기).*(?:나|남|있|해)",
                    r"(두통|머리.*아|어지럼|현기증)",
                    r"(복통|배.*아|설사|변비|구토|메스꺼움)",
                    r"(피로|힘들|지침|무기력)",
                    r"(가려|발진|부종|붓)",
                    r"(숨.*차|호흡.*어려|답답)"
                ]
            },
            
            "medication_recommend": {
                "keywords": [
                    "약", "약품", "의약품", "처방", "복용", "먹어야", "드셔야",
                    "치료제", "처방전", "약국", "먹으면", "복용법",
                    "어떤약", "무슨약", "약추천", "약종류"
                ],
                "patterns": [
                    r"(어떤|무슨|뭔|어느)\s*약",
                    r"약.*(?:먹어야|드셔야|복용|치료)",
                    r"(?:먹으면|복용하면).*좋",
                    r"약.*(?:추천|종류|있)"
                ]
            },
            
            "disease_info": {
                "keywords": [
                    "질병", "병", "질환", "증상", "원인", "치료법", "예방",
                    "대해", "이란", "무엇", "설명", "정보", "알려줘", "궁금"
                ],
                "disease_patterns": [
                    r"([가-힣]{2,}(?:병|증|염|암))\s*(?:대해|이란|무엇|설명|정보|알려|궁금)",
                    r"([가-힣]{2,})\s*(?:질병|질환)\s*(?:대해|이란|무엇|설명|정보|알려|궁금)",
                    r"([가-힣]{2,})\s*(?:원인|치료법|예방법|증상)"
                ]
            },
            
            "medication_info": {
                "keywords": [
                    "성분", "부작용", "효능", "효과", "용법", "용량", "주의사항",
                    "복용법", "먹는법", "사용법", "금기", "상호작용"
                ],
                "medication_patterns": [
                    r"([가-힣A-Za-z0-9]+(?:정|캡슐|시럽|연고))\s*(?:부작용|효능|용법|성분)",
                    r"(타이레놀|게보린|낙센|애드빌|부루펜|이부프로펜|아스피린)\s*(?:부작용|효능|용법|성분)",
                    r"([가-힣A-Za-z0-9]+)\s*(?:약|의약품)\s*(?:부작용|효능|용법|성분)"
                ]
            },
            
            "session_reset": {
                "keywords": ["처음", "초기화", "리셋", "새로", "다시"],
                "patterns": [
                    r"처음으?로",
                    r"초기화",
                    r"리셋",
                    r"새로.*시작",
                    r"다시.*시작"
                ]
            }
        }
        
        logger.info("🧠 의도 분류기 초기화 완료")
    
    def classify_intent(self, message: str, session: IntegratedSession) -> str:
        """
        메시지 의도 분류 (CLI와 완전 동일한 로직)
        
        Args:
            message: 사용자 메시지
            session: 현재 세션
            
        Returns:
            str: 분류된 의도
        """
        message = message.strip()
        message_lower = message.lower()
        
        logger.debug(f"🧠 의도 분류 시작: {message[:50]}...")
        
        # 1️⃣ 세션 초기화 의도 확인 (최우선)
        if self._check_session_reset_intent(message_lower):
            logger.info("🔄 세션 초기화 의도 감지")
            return "session_reset"
        
        # 2️⃣ 질병-의약품 연계 의도 확인 (세션 문맥 기반)
        disease_to_med_intent = self._check_disease_to_medication_intent(message_lower, session)
        if disease_to_med_intent:
            logger.info("💊➡️🏥 질병-의약품 연계 의도 감지")
            return "disease_to_medication"
        
        # 3️⃣ 질문 모드일 때의 의도 처리
        if session.context["questioning_state"]["is_questioning"]:
            # 질문에 대한 답변은 질병 진단으로 처리
            logger.info("❓ 질문 모드 중 - 질병 진단으로 처리")
            return "disease_diagnosis"
        
        # 4️⃣ 의약품 정보 의도 확인
        if self._check_medication_info_intent(message_lower):
            logger.info("💊ℹ️ 의약품 정보 의도 감지")
            return "medication_info"
        
        # 5️⃣ 의약품 추천 의도 확인
        if self._check_medication_recommend_intent(message_lower):
            logger.info("💊 의약품 추천 의도 감지")
            return "medication_recommend"
        
        # 6️⃣ 질병 정보 의도 확인
        if self._check_disease_info_intent(message_lower):
            logger.info("🏥ℹ️ 질병 정보 의도 감지")
            return "disease_info"
        
        # 7️⃣ 질병 진단 의도 확인 (증상 기반)
        if self._check_disease_diagnosis_intent(message_lower):
            logger.info("🏥 질병 진단 의도 감지")
            return "disease_diagnosis"
        
        # 8️⃣ 기본값: 일반 대화
        logger.info("💬 일반 대화 의도")
        return "general"
    
    def _check_session_reset_intent(self, message: str) -> bool:
        """세션 초기화 의도 확인"""
        patterns = self.intent_patterns["session_reset"]
        
        # 키워드 확인
        for keyword in patterns["keywords"]:
            if keyword in message:
                return True
        
        # 패턴 확인
        for pattern in patterns["patterns"]:
            if re.search(pattern, message):
                return True
        
        return False
    
    def _check_disease_to_medication_intent(self, message: str, session: IntegratedSession) -> bool:
        """질병-의약품 연계 의도 확인 (새 기능)"""
        # 세션에 최근 진단 정보가 있어야 함
        recent_diagnosis = session.get_recent_diagnosis()
        if not recent_diagnosis:
            return False
        
        # 의약품 관련 키워드 확인
        medication_keywords = [
            "약", "약품", "의약품", "처방", "복용", "먹어야", "드셔야",
            "치료제", "처방전", "어떤약", "무슨약", "뭘먹어야"
        ]
        
        for keyword in medication_keywords:
            if keyword in message:
                logger.debug(f"💊➡️🏥 질병-의약품 연계: {recent_diagnosis} + {keyword}")
                return True
        
        return False
    
    def _check_disease_diagnosis_intent(self, message: str) -> bool:
        """질병 진단 의도 확인 (증상 기반)"""
        patterns = self.intent_patterns["disease_diagnosis"]
        
        # 증상 키워드 확인
        for keyword in patterns["symptom_keywords"]:
            if keyword in message:
                return True
        
        # 증상 패턴 확인
        for pattern in patterns["patterns"]:
            if re.search(pattern, message):
                return True
        
        return False
    
    def _check_medication_recommend_intent(self, message: str) -> bool:
        """의약품 추천 의도 확인"""
        patterns = self.intent_patterns["medication_recommend"]
        
        # 키워드 확인
        for keyword in patterns["keywords"]:
            if keyword in message:
                return True
        
        # 패턴 확인
        for pattern in patterns["patterns"]:
            if re.search(pattern, message):
                return True
        
        return False
    
    def _check_disease_info_intent(self, message: str) -> bool:
        """질병 정보 의도 확인"""
        patterns = self.intent_patterns["disease_info"]
        
        # 기본 키워드 확인
        for keyword in patterns["keywords"]:
            if keyword in message:
                # 질병명 패턴과 함께 나타나는지 확인
                for pattern in patterns["disease_patterns"]:
                    if re.search(pattern, message):
                        return True
        
        # 질병명 패턴만으로도 확인
        for pattern in patterns["disease_patterns"]:
            if re.search(pattern, message):
                return True
        
        return False
    
    def _check_medication_info_intent(self, message: str) -> bool:
        """의약품 정보 의도 확인"""
        patterns = self.intent_patterns["medication_info"]
        
        # 키워드 확인
        for keyword in patterns["keywords"]:
            if keyword in message:
                return True
        
        # 의약품 패턴 확인
        for pattern in patterns["medication_patterns"]:
            if re.search(pattern, message):
                return True
        
        return False
    
    def get_detailed_classification(self, message: str, session: IntegratedSession) -> IntentResult:
        """상세한 의도 분류 결과 반환"""
        intent = self.classify_intent(message, session)
        
        # 매칭된 키워드들 찾기
        matched_keywords = self._find_matched_keywords(message.lower(), intent)
        
        # 신뢰도 계산 (단순 버전)
        confidence = self._calculate_confidence(message.lower(), intent, matched_keywords)
        
        # 분류 근거 생성
        reasoning = self._generate_reasoning(intent, matched_keywords, session)
        
        return IntentResult(
            intent=intent,
            confidence=confidence,
            reasoning=reasoning,
            keywords_matched=matched_keywords
        )
    
    def _find_matched_keywords(self, message: str, intent: str) -> List[str]:
        """매칭된 키워드들 찾기"""
        matched = []
        
        if intent in self.intent_patterns:
            patterns = self.intent_patterns[intent]
            
            # 키워드 매칭
            if "keywords" in patterns:
                for keyword in patterns["keywords"]:
                    if keyword in message:
                        matched.append(keyword)
            
            # 증상 키워드 매칭
            if "symptom_keywords" in patterns:
                for keyword in patterns["symptom_keywords"]:
                    if keyword in message:
                        matched.append(keyword)
        
        return matched
    
    def _calculate_confidence(self, message: str, intent: str, matched_keywords: List[str]) -> float:
        """신뢰도 계산 (0.0 ~ 1.0)"""
        if not matched_keywords:
            return 0.3  # 기본 낮은 신뢰도
        
        # 키워드 수에 따른 기본 신뢰도
        base_confidence = min(0.9, 0.5 + len(matched_keywords) * 0.1)
        
        # 의도별 보정
        if intent == "disease_diagnosis" and any(symptom in message for symptom in ["아프", "아픈", "아파", "통증"]):
            base_confidence += 0.1
        elif intent == "medication_recommend" and "약" in message:
            base_confidence += 0.1
        
        return min(1.0, base_confidence)
    
    def _generate_reasoning(self, intent: str, matched_keywords: List[str], session: IntegratedSession) -> str:
        """분류 근거 생성"""
        if intent == "disease_diagnosis":
            return f"증상 관련 키워드 감지: {', '.join(matched_keywords)}"
        elif intent == "medication_recommend":
            return f"의약품 추천 키워드 감지: {', '.join(matched_keywords)}"
        elif intent == "disease_info":
            return f"질병 정보 요청 키워드 감지: {', '.join(matched_keywords)}"
        elif intent == "medication_info":
            return f"의약품 정보 요청 키워드 감지: {', '.join(matched_keywords)}"
        elif intent == "disease_to_medication":
            recent_diagnosis = session.get_recent_diagnosis()
            return f"질병-의약품 연계: 최근 진단({recent_diagnosis}) + 의약품 키워드({', '.join(matched_keywords)})"
        elif intent == "session_reset":
            return f"세션 초기화 키워드 감지: {', '.join(matched_keywords)}"
        else:
            return "일반 대화로 분류"
    
    def get_supported_intents(self) -> Dict[str, str]:
        """지원되는 의도 목록 반환"""
        return {
            "disease_diagnosis": "질병 진단 (증상 → 질병)",
            "medication_recommend": "의약품 추천",
            "disease_info": "질병 정보 검색",
            "medication_info": "의약품 정보 검색",
            "disease_to_medication": "질병-의약품 연계",
            "session_reset": "세션 초기화",
            "general": "일반 대화"
        }