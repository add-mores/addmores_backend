"""
질병 입력 검증 모듈
위치: ~/backend/app/utils/disease_validators.py

🎯 목적: 질병 API 입력값 검증 로직
📋 기능: 메시지 길이, 문자, 의료 관련성 검증
⚙️ 사용: DiseaseValidator 클래스로 통합 검증
"""

import re
import logging
from typing import List, Tuple

from .disease_constants import MIN_MESSAGE_LENGTH, MAX_MESSAGE_LENGTH, INVALID_PATTERNS
from .disease_exceptions import (
    DiseaseValidationError, EmptyMessageError, MessageTooShortError, 
    MessageTooLongError, InvalidCharacterError
)

logger = logging.getLogger(__name__)

# =============================================================================
# 의료 관련 키워드 정의
# =============================================================================

MEDICAL_KEYWORDS = [
    # 증상 관련
    "아프", "아픈", "아파", "통증", "아픔", "저리", "쑤시", "따끔", "콕콕",
    "열", "발열", "기침", "콧물", "목아픔", "두통", "복통", "설사", "구토",
    "어지러움", "현기증", "숨차", "가슴답답", "메스꺼움", "피로", "무기력",
    
    # 신체 부위
    "머리", "목", "어깨", "팔", "손", "가슴", "배", "등", "허리", "다리", "발",
    "눈", "귀", "코", "입", "목구멍", "심장", "폐", "위", "장", "간", "신장",
    
    # 상태 표현
    "아픈데", "괜찮", "이상", "문제", "증상", "병", "질병", "감기", "독감",
    "염증", "부종", "멍", "상처", "화상", "타박상", "골절",
    
    # 감정/상태
    "힘들", "고통", "괴로", "불편", "답답", "막막", "걱정", "불안",
    
    # 시간 표현
    "어제", "오늘", "며칠", "일주일", "한달", "계속", "자주", "가끔",
    
    # 정도 표현
    "조금", "많이", "심하게", "가볍게", "심각", "위험", "응급"
]

# =============================================================================
# 기본 검증 함수들
# =============================================================================

def validate_message_basic(message: str) -> bool:
    """기본 메시지 검증"""
    if not message:
        raise EmptyMessageError()
    
    if not isinstance(message, str):
        raise DiseaseValidationError("메시지는 문자열이어야 합니다.")
    
    message = message.strip()
    if not message:
        raise EmptyMessageError()
    
    return True


def validate_message_length(message: str) -> bool:
    """메시지 길이 검증"""
    message = message.strip()
    length = len(message)
    
    if length < MIN_MESSAGE_LENGTH:
        raise MessageTooShortError(length, MIN_MESSAGE_LENGTH)
    
    if length > MAX_MESSAGE_LENGTH:
        raise MessageTooLongError(length, MAX_MESSAGE_LENGTH)
    
    return True


def validate_message_characters(message: str) -> bool:
    """허용되지 않는 문자 검증"""
    for pattern in INVALID_PATTERNS:
        if re.match(pattern, message):
            raise InvalidCharacterError()
    
    # 연속된 특수문자 체크
    if re.search(r'[^\w\s가-힣]{3,}', message):
        raise InvalidCharacterError("연속된 특수문자가 너무 많습니다.")
    
    # 숫자만 있는 경우 체크
    if re.match(r'^\d+$', message.strip()):
        raise InvalidCharacterError("숫자만으로는 증상을 설명할 수 없습니다.")
    
    return True


def validate_medical_relevance(message: str) -> bool:
    """의료 관련성 검증"""
    message_lower = message.lower()
    
    # 의료 관련 키워드 포함 여부 확인
    medical_count = 0
    for keyword in MEDICAL_KEYWORDS:
        if keyword in message_lower:
            medical_count += 1
    
    # 최소 1개 이상의 의료 관련 키워드 필요
    if medical_count == 0:
        # 추가적인 의료 관련 패턴 체크
        medical_patterns = [
            r'(아|아프|아픈|아파)',  # 아픔 표현
            r'(통증|아픔|저리|쑤시)',  # 통증 표현
            r'(머리|목|배|가슴|등|허리|다리|팔)',  # 신체 부위
            r'(열|기침|콧물|설사|구토)',  # 증상
            r'(어지러|현기증|숨차|답답)',  # 상태
            r'(병|질병|감기|독감)',  # 질병명
        ]
        
        pattern_found = False
        for pattern in medical_patterns:
            if re.search(pattern, message_lower):
                pattern_found = True
                break
        
        if not pattern_found:
            raise DiseaseValidationError(
                "의료 관련 증상이나 키워드를 포함해서 설명해주세요. "
                "(예: '머리가 아파요', '기침이 나요', '배가 아픈데요')"
            )
    
    return True


def check_spam_patterns(message: str) -> bool:
    """스팸성 메시지 패턴 검증"""
    spam_patterns = [
        r'^(.)\1{5,}$',  # 같은 문자 반복
        r'^[ㅋㅎㅠㅜㅇ]+$',  # 감탄사만
        r'^\s*[.!?]+\s*$',  # 문장부호만
        r'^(하하|헤헤|히히|호호){3,}',  # 웃음 반복
    ]
    
    for pattern in spam_patterns:
        if re.search(pattern, message):
            raise DiseaseValidationError("의미 있는 증상을 설명해주세요.")
    
    return True


# =============================================================================
# DiseaseValidator 클래스 - 모든 검증 로직을 통합
# =============================================================================

class DiseaseValidator:
    """질병 입력 검증 클래스"""
    
    def __init__(self):
        self.min_length = MIN_MESSAGE_LENGTH
        self.max_length = MAX_MESSAGE_LENGTH
        self.invalid_patterns = INVALID_PATTERNS
        self.medical_keywords = MEDICAL_KEYWORDS
        
        logger.debug("DiseaseValidator 초기화 완료")
    
    def validate_message(self, message: str) -> bool:
        """메시지 전체 검증 (예외 발생)"""
        logger.debug(f"메시지 검증 시작: '{message}'")
        
        # 1. 기본 검증
        validate_message_basic(message)
        
        # 2. 길이 검증
        validate_message_length(message)
        
        # 3. 문자 검증
        validate_message_characters(message)
        
        # 4. 스팸 패턴 검증
        check_spam_patterns(message)
        
        # 5. 의료 관련성 검증
        validate_medical_relevance(message)
        
        logger.debug(f"메시지 검증 통과: '{message}'")
        return True
    
    def is_valid_message(self, message: str) -> Tuple[bool, str]:
        """메시지 검증 (예외 없이 결과 반환)"""
        try:
            self.validate_message(message)
            return True, "유효한 메시지입니다."
        except DiseaseValidationError as e:
            logger.warning(f"검증 실패: {e}")
            return False, str(e)
        except Exception as e:
            logger.error(f"검증 중 예상치 못한 오류: {e}")
            return False, f"검증 중 오류 발생: {str(e)}"
    
    def get_medical_relevance_score(self, message: str) -> float:
        """의료 관련성 점수 반환 (0.0 ~ 1.0)"""
        if not message:
            return 0.0
        
        message_lower = message.lower()
        medical_count = 0
        
        for keyword in self.medical_keywords:
            if keyword in message_lower:
                medical_count += 1
        
        # 전체 단어 수 대비 의료 키워드 비율
        words = message.split()
        if not words:
            return 0.0
        
        relevance_score = min(medical_count / len(words), 1.0)
        
        # 최소값 보정 (의료 키워드가 하나라도 있으면 최소 0.3)
        if medical_count > 0:
            relevance_score = max(relevance_score, 0.3)
        
        return relevance_score
    
    def get_validation_summary(self, message: str) -> dict:
        """검증 결과 상세 정보 반환"""
        summary = {
            "message": message,
            "is_valid": False,
            "error": None,
            "length": len(message) if message else 0,
            "medical_relevance_score": 0.0,
            "medical_keywords_found": [],
            "checks": {
                "basic": False,
                "length": False,
                "characters": False,
                "spam": False,
                "medical_relevance": False
            }
        }
        
        if not message:
            summary["error"] = "빈 메시지"
            return summary
        
        try:
            # 단계별 검증
            validate_message_basic(message)
            summary["checks"]["basic"] = True
            
            validate_message_length(message)
            summary["checks"]["length"] = True
            
            validate_message_characters(message)
            summary["checks"]["characters"] = True
            
            check_spam_patterns(message)
            summary["checks"]["spam"] = True
            
            validate_medical_relevance(message)
            summary["checks"]["medical_relevance"] = True
            
            # 추가 정보
            summary["medical_relevance_score"] = self.get_medical_relevance_score(message)
            
            # 발견된 의료 키워드
            message_lower = message.lower()
            found_keywords = [kw for kw in self.medical_keywords if kw in message_lower]
            summary["medical_keywords_found"] = found_keywords[:5]  # 최대 5개
            
            summary["is_valid"] = True
            
        except DiseaseValidationError as e:
            summary["error"] = str(e)
        except Exception as e:
            summary["error"] = f"검증 중 오류: {str(e)}"
        
        return summary


# =============================================================================
# 편의 함수들
# =============================================================================

def quick_validate(message: str) -> bool:
    """빠른 검증 (예외 발생)"""
    validator = DiseaseValidator()
    return validator.validate_message(message)


def safe_validate(message: str) -> Tuple[bool, str]:
    """안전한 검증 (예외 없음)"""
    validator = DiseaseValidator()
    return validator.is_valid_message(message)


# =============================================================================
# Export할 함수와 클래스들
# =============================================================================

__all__ = [
    # 메인 클래스
    "DiseaseValidator",
    
    # 개별 검증 함수들
    "validate_message_basic",
    "validate_message_length", 
    "validate_message_characters",
    "validate_medical_relevance",
    "check_spam_patterns",
    
    # 편의 함수들
    "quick_validate",
    "safe_validate",
    
    # 상수들
    "MEDICAL_KEYWORDS",
]