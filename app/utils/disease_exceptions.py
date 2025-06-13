"""
질병 모듈 커스텀 예외 정의
~/app/utils/disease_exceptions.py

질병 API에서 발생할 수 있는 모든 예외 상황을 정의
"""

from typing import Optional, Dict, Any


class DiseaseBaseException(Exception):
    """질병 API 기본 예외 클래스"""
    
    def __init__(self, message: str, error_code: str = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """예외를 딕셔너리로 변환 (API 응답용)"""
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details
        }


# =============================================================================
# 입력 검증 관련 예외
# =============================================================================

class DiseaseValidationError(DiseaseBaseException):
    """질병 입력 검증 오류"""
    pass


class EmptyMessageError(DiseaseValidationError):
    """빈 메시지 오류"""
    
    def __init__(self):
        super().__init__(
            message="입력 메시지가 비어있습니다.",
            error_code="EMPTY_MESSAGE"
        )


class MessageTooShortError(DiseaseValidationError):
    """메시지 너무 짧음 오류"""
    
    def __init__(self, length: int, min_length: int):
        super().__init__(
            message=f"입력 메시지가 너무 짧습니다. 최소 {min_length}자 이상 입력해주세요.",
            error_code="MESSAGE_TOO_SHORT",
            details={"length": length, "min_length": min_length}
        )


class MessageTooLongError(DiseaseValidationError):
    """메시지 너무 긺 오류"""
    
    def __init__(self, length: int, max_length: int):
        super().__init__(
            message=f"입력 메시지가 너무 깁니다. 최대 {max_length}자까지 입력 가능합니다.",
            error_code="MESSAGE_TOO_LONG",
            details={"length": length, "max_length": max_length}
        )


class InvalidCharacterError(DiseaseValidationError):
    """유효하지 않은 문자 오류"""
    
    def __init__(self, invalid_chars: str = ""):
        super().__init__(
            message="유효하지 않은 문자가 포함되어 있습니다. 한글, 영문, 숫자, 기본 문장부호만 사용해주세요.",
            error_code="INVALID_CHARACTERS",
            details={"invalid_chars": invalid_chars}
        )


# =============================================================================
# FAISS 관련 예외
# =============================================================================

class DiseaseFaissError(DiseaseBaseException):
    """질병 FAISS 관련 오류"""
    pass


class FaissDirectoryNotFoundError(DiseaseFaissError):
    """FAISS 디렉토리 없음 오류"""
    
    def __init__(self, directory_path: str):
        super().__init__(
            message=f"FAISS 인덱스 디렉토리를 찾을 수 없습니다: {directory_path}",
            error_code="FAISS_DIRECTORY_NOT_FOUND",
            details={"directory_path": directory_path}
        )


class FaissFileNotFoundError(DiseaseFaissError):
    """FAISS 파일 없음 오류"""
    
    def __init__(self, file_path: str):
        super().__init__(
            message=f"FAISS 인덱스 파일을 찾을 수 없습니다: {file_path}",
            error_code="FAISS_FILE_NOT_FOUND",
            details={"file_path": file_path}
        )


class FaissLoadError(DiseaseFaissError):
    """FAISS 로드 오류"""
    
    def __init__(self, file_path: str, original_error: str):
        super().__init__(
            message=f"FAISS 인덱스 로드 중 오류가 발생했습니다: {file_path}",
            error_code="FAISS_LOAD_ERROR",
            details={"file_path": file_path, "original_error": original_error}
        )


# =============================================================================
# 임베딩 관련 예외
# =============================================================================

class DiseaseEmbeddingError(DiseaseBaseException):
    """질병 임베딩 관련 오류"""
    pass


class EmbeddingModelLoadError(DiseaseEmbeddingError):
    """임베딩 모델 로드 오류"""
    
    def __init__(self, model_name: str, original_error: str):
        super().__init__(
            message=f"임베딩 모델 로드 중 오류가 발생했습니다: {model_name}",
            error_code="EMBEDDING_MODEL_LOAD_ERROR",
            details={"model_name": model_name, "original_error": original_error}
        )


class EmbeddingGenerationError(DiseaseEmbeddingError):
    """임베딩 생성 오류"""
    
    def __init__(self, text: str, original_error: str):
        super().__init__(
            message="텍스트 임베딩 생성 중 오류가 발생했습니다.",
            error_code="EMBEDDING_GENERATION_ERROR",
            details={"text_length": len(text), "original_error": original_error}
        )


# =============================================================================
# RAG 검색 관련 예외
# =============================================================================

class DiseaseRagError(DiseaseBaseException):
    """질병 RAG 검색 관련 오류"""
    pass


class RagIndexNotLoadedError(DiseaseRagError):
    """RAG 인덱스 로드되지 않음 오류"""
    
    def __init__(self, index_type: str):
        super().__init__(
            message=f"RAG 인덱스가 로드되지 않았습니다: {index_type}",
            error_code="RAG_INDEX_NOT_LOADED",
            details={"index_type": index_type}
        )


class RagSearchError(DiseaseRagError):
    """RAG 검색 오류"""
    
    def __init__(self, query: str, original_error: str):
        super().__init__(
            message="RAG 검색 중 오류가 발생했습니다.",
            error_code="RAG_SEARCH_ERROR",
            details={"query_length": len(query), "original_error": original_error}
        )


# =============================================================================
# 질병 진단 관련 예외
# =============================================================================

class DiseaseDiagnosisError(DiseaseBaseException):
    """질병 진단 관련 오류"""
    pass


class ExaoneConnectionError(DiseaseDiagnosisError):
    """EXAONE LLM 연결 오류"""
    
    def __init__(self, url: str, original_error: str):
        super().__init__(
            message="EXAONE LLM 서버에 연결할 수 없습니다. Ollama가 실행 중인지 확인해주세요.",
            error_code="EXAONE_CONNECTION_ERROR",
            details={"url": url, "original_error": original_error}
        )


class ExaoneResponseError(DiseaseDiagnosisError):
    """EXAONE 응답 오류"""
    
    def __init__(self, status_code: int, response_text: str):
        super().__init__(
            message="EXAONE LLM에서 응답 생성 중 오류가 발생했습니다.",
            error_code="EXAONE_RESPONSE_ERROR",
            details={"status_code": status_code, "response_text": response_text}
        )


class DiagnosisTimeoutError(DiseaseDiagnosisError):
    """진단 시간 초과 오류"""
    
    def __init__(self, timeout_seconds: float):
        super().__init__(
            message=f"진단 처리 시간이 초과되었습니다 ({timeout_seconds}초). 다시 시도해주세요.",
            error_code="DIAGNOSIS_TIMEOUT",
            details={"timeout_seconds": timeout_seconds}
        )


class LowConfidenceError(DiseaseDiagnosisError):
    """낮은 신뢰도 오류"""
    
    def __init__(self, confidence: float, threshold: float):
        super().__init__(
            message="입력된 증상으로는 정확한 진단이 어렵습니다. 증상을 더 자세히 설명해주세요.",
            error_code="LOW_CONFIDENCE",
            details={"confidence": confidence, "threshold": threshold}
        )


# =============================================================================
# 시스템 관련 예외
# =============================================================================

class DiseaseSystemError(DiseaseBaseException):
    """질병 시스템 관련 오류"""
    pass


class ServiceNotInitializedError(DiseaseSystemError):
    """서비스 초기화되지 않음 오류"""
    
    def __init__(self, service_name: str):
        super().__init__(
            message=f"서비스가 초기화되지 않았습니다: {service_name}",
            error_code="SERVICE_NOT_INITIALIZED",
            details={"service_name": service_name}
        )


class ConfigurationError(DiseaseSystemError):
    """설정 오류"""
    
    def __init__(self, config_key: str, expected_type: str = None):
        super().__init__(
            message=f"설정 값이 올바르지 않습니다: {config_key}",
            error_code="CONFIGURATION_ERROR",
            details={"config_key": config_key, "expected_type": expected_type}
        )


# =============================================================================
# 예외 헬퍼 함수들
# =============================================================================

def handle_validation_error(func):
    """검증 예외 데코레이터"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except DiseaseValidationError:
            raise  # 다시 던지기
        except Exception as e:
            raise DiseaseValidationError(f"검증 중 예상치 못한 오류: {str(e)}")
    return wrapper


def handle_faiss_error(func):
    """FAISS 예외 데코레이터"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except DiseaseFaissError:
            raise  # 다시 던지기
        except FileNotFoundError as e:
            raise FaissFileNotFoundError(str(e))
        except Exception as e:
            raise DiseaseFaissError(f"FAISS 처리 중 예상치 못한 오류: {str(e)}")
    return wrapper


def handle_diagnosis_error(func):
    """진단 예외 데코레이터"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except DiseaseDiagnosisError:
            raise  # 다시 던지기
        except Exception as e:
            raise DiseaseDiagnosisError(f"진단 처리 중 예상치 못한 오류: {str(e)}")
    return wrapper