"""
공통 기본 모델 정의
~/app/models/base_models.py

모든 모듈에서 공통으로 사용할 기본 모델들 정의
"""

from typing import Any, Optional, Dict, List
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
import uuid


# =============================================================================
# 공통 Enum 정의
# =============================================================================

class ServiceStatus(str, Enum):
    """서비스 상태"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    LOADING = "loading"
    ERROR = "error"


class LogLevel(str, Enum):
    """로그 레벨"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class APIVersion(str, Enum):
    """API 버전"""
    V1 = "v1"
    V2 = "v2"


# =============================================================================
# 기본 Response 모델
# =============================================================================

class BaseResponse(BaseModel):
    """기본 응답 모델"""
    
    success: bool = Field(
        ...,
        description="요청 성공 여부"
    )
    
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="응답 생성 시간"
    )
    
    request_id: Optional[str] = Field(
        default=None,
        description="요청 추적 ID"
    )
    
    version: str = Field(
        default="1.0.0",
        description="API 버전"
    )


class SuccessResponse(BaseResponse):
    """성공 응답 모델"""
    
    success: bool = Field(default=True)
    data: Any = Field(..., description="응답 데이터")
    message: Optional[str] = Field(default=None, description="성공 메시지")


class ErrorResponse(BaseResponse):
    """에러 응답 모델"""
    
    success: bool = Field(default=False)
    error_code: str = Field(..., description="에러 코드")
    error_message: str = Field(..., description="에러 메시지")
    details: Optional[Dict[str, Any]] = Field(default=None, description="에러 상세 정보")


# =============================================================================
# 페이지네이션 모델
# =============================================================================

class PaginationInfo(BaseModel):
    """페이지네이션 정보"""
    
    page: int = Field(
        default=1,
        ge=1,
        description="현재 페이지"
    )
    
    size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="페이지 크기"
    )
    
    total: int = Field(
        default=0,
        ge=0,
        description="전체 항목 수"
    )
    
    total_pages: int = Field(
        default=0,
        ge=0,
        description="전체 페이지 수"
    )
    
    has_next: bool = Field(
        default=False,
        description="다음 페이지 존재 여부"
    )
    
    has_prev: bool = Field(
        default=False,
        description="이전 페이지 존재 여부"
    )


class PaginatedResponse(BaseResponse):
    """페이지네이션된 응답"""
    
    success: bool = Field(default=True)
    data: List[Any] = Field(default_factory=list, description="데이터 목록")
    pagination: PaginationInfo = Field(..., description="페이지네이션 정보")


# =============================================================================
# 메타데이터 모델
# =============================================================================

class ProcessingMetadata(BaseModel):
    """처리 메타데이터"""
    
    processing_time: float = Field(
        default=0.0,
        description="처리 시간 (초)"
    )
    
    memory_usage: Optional[float] = Field(
        default=None,
        description="메모리 사용량 (MB)"
    )
    
    cache_hit: Optional[bool] = Field(
        default=None,
        description="캐시 히트 여부"
    )
    
    model_version: Optional[str] = Field(
        default=None,
        description="사용된 모델 버전"
    )


class ServiceInfo(BaseModel):
    """서비스 정보"""
    
    name: str = Field(..., description="서비스 이름")
    status: ServiceStatus = Field(..., description="서비스 상태")
    version: str = Field(..., description="서비스 버전")
    uptime: Optional[float] = Field(default=None, description="가동 시간 (초)")
    last_updated: Optional[str] = Field(default=None, description="마지막 업데이트")


# =============================================================================
# 검색 관련 모델
# =============================================================================

class SearchQuery(BaseModel):
    """검색 쿼리"""
    
    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="검색어"
    )
    
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="결과 개수 제한"
    )
    
    threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="최소 유사도 임계값"
    )


class SearchResult(BaseModel):
    """검색 결과"""
    
    id: str = Field(..., description="결과 ID")
    score: float = Field(..., description="유사도 점수")
    content: str = Field(..., description="검색된 내용")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="메타데이터")
    rank: int = Field(..., description="순위")


class SearchResponse(BaseResponse):
    """검색 응답"""
    
    success: bool = Field(default=True)
    query: str = Field(..., description="검색어")
    results: List[SearchResult] = Field(default_factory=list, description="검색 결과")
    total_found: int = Field(default=0, description="전체 검색 결과 수")
    processing_time: float = Field(default=0.0, description="검색 시간")


# =============================================================================
# 로그 모델
# =============================================================================

class LogEntry(BaseModel):
    """로그 엔트리"""
    
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="로그 시간"
    )
    
    level: LogLevel = Field(..., description="로그 레벨")
    logger: str = Field(..., description="로거 이름")
    message: str = Field(..., description="로그 메시지")
    
    request_id: Optional[str] = Field(default=None, description="요청 ID")
    user_id: Optional[str] = Field(default=None, description="사용자 ID")
    session_id: Optional[str] = Field(default=None, description="세션 ID")
    
    extra: Optional[Dict[str, Any]] = Field(default=None, description="추가 정보")


# =============================================================================
# 설정 모델
# =============================================================================

class APIConfig(BaseModel):
    """API 설정"""
    
    title: str = Field(default="Medical Disease API", description="API 제목")
    description: str = Field(default="Disease diagnosis API", description="API 설명")
    version: str = Field(default="1.0.0", description="API 버전")
    
    host: str = Field(default="0.0.0.0", description="호스트")
    port: int = Field(default=8000, description="포트")
    debug: bool = Field(default=False, description="디버그 모드")
    
    cors_origins: List[str] = Field(
        default_factory=lambda: ["*"],
        description="CORS 허용 도메인"
    )


# =============================================================================
# 유틸리티 함수들
# =============================================================================

def generate_request_id() -> str:
    """요청 ID 생성"""
    return str(uuid.uuid4())


def create_success_response(data: Any, message: str = None, request_id: str = None) -> SuccessResponse:
    """성공 응답 생성"""
    return SuccessResponse(
        data=data,
        message=message,
        request_id=request_id
    )


def create_error_response(
    error_code: str,
    error_message: str,
    details: Dict[str, Any] = None,
    request_id: str = None
) -> ErrorResponse:
    """에러 응답 생성"""
    return ErrorResponse(
        error_code=error_code,
        error_message=error_message,
        details=details,
        request_id=request_id
    )


def create_paginated_response(
    data: List[Any],
    page: int,
    size: int,
    total: int,
    request_id: str = None
) -> PaginatedResponse:
    """페이지네이션된 응답 생성"""
    total_pages = (total + size - 1) // size
    
    pagination = PaginationInfo(
        page=page,
        size=size,
        total=total,
        total_pages=total_pages,
        has_next=page < total_pages,
        has_prev=page > 1
    )
    
    return PaginatedResponse(
        data=data,
        pagination=pagination,
        request_id=request_id
    )


# =============================================================================
# 검증 함수들
# =============================================================================

def validate_uuid(value: str) -> bool:
    """UUID 형식 검증"""
    try:
        uuid.UUID(value)
        return True
    except ValueError:
        return False


def validate_timestamp(value: str) -> bool:
    """타임스탬프 형식 검증"""
    try:
        datetime.fromisoformat(value.replace('Z', '+00:00'))
        return True
    except ValueError:
        return False