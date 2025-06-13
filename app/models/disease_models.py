"""
질병 모듈 데이터 모델 정의
~/app/models/disease_models.py

질병 API의 Request/Response 모델 및 내부 데이터 구조 정의
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime
import uuid


# =============================================================================
# Request 모델
# =============================================================================

class DiseaseRequest(BaseModel):
    """질병 진단 요청 모델"""
    
    message: str = Field(
        ...,
        description="사용자 증상 메시지",
        example="머리가 아프고 열이 나요"
    )
    
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="추가 컨텍스트 정보 (향후 확장용)",
        example={"previous_symptoms": ["두통"], "age": "30대"}
    )
    
    request_id: Optional[str] = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="요청 고유 ID"
    )
    
    @validator('message')
    def validate_message(cls, v):
        """메시지 기본 검증"""
        if not v or not v.strip():
            raise ValueError("메시지가 비어있습니다")
        return v.strip()
    
    class Config:
        schema_extra = {
            "example": {
                "message": "머리가 아프고 열이 나면서 기침도 해요",
                "context": {
                    "age_group": "30대",
                    "chronic_conditions": []
                }
            }
        }


# =============================================================================
# Response 모델
# =============================================================================

class DiseaseResponse(BaseModel):
    """질병 진단 응답 모델"""
    
    # 필수 응답 필드
    diagnosis: str = Field(
        ...,
        description="진단된 질병명",
        example="감기"
    )
    
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="진단 신뢰도 (0.0 ~ 1.0)",
        example=0.85
    )
    
    department: str = Field(
        ...,
        description="진료과",
        example="내과"
    )
    
    symptoms: List[str] = Field(
        default_factory=list,
        description="감지된 주요 증상들",
        example=["두통", "발열", "기침"]
    )
    
    recommendations: str = Field(
        ...,
        description="생활 관리 및 치료 권장사항",
        example="충분한 휴식과 수분 섭취를 권장합니다..."
    )
    
    reasoning: str = Field(
        ...,
        description="진단 근거 및 추론 설명",
        example="두통과 발열은 감기의 대표적인 초기 증상입니다..."
    )
    
    disclaimer: str = Field(
        ...,
        description="의료 면책 조항",
        example="⚠️ 이는 참고용이며 실제 진료를 대체하지 않습니다..."
    )
    
    # 메타데이터 필드
    response_time: float = Field(
        ...,
        description="응답 시간 (초)",
        example=2.3
    )
    
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="응답 생성 시간",
        example="2024-06-12T15:30:00.123456"
    )
    
    request_id: Optional[str] = Field(
        default=None,
        description="요청 ID (추적용)"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "diagnosis": "감기",
                "confidence": 0.85,
                "department": "내과",
                "symptoms": ["두통", "발열", "기침"],
                "recommendations": "충분한 휴식과 수분 섭취를 권장합니다. 증상이 3-4일 이상 지속되거나 고열이 계속될 경우 병원 진료를 받으시기 바랍니다.",
                "reasoning": "두통과 발열은 감기의 대표적인 초기 증상입니다. 기침이 동반되어 상기도 감염 가능성이 높습니다. 유사 사례 분석 결과 85% 확률로 일반 감기로 추정됩니다.",
                "disclaimer": "⚠️ 이는 참고용이며 실제 진료를 대체하지 않습니다. 증상이 지속되거나 악화될 경우 반드시 의료진의 진료를 받으시기 바랍니다.",
                "response_time": 2.3,
                "timestamp": "2024-06-12T15:30:00.123456",
                "request_id": "550e8400-e29b-41d4-a716-446655440000"
            }
        }


# =============================================================================
# 에러 응답 모델
# =============================================================================

class DiseaseErrorResponse(BaseModel):
    """질병 API 에러 응답 모델"""
    
    success: bool = Field(
        default=False,
        description="요청 성공 여부"
    )
    
    error: str = Field(
        ...,
        description="에러 코드",
        example="VALIDATION_ERROR"
    )
    
    message: str = Field(
        ...,
        description="에러 메시지",
        example="입력 메시지가 너무 짧습니다"
    )
    
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="에러 상세 정보"
    )
    
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="에러 발생 시간"
    )
    
    request_id: Optional[str] = Field(
        default=None,
        description="요청 ID (추적용)"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "success": False,
                "error": "MESSAGE_TOO_SHORT",
                "message": "입력 메시지가 너무 짧습니다. 최소 2자 이상 입력해주세요.",
                "details": {
                    "length": 1,
                    "min_length": 2
                },
                "timestamp": "2024-06-12T15:30:00.123456",
                "request_id": "550e8400-e29b-41d4-a716-446655440000"
            }
        }


# =============================================================================
# 내부 데이터 모델
# =============================================================================

class DiseaseMetadata(BaseModel):
    """질병 메타데이터 모델 (FAISS에서 로드된 데이터)"""
    
    disease: str = Field(..., description="질병명")
    symptoms: str = Field(..., description="증상 설명")
    department: Optional[str] = Field(default=None, description="진료과")
    key_text: str = Field(..., description="키워드 검색용 텍스트")
    full_text: str = Field(..., description="전체 검색용 텍스트")
    source_file: Optional[str] = Field(default=None, description="원본 파일명")


class RagDocument(BaseModel):
    """RAG 문서 모델"""
    
    doc_id: str = Field(..., description="문서 ID")
    content: str = Field(..., description="문서 내용")
    metadata: Dict[str, Any] = Field(..., description="문서 메타데이터")
    content_type: str = Field(..., description="문서 타입 (qa/medical_doc)")
    embedding: Optional[List[float]] = Field(default=None, description="임베딩 벡터")


class SearchResult(BaseModel):
    """검색 결과 모델"""
    
    score: float = Field(..., description="유사도 점수")
    metadata: DiseaseMetadata = Field(..., description="질병 메타데이터")
    rank: int = Field(..., description="검색 순위")


class DiagnosisContext(BaseModel):
    """진단 컨텍스트 모델 (내부 처리용)"""
    
    user_message: str = Field(..., description="사용자 메시지")
    search_results: List[SearchResult] = Field(default_factory=list, description="벡터 검색 결과")
    rag_results: List[RagDocument] = Field(default_factory=list, description="RAG 검색 결과")
    extracted_symptoms: List[str] = Field(default_factory=list, description="추출된 증상")
    confidence_score: float = Field(default=0.0, description="신뢰도 점수")
    processing_time: float = Field(default=0.0, description="처리 시간")


# =============================================================================
# 헬스체크 모델
# =============================================================================

class HealthCheckResponse(BaseModel):
    """헬스체크 응답 모델"""
    
    status: str = Field(
        default="healthy",
        description="서비스 상태",
        example="healthy"
    )
    
    message: str = Field(
        default="Disease API is running normally",
        description="상태 메시지"
    )
    
    services: Dict[str, str] = Field(
        default_factory=dict,
        description="각 서비스 상태",
        example={
            "faiss_loader": "healthy",
            "embedding_service": "healthy",
            "rag_service": "healthy",
            "exaone_llm": "healthy"
        }
    )
    
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="체크 시간"
    )
    
    version: str = Field(
        default="1.0.0",
        description="API 버전"
    )


# =============================================================================
# 공통 응답 래퍼
# =============================================================================

class APIResponse(BaseModel):
    """공통 API 응답 래퍼"""
    
    success: bool = Field(..., description="요청 성공 여부")
    data: Optional[Any] = Field(default=None, description="응답 데이터")
    error: Optional[DiseaseErrorResponse] = Field(default=None, description="에러 정보")
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="응답 시간"
    )
    
    @classmethod
    def success_response(cls, data: Any):
        """성공 응답 생성"""
        return cls(success=True, data=data, error=None)
    
    @classmethod  
    def error_response(cls, error: DiseaseErrorResponse):
        """에러 응답 생성"""
        return cls(success=False, data=None, error=error)