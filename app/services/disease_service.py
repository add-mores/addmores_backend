"""
질병 진단 서비스
위치: ~/backend/app/services/disease_service.py

🎯 목적: 질병 진단 핵심 로직 제공
📋 기능:
   - 증상 기반 질병 진단
   - FAISS 벡터 검색
   - RAG 검색 결합
   - EXAONE LLM 추론
   - 질병 정보 제공

⚙️ 의존성: faiss, numpy, requests, re
"""

import faiss
import numpy as np
import requests
import re
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from .disease_embedding_service import get_embedding_service
from .disease_faiss_loader import get_faiss_loader
from .disease_rag_service import get_rag_service
from ..models.disease_models import DiseaseRequest, DiseaseResponse
from ..utils.disease_constants import (
    EXAONE_BASE_URL, EXAONE_MODEL_NAME, EXAONE_CONFIG,
    DEPARTMENT_MAPPING, DEFAULT_DEPARTMENT, DISCLAIMER_TEXT
)
from ..utils.disease_exceptions import (
    DiseaseDiagnosisError, ExaoneConnectionError, ExaoneResponseError
)
from ..utils.disease_validators import DiseaseValidator

logger = logging.getLogger(__name__)


class DiseaseService:
    """질병 진단 서비스 클래스"""
    
    def __init__(self):
        self.embedding_service = get_embedding_service()
        self.faiss_loader = get_faiss_loader()
        self.rag_service = get_rag_service()
        self.validator = DiseaseValidator()
        
        # FAISS 인덱스와 메타데이터
        self.disease_key_index: Optional[faiss.IndexFlatIP] = None
        self.disease_full_index: Optional[faiss.IndexFlatIP] = None
        self.disease_metadata: List[Dict] = []
        
        # EXAONE 설정
        self.exaone_url = EXAONE_BASE_URL.rstrip("/")
        self.model_name = EXAONE_MODEL_NAME
        self.exaone_endpoint = None
        
        # EXAONE 설정 정보 (최적화됨)
        self.exaone_config = {
            "temperature": 0.2,      # 더 일관된 응답을 위해 낮춤
            "top_p": 0.8,
            "max_tokens": 1500,      # 토큰 수 제한으로 응답 시간 단축
            "repeat_penalty": 1.1,
            "num_predict": 1500,     # 예측 토큰 수 제한
            "stop": ["사용자:", "환자:", "Human:", "Assistant:", "\n\n---", "질문:"]
        }
        
        self.is_initialized = False
        logger.info("🏥 질병 진단 서비스 초기화됨")
    
    def initialize(self) -> bool:
        """질병 서비스 초기화"""
        try:
            logger.info("🔄 질병 서비스 초기화 중...")
            
            # 의존 서비스 확인
            if not self.embedding_service.is_loaded:
                raise DiseaseDiagnosisError("임베딩 서비스가 초기화되지 않았습니다.")
            
            if not self.faiss_loader.is_loaded:
                raise DiseaseDiagnosisError("FAISS 인덱스가 로드되지 않았습니다.")
            
            if not self.rag_service.is_initialized:
                raise DiseaseDiagnosisError("RAG 서비스가 초기화되지 않았습니다.")
            
            # 질병 인덱스와 메타데이터 로드
            self.disease_key_index, self.disease_full_index = self.faiss_loader.get_disease_indexes()
            self.disease_metadata = self.faiss_loader.get_disease_metadata()
            
            # EXAONE 연결 확인
            self._initialize_exaone_connection()
            
            # 상태 로깅
            self._log_service_status()
            
            self.is_initialized = True
            logger.info("✅ 질병 서비스 초기화 완료!")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 질병 서비스 초기화 실패: {e}")
            self.is_initialized = False
            raise DiseaseDiagnosisError(f"질병 서비스 초기화 실패: {e}")
    
    def _initialize_exaone_connection(self):
        """EXAONE 연결 초기화"""
        logger.info("🤖 EXAONE 연결 확인 중...")
        
        # generate 엔드포인트 확인
        if self._check_exaone_endpoint("generate"):
            self.exaone_endpoint = "generate"
            logger.info("✅ EXAONE generate 엔드포인트 연결됨")
        # chat 엔드포인트 확인
        elif self._check_exaone_endpoint("chat"):
            self.exaone_endpoint = "chat"
            logger.info("✅ EXAONE chat 엔드포인트 연결됨")
        else:
            logger.warning("⚠️ EXAONE 서버에 연결할 수 없습니다. 기본 모드로 실행됩니다.")
            self.exaone_endpoint = None
            return
        
        # 모델 미리 로드 (warming up)
        self._warm_up_exaone_model()
    
    def _warm_up_exaone_model(self):
        """EXAONE 모델 미리 로드 (응답 속도 향상)"""
        try:
            logger.info("🔥 EXAONE 모델 워밍업 중...")
            
            # 간단한 테스트 쿼리로 모델 로드
            warm_up_prompt = "안녕하세요"
            
            if self.exaone_endpoint == "chat":
                payload = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": warm_up_prompt}],
                    "stream": False,
                    "options": {"num_predict": 50}  # 짧은 응답
                }
                requests.post(f"{self.exaone_url}/api/chat", json=payload, timeout=10)
            else:
                payload = {
                    "model": self.model_name,
                    "prompt": warm_up_prompt,
                    "stream": False,
                    "options": {"num_predict": 50}
                }
                requests.post(f"{self.exaone_url}/api/generate", json=payload, timeout=10)
            
            logger.info("✅ EXAONE 모델 워밍업 완료")
            
        except Exception as e:
            logger.warning(f"⚠️ EXAONE 모델 워밍업 실패: {e}")
    
    def _check_exaone_endpoint(self, endpoint: str) -> bool:
        """EXAONE 엔드포인트 연결 확인"""
        try:
            response = requests.get(f"{self.exaone_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def _log_service_status(self):
        """서비스 상태 로깅"""
        logger.info("📊 질병 서비스 상태:")
        logger.info(f"   - 질병 Key 인덱스: {self.disease_key_index.ntotal}개 벡터")
        logger.info(f"   - 질병 Full 인덱스: {self.disease_full_index.ntotal}개 벡터")
        logger.info(f"   - 질병 메타데이터: {len(self.disease_metadata)}개 문서")
        logger.info(f"   - EXAONE 연결: {'연결됨' if self.exaone_endpoint else '연결 안됨'}")
    
    def _call_exaone(self, prompt: str, system_prompt: str = "") -> str:
        """
        EXAONE LLM 호출 메서드 (누락된 메서드 구현)
        
        Args:
            prompt: 사용자 프롬프트
            system_prompt: 시스템 프롬프트
            
        Returns:
            EXAONE 응답 텍스트
        """
        if not self.exaone_endpoint:
            return "⚠️ EXAONE 서버에 연결할 수 없습니다. Ollama가 실행 중인지 확인해주세요."
        
        try:
            # 요청 페이로드 구성
            if self.exaone_endpoint == "chat":
                # Chat 엔드포인트 사용
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                payload = {
                    "model": self.model_name,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": self.exaone_config["temperature"],
                        "top_p": self.exaone_config["top_p"],
                        "repeat_penalty": self.exaone_config["repeat_penalty"],
                        "num_predict": self.exaone_config["num_predict"],
                        "stop": self.exaone_config["stop"]
                    }
                }
                
                response = requests.post(
                    f"{self.exaone_url}/api/chat",
                    json=payload,
                    timeout=60,  # 60초로 증가
                    headers={
                        'Content-Type': 'application/json',
                        'Accept': 'application/json'
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get("message", {}).get("content", "응답을 받을 수 없습니다.")
                else:
                    logger.error(f"EXAONE chat 호출 실패: {response.status_code}")
                    return "AI 모델 호출에 실패했습니다."
                    
            else:
                # Generate 엔드포인트 사용
                full_prompt = prompt
                if system_prompt:
                    full_prompt = f"{system_prompt}\n\n{prompt}"
                
                payload = {
                    "model": self.model_name,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.exaone_config["temperature"],
                        "top_p": self.exaone_config["top_p"],
                        "repeat_penalty": self.exaone_config["repeat_penalty"],
                        "num_predict": self.exaone_config["num_predict"],
                        "stop": self.exaone_config["stop"]
                    }
                }
                
                response = requests.post(
                    f"{self.exaone_url}/api/generate",
                    json=payload,
                    timeout=60  # 60초로 증가
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get("response", "응답을 받을 수 없습니다.")
                else:
                    logger.error(f"EXAONE generate 호출 실패: {response.status_code}")
                    return "AI 모델 호출에 실패했습니다."
                    
        except requests.exceptions.Timeout:
            logger.error("EXAONE 호출 타임아웃")
            return "⏰ AI 분석 중입니다. 복잡한 의료 정보 생성에 시간이 걸리고 있습니다. 잠시 후 다시 시도해주세요."
        except requests.exceptions.ConnectionError:
            logger.error("EXAONE 연결 오류")
            return "AI 모델 서버에 연결할 수 없습니다. 서버 상태를 확인해주세요."
        except Exception as e:
            logger.error(f"EXAONE 호출 중 예외 발생: {e}")
            return f"AI 모델 호출 중 오류가 발생했습니다: {str(e)}"
    
    def diagnose_disease(self, request: DiseaseRequest) -> DiseaseResponse:
        """질병 진단 메인 함수"""
        if not self.is_initialized:
            raise DiseaseDiagnosisError("질병 서비스가 초기화되지 않았습니다.")
        
        start_time = datetime.now()
        
        try:
            logger.info(f"🔍 질병 진단 시작: '{request.message}'")
            
            # 1. 입력 검증
            self.validator.validate_message(request.message)
            
            # 2. 벡터 검색으로 유사한 질병 찾기
            similar_diseases = self._search_similar_diseases(request.message, top_k=3)
            
            # 3. RAG 검색으로 관련 정보 찾기
            rag_context = self.rag_service.get_rag_context(request.message, max_chars=800)
            
            # 4. EXAONE을 이용한 진단
            diagnosis_result = self._generate_diagnosis_with_exaone(
                request.message, similar_diseases, rag_context
            )
            
            # 5. 응답 구성
            response = self._build_diagnosis_response(
                diagnosis_result, request.message, start_time
            )
            
            logger.info(f"✅ 질병 진단 완료: {response.diagnosis} (진료과: {response.department})")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ 질병 진단 실패: {e}")
            # 오류 발생 시 기본 응답 반환
            return self._build_error_response(str(e), start_time)
    
    def get_disease_info(self, disease_name: str) -> DiseaseResponse:
        """질병 정보 제공"""
        if not self.is_initialized:
            raise DiseaseDiagnosisError("질병 서비스가 초기화되지 않았습니다.")
        
        start_time = datetime.now()
        
        try:
            logger.info(f"📚 질병 정보 요청: '{disease_name}'")
            
            # 1. 입력 검증
            self.validator.validate_message(disease_name)
            
            # 2. 관련 질병 정보 검색
            related_diseases = self._search_similar_diseases(disease_name, top_k=2)
            
            # 3. RAG 검색
            rag_context = self.rag_service.get_rag_context(disease_name, max_chars=800)
            
            # 4. EXAONE을 이용한 정보 제공
            info_result = self._generate_disease_info_with_exaone(
                disease_name, related_diseases, rag_context
            )
            
            # 5. 응답 구성
            response = self._build_info_response(info_result, disease_name, start_time)
            
            logger.info(f"✅ 질병 정보 제공 완료: {disease_name}")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ 질병 정보 제공 실패: {e}")
            return self._build_error_response(str(e), start_time)
    
    def _search_similar_diseases(self, query: str, top_k: int = 3) -> List[Dict]:
        """벡터 검색으로 유사한 질병 찾기"""
        try:
            # 쿼리 임베딩 생성
            query_embedding = self.embedding_service.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # Key 인덱스에서 검색
            scores, indices = self.disease_key_index.search(query_embedding, top_k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if 0 <= idx < len(self.disease_metadata):
                    disease_info = self.disease_metadata[idx].copy()
                    disease_info['similarity_score'] = float(score)
                    results.append(disease_info)
            
            logger.debug(f"🔍 유사 질병 검색 완료: {len(results)}개")
            return results
            
        except Exception as e:
            logger.error(f"❌ 질병 벡터 검색 실패: {e}")
            return []
    
    def _generate_diagnosis_with_exaone(self, symptoms: str, similar_diseases: List[Dict], rag_context: str) -> str:
        """EXAONE을 이용한 질병 진단"""
        
        # 컨텍스트 정보 구성
        context_info = ""
        
        if similar_diseases:
            context_info += "\n🔍 유사한 질병 데이터:\n"
            for i, disease in enumerate(similar_diseases, 1):
                context_info += f"{i}. {disease['disease']}: {disease['symptoms']}\n"
        
        if rag_context:
            context_info += f"\n{rag_context}"
        
        # ✅ 강화된 시스템 프롬프트
        system_prompt = """당신은 20년 경력의 임상의학 전문의입니다. 
다양한 진료과 경험을 바탕으로 환자의 증상을 체계적으로 분석하고 정확한 진단을 내립니다.

진단 원칙:
- 증상의 빈도, 지속시간, 강도, 유발요인을 종합적으로 분석
- 감별진단을 통한 다른 질병 배제 과정 고려
- 연령, 성별, 기저질환 등 환자 특성 반영
- 객관적 근거와 임상경험을 바탕으로 신뢰도 있는 진단
- Red flag 증상 인지 시 즉시 전문의 진료 권유

응답 시 반드시 의학적 근거를 제시하고, 불확실한 경우 솔직히 인정하여 추가 검사나 전문의 상담을 권하세요."""

        # ✅ 효율적인 사용자 프롬프트 (진단용)
        user_prompt = f"""
환자 주증상: {symptoms}
{context_info}

위 정보를 바탕으로 다음과 같이 간결하게 분석해주세요:

1. **예상 진단**: 가장 가능성 높은 질병명
2. **의학적 근거**: 해당 진단의 주요 근거 (2-3문장)
3. **추가 확인사항**: 정확한 진단을 위해 확인이 필요한 사항
4. **즉시 조치**: 응급성 여부 및 병원 방문 필요성
5. **생활관리**: 증상 완화를 위한 기본 관리법

전체 답변은 600자 이내로 핵심만 작성해주세요.

⚠️ 주의: 이는 예비 진단이며, 정확한 진단을 위해서는 반드시 의료진의 직접 진료가 필요합니다.

전문의 소견:"""

        # EXAONE 호출
        exaone_response = self._call_exaone(user_prompt, system_prompt)
        
        # Fallback 메커니즘: EXAONE 실패 시 기본 진단 제공
        if "AI 분석 중입니다" in exaone_response or "연결할 수 없습니다" in exaone_response:
            return self._generate_fallback_diagnosis(symptoms, similar_diseases)
        
        return exaone_response
    
    def _generate_fallback_diagnosis(self, symptoms: str, similar_diseases: List[Dict]) -> str:
        """EXAONE 실패 시 기본 진단 제공"""
        
        diagnosis_parts = []
        
        # 1. 기본 증상 분석
        diagnosis_parts.append(f"**증상 분석**: {symptoms}")
        
        # 2. 유사 질병 정보 활용
        if similar_diseases:
            most_similar = similar_diseases[0]
            diagnosis_parts.append(f"\n**예상 진단**: {most_similar.get('disease', '증상 분석 필요')}")
            diagnosis_parts.append(f"**관련 증상**: {most_similar.get('symptoms', '추가 정보 필요')}")
        else:
            diagnosis_parts.append(f"\n**예상 진단**: 증상에 대한 추가 분석이 필요합니다")
        
        # 3. 기본 권장사항
        diagnosis_parts.append(f"""
        
**기본 권장사항**:
- 증상이 지속되거나 악화될 경우 즉시 병원 진료
- 충분한 휴식과 수분 섭취
- 자가진단보다는 전문의 상담이 중요

**권장 진료과**: {self._determine_department(similar_diseases[0].get('disease', '') if similar_diseases else '')}

⚠️ 현재 AI 서버 응답이 지연되어 기본 분석만 제공됩니다. 
정확한 진단을 위해서는 반드시 의료진의 직접 진료를 받으시기 바랍니다.""")
        
        return "\n".join(diagnosis_parts)


    def _generate_disease_info_with_exaone(self, disease_name: str, related_diseases: List[Dict], rag_context: str) -> str:
        """EXAONE을 이용한 질병 정보 제공"""
        
        # 컨텍스트 정보 구성
        context_info = ""
        
        if related_diseases:
            context_info += "\n🔍 데이터베이스 정보:\n"
            for disease in related_diseases:
                context_info += f"- {disease['disease']}: {disease['symptoms']}\n"
        
        if rag_context:
            context_info += f"\n{rag_context}"
        
        # ✅ 강화된 시스템 프롬프트 (정보 제공용)
        system_prompt = """당신은 의학박사 학위를 가진 의료정보 전문가입니다.
대학병원에서 임상경험과 의학교육을 병행하며, 최신 의학 지식과 근거중심의학(EBM)을 바탕으로 정확하고 신뢰할 수 있는 의료정보를 제공합니다.

정보 제공 원칙:
- 최신 의학 가이드라인과 진료지침 반영
- 근거 수준이 높은 연구결과 우선 인용
- 환자가 이해하기 쉬운 용어로 설명하되 의학적 정확성 유지
- 질병의 예후와 합병증에 대한 균형잡힌 정보 제공
- 자가진단의 위험성과 전문의 진료의 중요성 강조

모든 정보는 의학적 사실에 근거하여 제공하며, 불분명한 사항은 추가 연구나 전문의 상담이 필요함을 명시하세요."""

        # ✅ 효율적인 사용자 프롬프트 (정보 제공용)
        user_prompt = f"""
질병명: {disease_name}
{context_info}

다음 핵심 항목에 대해 간결하고 정확하게 설명해주세요:

1. **정의와 원인**: 질병의 의학적 정의와 주요 원인
2. **주요 증상**: 가장 흔한 증상과 특징적인 증상  
3. **진단 방법**: 진단에 필요한 검사나 기준
4. **치료 방법**: 일반적인 치료법과 관리법
5. **예방과 주의사항**: 예방 방법과 환자가 알아야 할 중요 정보

각 항목은 2-3문장으로 핵심만 설명하고, 전체 답변은 800자 이내로 작성해주세요.

의학 전문가 답변:"""

        # EXAONE 호출
        exaone_response = self._call_exaone(user_prompt, system_prompt)
        
        # Fallback 메커니즘: EXAONE 실패 시 기본 정보 제공
        if "AI 분석 중입니다" in exaone_response or "연결할 수 없습니다" in exaone_response:
            return self._generate_fallback_disease_info(disease_name, related_diseases, rag_context)
        
        return exaone_response
    
    def _generate_fallback_disease_info(self, disease_name: str, related_diseases: List[Dict], rag_context: str) -> str:
        """EXAONE 실패 시 기본 정보 제공"""
        
        info_parts = []
        
        # 1. 기본 정의
        info_parts.append(f"**{disease_name} 기본 정보**")
        
        # 2. 관련 질병 정보가 있다면 활용
        if related_diseases:
            for disease in related_diseases[:2]:  # 상위 2개만
                if disease_name.lower() in disease.get('disease', '').lower():
                    info_parts.append(f"\n**주요 증상**: {disease.get('symptoms', '정보 없음')}")
                    break
        
        # 3. RAG 컨텍스트가 있다면 활용
        if rag_context:
            info_parts.append(f"\n**추가 정보**:\n{rag_context[:300]}...")
        
        # 4. 일반적인 권장사항
        info_parts.append(f"""
        
**일반적인 권장사항**:
- 정확한 진단을 위해 병원 진료를 받으시기 바랍니다
- 증상이 악화되거나 지속될 경우 즉시 의료진 상담
- 자가진단보다는 전문의 진료가 중요합니다

**진료과**: {self._determine_department(disease_name)}

⚠️ 현재 AI 서버 응답이 지연되어 기본 정보만 제공됩니다. 
더 상세한 정보는 잠시 후 다시 시도해주세요.""")
        
        return "\n".join(info_parts)
    
    def _build_diagnosis_response(self, diagnosis_result: str, original_message: str, start_time: datetime) -> DiseaseResponse:
        """진단 응답 구성 (개선된 버전)"""
        
        # 진단 결과에서 정보 추출
        diagnosed_disease = self._extract_disease_from_diagnosis(diagnosis_result)
        symptoms = self._extract_symptoms_from_text(original_message)
        department = self._determine_department(diagnosed_disease)
        confidence = self._calculate_confidence(diagnosis_result)
        
        # 응답 시간 계산
        response_time = (datetime.now() - start_time).total_seconds()
        
        # 로깅
        logger.info(f"진단 파싱 결과: 질병='{diagnosed_disease}', 진료과='{department}', 신뢰도={confidence}")
        
        return DiseaseResponse(
            diagnosis=diagnosed_disease or "증상 분석이 필요합니다",
            confidence=confidence,
            department=department,
            symptoms=symptoms,
            recommendations=diagnosis_result,  # EXAONE 전체 응답
            reasoning=f"제공하신 증상 '{original_message}'을 AI가 분석한 결과입니다.",
            disclaimer=DISCLAIMER_TEXT,
            response_time=response_time,
            timestamp=datetime.now().isoformat()
        )
    
    def _build_info_response(self, info_result: str, disease_name: str, start_time: datetime) -> DiseaseResponse:
        """정보 제공 응답 구성"""
        
        department = self._determine_department(disease_name)
        response_time = (datetime.now() - start_time).total_seconds()
        
        return DiseaseResponse(
            diagnosis=disease_name,
            confidence=1.0,  # 정보 제공은 확신도 최대
            department=department,
            symptoms=[],
            recommendations=info_result,
            reasoning=f"'{disease_name}'에 대한 의료 정보입니다.",
            disclaimer=DISCLAIMER_TEXT,
            response_time=response_time,
            timestamp=datetime.now().isoformat()
        )
    
    def _build_error_response(self, error_message: str, start_time: datetime) -> DiseaseResponse:
        """오류 응답 구성"""
        response_time = (datetime.now() - start_time).total_seconds()
        
        return DiseaseResponse(
            diagnosis="처리 중 오류 발생",
            confidence=0.0,
            department=DEFAULT_DEPARTMENT,
            symptoms=[],
            recommendations=f"죄송합니다. {error_message} 다시 시도해주세요.",
            reasoning="시스템 오류로 인해 정상적인 분석이 어렵습니다.",
            disclaimer=DISCLAIMER_TEXT,
            response_time=response_time,
            timestamp=datetime.now().isoformat()
        )
    
    def _extract_disease_from_diagnosis(self, diagnosis_text: str) -> Optional[str]:
        """진단 결과에서 질병명 추출 (신경계 질환 포함 강화)"""
        if not diagnosis_text:
            return None
        
        # 패턴 1: "예상 질병: 질병명" 형태
        primary_patterns = [
            r'예상 질병[:\s*]+\*?\*?([가-힣\s\w\(\)]+?)(?:\*?\*?|\n|$|,)',
            r'진단[:\s*]+\*?\*?([가-힣\s\w\(\)]+?)(?:\*?\*?|\n|$|,)',
            r'질병[:\s*]+\*?\*?([가-힣\s\w\(\)]+?)(?:\*?\*?|\n|$|,)',
            r'1\.\s*\*?\*?예상 질병\*?\*?[:\s*]+([가-힣\s\w\(\)]+?)(?:\*?\*?|\n|$|,)',
        ]
        
        for pattern in primary_patterns:
            match = re.search(pattern, diagnosis_text, re.IGNORECASE)
            if match:
                disease = match.group(1).strip()
                # 괄호 안 내용 포함하여 정리
                disease = re.sub(r'\s+', ' ', disease).strip()
                disease = re.sub(r'(이|가|은|는|을|를|의|에|로|으로|과|와)$', '', disease)
                
                # 최소 길이 체크
                if len(disease) > 2:
                    logger.debug(f"1차 패턴에서 추출된 질병명: '{disease}'")
                    return disease
        
        # 패턴 2: 복합 질병명 추출 (다발성 신경병증, 바이러스성 감염 등)
        complex_patterns = [
            r'(다발성\s*신경병증)',
            r'(말초\s*신경\s*손상)',
            r'(신경\s*압박\s*증후군)',
            r'(수근관\s*증후군)',
            r'(바이러스성\s*감염)',
            r'(상부\s*호흡기\s*감염)',
            r'(급성\s*위장염)',
            r'(과민성\s*대장\s*증후군)',
            r'(역류성\s*식도염)',
            r'(긴장성\s*두통)',
            r'(편측성\s*두통)',
        ]
        
        for pattern in complex_patterns:
            match = re.search(pattern, diagnosis_text, re.IGNORECASE)
            if match:
                disease = match.group(1).strip()
                disease = re.sub(r'\s+', ' ', disease)
                logger.debug(f"복합 패턴에서 추출된 질병명: '{disease}'")
                return disease
        
        # 패턴 3: 일반적인 질병명 직접 매칭 (확장)
        common_diseases = [
            # 기존 질병들
            '감기', '독감', '몸살', '두통', '편두통', '발열', '인후염', '목감기',
            '복통', '위염', '장염', '설사', '구토', '변비', '소화불량',
            '기침', '가래', '콧물', '비염', '축농증', '중이염',
            '관절염', '근육통', '허리통증', '목통증', '어깨통증',
            '피부염', '습진', '두드러기', '아토피',
            
            # 신경계 질환 추가
            '다발성신경병증', '신경병증', '신경염', '신경통',
            '말초신경손상', '수근관증후군', '손목터널증후군',
            '신경압박증후군', '척골신경마비', '요골신경마비',
            '안면신경마비', '삼차신경통', '좌골신경통',
            
            # 기타 추가
            '혈관질환', '순환장애', '당뇨병성신경병증',
            '갑상선기능저하증', '비타민결핍', '자가면역질환',
            '코로나19', '코로나', 'COVID-19'
        ]
        
        diagnosis_lower = diagnosis_text.lower()
        for disease in common_diseases:
            if disease in diagnosis_lower:
                logger.debug(f"직접 매칭된 질병명: '{disease}'")
                return disease
        
        # 패턴 4: 문맥상 추론
        context_mapping = {
            '감각': '말초신경병증',
            '저림': '신경병증', 
            '무감각': '신경손상',
            '마비': '신경마비',
            '바이러스': '감기',
            '호흡기': '감기',
            '소화': '위장염',
            '혈압': '고혈압',
            '혈당': '당뇨병'
        }
        
        for keyword, disease in context_mapping.items():
            if keyword in diagnosis_lower:
                logger.debug(f"문맥 추론된 질병명: '{disease}' (키워드: {keyword})")
                return disease
        
        logger.warning(f"질병명 추출 실패: '{diagnosis_text[:200]}...'")
        return "증상 분석"
    
    
    def _extract_symptoms_from_text(self, text: str) -> List[str]:
        """텍스트에서 증상 키워드 추출 (신경계 증상 포함 강화)"""
        if not text:
            return []
        
        # 증상 키워드 매핑 (확장)
        symptom_patterns = {
            # 기존 증상들
            "두통": ["머리.*아프", "두통", "머리.*통증"],
            "발열": ["열", "발열", "고열", "체온.*높"],
            "기침": ["기침"],
            "가래": ["가래", "담"],
            "콧물": ["콧물", "코.*나오"],
            "목아픔": ["목.*아프", "인후통", "목.*통증", "목.*쓰림"],
            "복통": ["배.*아프", "복통", "배.*통증", "속.*쓰림"],
            "설사": ["설사", "묽은.*변"],
            "구토": ["구토", "토", "메스꺼"],
            "어지러움": ["어지럽", "현기증", "돌고"],
            "피로": ["피로", "무기력", "기운.*없"],
            "근육통": ["근육.*아프", "근육통", "몸살"],
            "관절통": ["관절.*아프", "관절통"],
            
            # 신경계 증상 추가
            "감각저하": ["감각.*없", "감각.*떨어", "느낌.*없", "무감각"],
            "저림": ["저리", "저림", "찌릿", "전기.*느낌"],
            "마비": ["마비", "움직.*않", "힘.*없", "못.*움직"],
            "손가락저림": ["손가락.*저리", "손가락.*감각", "손가락.*없"],
            "발저림": ["발.*저리", "발가락.*저리", "발.*감각"],
            "손목통증": ["손목.*아프", "손목.*통증"],
            "신경통": ["신경.*아프", "신경통", "찌르듯", "날카롭게"],
            
            # 기타 추가
            "시야장애": ["보기.*어렵", "시야.*흐림", "눈.*안.*보"],
            "청력저하": ["들리지.*않", "귀.*안.*들"],
            "언어장애": ["말.*안.*나", "발음.*어렵"],
            "균형장애": ["비틀", "넘어질.*것.*같", "중심.*잡기.*어렵"]
        }
        
        found_symptoms = []
        text_lower = text.lower()
        
        for symptom, patterns in symptom_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    found_symptoms.append(symptom)
                    break
        
        # 중복 제거
        unique_symptoms = list(set(found_symptoms))
        
        if unique_symptoms:
            logger.debug(f"추출된 증상: {unique_symptoms}")
        
        return unique_symptoms
    
    def _determine_department(self, disease_name: str) -> str:
        """질병명을 바탕으로 진료과 결정 (신경과 포함 강화)"""
        if not disease_name:
            return DEFAULT_DEPARTMENT
        
        disease_lower = disease_name.lower()
        
        # 정확한 진료과 매핑 (확장)
        department_rules = {
            # 신경과 (새로 추가)
            "신경과": [
                "다발성신경병증", "신경병증", "신경염", "신경통", "신경손상", "신경마비",
                "말초신경", "수근관증후군", "손목터널증후군", "신경압박",
                "척골신경", "요골신경", "안면신경", "삼차신경", "좌골신경",
                "두통", "편두통", "어지러움", "현기증", "머리아픔", "뇌",
                "마비", "경련", "치매", "파킨슨", "뇌전증", "간질",
                "감각", "무감각", "저림", "마비감"
            ],
            
            # 내과
            "내과": [
                "감기", "독감", "몸살", "발열", "기침", "가래", "콧물",
                "복통", "위염", "장염", "설사", "구토", "변비", "소화불량",
                "당뇨", "고혈압", "고지혈증", "갑상선", "간염",
                "바이러스", "세균", "감염", "호흡기", "코로나", "covid"
            ],
            
            # 정형외과
            "정형외과": [
                "관절염", "근육통", "허리통증", "목통증", "어깨통증", 
                "무릎통증", "발목통증", "골절", "탈구", "디스크",
                "척추", "관절", "뼈", "인대", "건염"
            ],
            
            # 이비인후과
            "이비인후과": [
                "인후염", "목감기", "편도염", "중이염", "비염", "축농증",
                "코막힘", "재채기", "귀아픔", "목아픔", "후두염", "성대"
            ],
            
            # 피부과
            "피부과": [
                "피부염", "습진", "두드러기", "아토피", "여드름", "무좀",
                "건선", "알레르기", "발진", "가려움", "피부"
            ],
            
            # 내분비내과
            "내분비내과": [
                "당뇨병성신경병증", "갑상선기능저하증", "갑상선기능항진증",
                "당뇨합병증", "대사증후군"
            ],
            
            # 류마티스내과
            "류마티스내과": [
                "자가면역질환", "류마티스", "전신홍반루푸스", "강직성척추염"
            ],
            
            # 안과
            "안과": [
                "결막염", "다래끼", "백내장", "녹내장", "눈아픔", "시력"
            ],
            
            # 비뇨기과
            "비뇨기과": [
                "방광염", "요로감염", "전립선", "신장", "소변"
            ],
            
            # 산부인과
            "산부인과": [
                "생리통", "생리불순", "질염", "자궁", "난소"
            ]
        }
        
        # 질병명에서 진료과 찾기
        for department, diseases in department_rules.items():
            for disease_keyword in diseases:
                if disease_keyword in disease_lower:
                    logger.debug(f"진료과 매핑: '{disease_name}' -> '{department}' (키워드: {disease_keyword})")
                    return department
        
        # 매핑되지 않으면 기본값
        logger.debug(f"기본 진료과 적용: '{disease_name}' -> '{DEFAULT_DEPARTMENT}'")
        return DEFAULT_DEPARTMENT

    
    def _calculate_confidence(self, diagnosis_text: str) -> float:
        """진단 결과의 확신도 계산 (개선된 버전)"""
        if not diagnosis_text:
            return 0.5
        
        text_lower = diagnosis_text.lower()
        
        # 매우 높은 확신도 (0.9)
        very_high_indicators = [
            "확실", "명확", "분명", "명백", "전형적", "특징적"
        ]
        
        # 높은 확신도 (0.8)
        high_confidence_indicators = [
            "가능성이 높", "강하게 시사", "유력한 후보", "가장 유력한"
        ]
        
        # 중상 확신도 (0.7)
        medium_high_indicators = [
            "예상됩니다", "보입니다", "것으로 판단", "추정됩니다", "시사합니다"
        ]
        
        # 중간 확신도 (0.6)
        medium_indicators = [
            "가능성", "의심", "생각됩니다", "것으로 보임"
        ]
        
        # 낮은 확신도 (0.4)
        low_confidence_indicators = [
            "불확실", "추가 검사", "정확한 진단", "감별 진단", 
            "병원 방문", "의료진 상담", "전문의 진료"
        ]
        
        # 확신도 계산
        for indicator in very_high_indicators:
            if indicator in text_lower:
                return 0.9
        
        for indicator in high_confidence_indicators:
            if indicator in text_lower:
                return 0.8
        
        for indicator in medium_high_indicators:
            if indicator in text_lower:
                return 0.7
        
        for indicator in medium_indicators:
            if indicator in text_lower:
                return 0.6
        
        for indicator in low_confidence_indicators:
            if indicator in text_lower:
                return 0.4
        
        # 텍스트 길이와 상세도 기반 보정
        if len(diagnosis_text) > 500:  # 상세한 설명
            return 0.7
        elif len(diagnosis_text) > 200:  # 적당한 설명
            return 0.6
        else:  # 짧은 설명
            return 0.5
    
    def get_service_status(self) -> Dict:
        """서비스 상태 반환"""
        status = {
            "is_initialized": self.is_initialized,
            "embedding_service_loaded": self.embedding_service.is_loaded,
            "faiss_loader_loaded": self.faiss_loader.is_loaded,
            "rag_service_initialized": self.rag_service.is_initialized,
            "exaone_connected": self.exaone_endpoint is not None,
            "timestamp": datetime.now().isoformat()
        }
        
        if self.is_initialized:
            status.update({
                "disease_key_vectors": self.disease_key_index.ntotal,
                "disease_full_vectors": self.disease_full_index.ntotal,
                "disease_metadata_count": len(self.disease_metadata),
                "exaone_endpoint": self.exaone_endpoint
            })
        
        return status


# 전역 질병 서비스 인스턴스 (싱글톤 패턴)
_global_disease_service: Optional[DiseaseService] = None


def get_disease_service() -> DiseaseService:
    """질병 서비스 싱글톤 인스턴스 반환"""
    global _global_disease_service
    
    if _global_disease_service is None:
        _global_disease_service = DiseaseService()
    
    return _global_disease_service


def initialize_disease_service() -> bool:
    """질병 서비스 초기화"""
    try:
        service = get_disease_service()
        return service.initialize()
    except Exception as e:
        logger.error(f"❌ 질병 서비스 초기화 실패: {e}")
        raise DiseaseDiagnosisError(f"질병 서비스 초기화 실패: {e}")