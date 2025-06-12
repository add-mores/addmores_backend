"""
질병 서비스 - CLI의 EnhancedDiseaseService 완전 동일 로직
위치: backend/app/llm/services/disease_service.py

🎯 목적: 증상 기반 질병 진단 및 스마트한 차별화 질문
📋 기능: CLI의 모든 질병 진단 로직 100% 보존 + EXAONE 통합
"""

import os
import re
import logging
import pandas as pd
import numpy as np
import faiss
import requests
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass
from datetime import datetime

# 내부 모듈 imports
from app.llm.services.session_manager import IntegratedSession
from app.llm.services.embedding_service import EmbeddingModel, RAGIndexManager

# 로깅 설정
logger = logging.getLogger(__name__)

@dataclass
class DiseaseCandidate:
    """질병 후보 정보"""
    disease_name: str
    similarity_score: float
    matched_symptoms: List[str]
    additional_info: Dict[str, Any]

class EXAONE:
    """
    EXAONE 모델 통신 클래스 (CLI와 완전 동일)
    
    🧠 모델: EXAONE 3.5:7.8b
    🌐 연결: Ollama API 서버
    """
    
    def __init__(self, model_name: str = "exaone3.5:7.8b"):
        """EXAONE 모델 초기화"""
        self.model_name = model_name
        self.base_url = "http://localhost:11434"  # Ollama 기본 포트
        self.endpoint = None
        
        # EXAONE 설정 (CLI와 동일)
        self.exaone_config = {
            "temperature": 0.1,      # 낮은 창의성, 일관성 중시
            "top_p": 0.8,           # 토큰 선택 범위 제한
            "num_predict": 1000,    # 최대 토큰 수
            "stop": ["사용자:", "User:", "질문:", "답변:"]
        }
        
        # 연결 확인 및 엔드포인트 결정
        self._detect_endpoint()
        
        logger.info(f"🧠 EXAONE 모델 초기화: {model_name} | 엔드포인트: {self.endpoint}")
    
    def _detect_endpoint(self):
        """사용 가능한 엔드포인트 감지"""
        # Chat API 먼저 시도 (권장)
        if self._check_endpoint("chat"):
            self.endpoint = "chat"
        # Generate API 시도
        elif self._check_endpoint("generate"):
            self.endpoint = "generate"
        else:
            self.endpoint = None
            logger.warning("⚠️ EXAONE 서버에 연결할 수 없습니다. 기본 응답 모드로 실행합니다.")
    
    def _check_endpoint(self, endpoint: str) -> bool:
        """엔드포인트 연결 확인"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def generate_response(self, prompt: str, system_prompt: str = "") -> str:
        """EXAONE 모델 응답 생성"""
        if not self.endpoint:
            return "⚠️ EXAONE 서버에 연결할 수 없습니다. Ollama가 실행 중인지 확인해주세요."
        
        try:
            if self.endpoint == "chat":
                return self._chat_request(prompt, system_prompt)
            else:
                return self._generate_request(prompt, system_prompt)
        except Exception as e:
            logger.error(f"EXAONE 응답 생성 오류: {e}")
            return f"⚠️ EXAONE 응답 생성 중 오류가 발생했습니다: {str(e)}"
    
    def _chat_request(self, prompt: str, system_prompt: str) -> str:
        """Chat API 요청"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                **self.exaone_config
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()["message"]["content"]
        else:
            return f"⚠️ API 요청 실패: {response.status_code}"
    
    def _generate_request(self, prompt: str, system_prompt: str) -> str:
        """Generate API 요청"""
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": full_prompt,
                "stream": False,
                **self.exaone_config
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return f"⚠️ API 요청 실패: {response.status_code}"

class EnhancedDiseaseService:
    """
    향상된 질병 진단 서비스 (CLI와 완전 동일)
    
    🔍 주요 기능:
    - 증상 기반 질병 예측
    - 스마트한 차별화 질문 (중복 방지)
    - EXAONE 기반 자연어 응답 생성
    - RAG 기반 질병 정보 검색
    """
    
    def __init__(self, embedding_model: EmbeddingModel, rag_manager: RAGIndexManager):
        """질병 서비스 초기화"""
        self.embedding_model = embedding_model
        self.rag_manager = rag_manager
        self.exaone = EXAONE()
        
        # 질병 데이터 로드
        self.disease_data = None
        self.disease_index = None
        self._load_disease_data()
        
        logger.info("🏥 질병 진단 서비스 초기화 완료")
    
    def _load_disease_data(self):
        """질병 데이터 로드 (CSV 파일)"""
        try:
            disease_file = "app/integration_test/disease_prototype.csv"
            if os.path.exists(disease_file):
                self.disease_data = pd.read_csv(disease_file)
                logger.info(f"📊 질병 데이터 로드: {len(self.disease_data)}개 질병")
                
                # 임베딩 인덱스 구축
                self._build_disease_index()
            else:
                logger.warning(f"⚠️ 질병 데이터 파일이 없습니다: {disease_file}")
                
        except Exception as e:
            logger.error(f"❌ 질병 데이터 로드 실패: {e}")
    
    def _build_disease_index(self):
        """질병 임베딩 인덱스 구축"""
        if self.disease_data is None:
            return
        
        try:
            # 질병명과 증상을 결합한 텍스트 생성
            disease_texts = []
            for _, row in self.disease_data.iterrows():
                disease_name = str(row.get('disease_name', ''))
                symptoms = str(row.get('symptoms', ''))
                combined_text = f"{disease_name} {symptoms}"
                disease_texts.append(combined_text)
            
            # 임베딩 생성
            logger.info("🔍 질병 임베딩 생성 중...")
            embeddings = self.embedding_model.encode(disease_texts)
            
            # FAISS 인덱스 구축
            self.disease_index = faiss.IndexFlatIP(self.embedding_model.embedding_dim)
            self.disease_index.add(embeddings.astype('float32'))
            
            logger.info(f"✅ 질병 인덱스 구축 완료: {self.disease_index.ntotal}개")
            
        except Exception as e:
            logger.error(f"❌ 질병 인덱스 구축 실패: {e}")
    
    def process_disease_diagnosis(self, message: str, session: IntegratedSession) -> str:
        """
        질병 진단 처리 (CLI와 완전 동일한 로직)
        
        Args:
            message: 사용자 메시지
            session: 세션 객체
            
        Returns:
            str: 진단 응답
        """
        logger.info(f"🔍 질병 진단 처리 시작: {message[:50]}...")
        
        # 질문 모드 상태 확인
        questioning_state = session.context.get("questioning_state", {})
        is_questioning = questioning_state.get("is_questioning", False)
        
        if is_questioning:
            # 📝 질문에 대한 답변 처리
            return self._handle_questioning_response(message, session)
        else:
            # 🔍 초기 증상 분석
            return self._handle_initial_symptoms(message, session)
    
    def _handle_initial_symptoms(self, message: str, session: IntegratedSession) -> str:
        """초기 증상 분석 처리"""
        # 초기 증상 저장
        session.set_initial_symptoms(message)
        
        # 증상 추출
        symptoms = self._extract_symptoms(message)
        for symptom in symptoms:
            session.add_symptom(symptom)
        
        # 질병 예측
        disease_candidates = self._predict_diseases(message)
        
        if disease_candidates:
            # 가장 가능성 높은 질병
            top_disease = disease_candidates[0]
            session.set_diagnosis(top_disease.disease_name)
            
            # 차별화 질문 생성
            question = self._generate_differential_question(top_disease, session)
            
            if question:
                session.start_questioning(question)
                
                # EXAONE을 통한 자연어 응답 생성
                return self._generate_initial_diagnosis_response(
                    top_disease, 
                    disease_candidates[:3], 
                    question,
                    session
                )
            else:
                # 차별화 질문이 없으면 최종 진단
                session.stop_questioning()
                return self._generate_final_diagnosis_response(top_disease, session)
        else:
            return "증상을 분석했지만 명확한 질병을 특정하기 어렵습니다. 더 구체적인 증상을 알려주세요."
    
    def _handle_questioning_response(self, message: str, session: IntegratedSession) -> str:
        """질문에 대한 답변 처리"""
        # 추가 증상 추출 및 저장
        additional_symptoms = self._extract_symptoms(message)
        for symptom in additional_symptoms:
            session.add_symptom(symptom)
        
        # 응답 분석 (긍정/부정)
        is_positive = self._analyze_response_sentiment(message)
        
        # 현재 질문 컨텍스트에 따른 증상 처리
        current_question = session.context["questioning_state"].get("current_question", "")
        if is_positive and current_question:
            # 질문에서 언급된 증상을 추가
            question_symptoms = self._extract_symptoms_from_question(current_question)
            for symptom in question_symptoms:
                session.add_symptom(symptom)
        
        # 업데이트된 증상으로 재진단
        all_symptoms = " ".join(session.context.get("symptoms", []))
        updated_candidates = self._predict_diseases(all_symptoms)
        
        if updated_candidates:
            top_disease = updated_candidates[0]
            session.set_diagnosis(top_disease.disease_name)
            
            # 추가 차별화 질문 확인
            next_question = self._generate_differential_question(top_disease, session)
            
            if next_question and session.context["questioning_state"]["question_count"] < 3:
                # 추가 질문 계속
                session.start_questioning(next_question)
                return self._generate_followup_question_response(
                    top_disease, 
                    updated_candidates[:3], 
                    next_question,
                    session
                )
            else:
                # 질문 종료, 최종 진단
                session.stop_questioning()
                return self._generate_final_diagnosis_response(top_disease, session)
        else:
            session.stop_questioning()
            return "추가 정보를 바탕으로 분석했지만 명확한 진단을 내리기 어렵습니다. 의료진과 상담하시기 바랍니다."
    
    def _extract_symptoms(self, text: str) -> List[str]:
        """텍스트에서 증상 추출 (CLI와 동일)"""
        symptoms = []
        
        # 증상 키워드 패턴들
        symptom_patterns = [
            (r"([가-힣]+)\s*(?:아|픈|아픈|아파|통증|쑤시|따끔)", "통증"),
            (r"(열|발열|체온).*(?:나|남|있|해|올라)", "발열"),
            (r"(기침|가래|콧물|재채기)", "호흡기"),
            (r"(두통|머리.*아|어지럼|현기증)", "신경계"),
            (r"(복통|배.*아|설사|변비|구토|메스꺼움)", "소화기"),
            (r"(피로|힘들|지침|무기력)", "전신"),
            (r"(가려|발진|부종|붓)", "피부"),
            (r"(숨.*차|호흡.*어려|답답)", "호흡")
        ]
        
        for pattern, category in symptom_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    symptom = match[0] if match[0] else match[1]
                else:
                    symptom = match
                
                if symptom and len(symptom) > 1:
                    symptoms.append(symptom)
        
        return list(set(symptoms))  # 중복 제거
    
    def _predict_diseases(self, symptoms_text: str, top_k: int = 5) -> List[DiseaseCandidate]:
        """증상 기반 질병 예측"""
        if not self.disease_index or self.disease_data is None:
            return []
        
        try:
            # 증상 임베딩 생성
            query_embedding = self.embedding_model.encode_single(symptoms_text)
            
            # FAISS 검색
            scores, indices = self.disease_index.search(
                query_embedding.reshape(1, -1).astype('float32'), 
                top_k
            )
            
            # 후보 질병 구성
            candidates = []
            for score, idx in zip(scores[0], indices[0]):
                if 0 <= idx < len(self.disease_data):
                    row = self.disease_data.iloc[idx]
                    
                    candidate = DiseaseCandidate(
                        disease_name=str(row.get('disease_name', '')),
                        similarity_score=float(score),
                        matched_symptoms=self._extract_symptoms(symptoms_text),
                        additional_info={
                            'symptoms': str(row.get('symptoms', '')),
                            'description': str(row.get('description', '')),
                            'source_idx': idx
                        }
                    )
                    
                    candidates.append(candidate)
            
            return candidates
            
        except Exception as e:
            logger.error(f"❌ 질병 예측 실패: {e}")
            return []
    
    def _generate_differential_question(self, disease: DiseaseCandidate, session: IntegratedSession) -> Optional[str]:
        """차별화 질문 생성 (중복 방지)"""
        mentioned_symptoms = set(s.lower() for s in session.context.get("mentioned_symptoms", []))
        
        # 질병별 차별화 질문 후보들 (CLI와 동일)
        differential_questions = {
            "감기": [
                "기침이나 가래 증상이 있으신가요?",
                "목이 아프거나 따가운 증상이 있나요?",
                "콧물이나 코막힘 증상이 있으신가요?",
                "몸살이나 근육통이 있으신가요?"
            ],
            "독감": [
                "갑작스럽게 고열이 났나요?",
                "심한 몸살이나 근육통이 있으신가요?",
                "극심한 피로감이나 무기력감이 있나요?",
                "오한이 심하게 드시나요?"
            ],
            "위염": [
                "식사 후 속이 쓰리거나 아픈가요?",
                "구토나 메스꺼움 증상이 있나요?",
                "트림이 자주 나오나요?",
                "특정 음식을 먹으면 증상이 악화되나요?"
            ],
            "두통": [
                "머리 어느 부위가 가장 아픈가요?",
                "맥박이 뛰는 것처럼 지끈거리나요?",
                "빛이나 소리에 민감해지셨나요?",
                "목이나 어깨 근육이 뻣뻣한가요?"
            ]
        }
        
        disease_name = disease.disease_name.lower()
        
        # 질병명에서 키워드 찾기
        for key, questions in differential_questions.items():
            if key in disease_name:
                # 아직 언급되지 않은 증상에 대한 질문 찾기
                for question in questions:
                    question_symptoms = self._extract_symptoms_from_question(question)
                    
                    # 질문의 증상이 이미 언급되었는지 확인
                    is_already_mentioned = any(
                        symptom.lower() in mentioned_symptoms 
                        for symptom in question_symptoms
                    )
                    
                    if not is_already_mentioned:
                        return question
        
        return None
    
    def _extract_symptoms_from_question(self, question: str) -> List[str]:
        """질문에서 증상 키워드 추출"""
        symptom_keywords = [
            "기침", "가래", "목", "아프", "따가", "콧물", "코막힘",
            "몸살", "근육통", "고열", "피로", "무기력", "오한",
            "속", "쓰리", "구토", "메스꺼움", "트림",
            "머리", "두통", "지끈", "빛", "소리", "민감", "뻣뻣"
        ]
        
        found_symptoms = []
        for keyword in symptom_keywords:
            if keyword in question:
                found_symptoms.append(keyword)
        
        return found_symptoms
    
    def _analyze_response_sentiment(self, response: str) -> bool:
        """응답의 긍정/부정 분석"""
        positive_keywords = ["네", "예", "있어요", "있습니다", "그래요", "맞아요", "심해요", "많이"]
        negative_keywords = ["아니", "없어요", "없습니다", "안", "별로", "그렇지"]
        
        response_lower = response.lower()
        
        positive_count = sum(1 for keyword in positive_keywords if keyword in response_lower)
        negative_count = sum(1 for keyword in negative_keywords if keyword in response_lower)
        
        return positive_count > negative_count
    
    def _generate_initial_diagnosis_response(self, top_disease: DiseaseCandidate, 
                                           candidates: List[DiseaseCandidate], 
                                           question: str, session: IntegratedSession) -> str:
        """초기 진단 응답 생성 (EXAONE 활용)"""
        symptoms = ", ".join(session.context.get("symptoms", []))
        
        prompt = f"""
환자의 증상: {symptoms}
가장 가능성 높은 질병: {top_disease.disease_name}
추가 질문: {question}

위 정보를 바탕으로 친근하고 전문적인 의료 상담 응답을 생성해주세요.
다음 구조로 작성해주세요:

1. 증상 분석 결과 요약
2. 예상 질병과 가능성
3. 추가 질문을 통한 정확한 진단 필요성 설명
4. 따뜻한 톤의 추가 질문

**중요**: 의료진과의 상담을 권하는 문구를 포함해주세요.
"""
        
        system_prompt = """당신은 친근하고 전문적인 의료 상담 AI입니다. 
환자에게 도움이 되는 정보를 제공하되, 의료진의 진단을 대체할 수 없음을 명확히 해주세요."""
        
        exaone_response = self.exaone.generate_response(prompt, system_prompt)
        
        if "⚠️" not in exaone_response:
            return f"{exaone_response}\n\n❓ **추가 질문**: {question}"
        else:
            # EXAONE 실패 시 기본 응답
            return f"""🔍 **초기 분석 결과**:
증상을 분석한 결과 **{top_disease.disease_name}**의 가능성이 높습니다.

📊 **가능한 질병들**:
{', '.join([c.disease_name for c in candidates[:3]])}

더 정확한 진단을 위해 추가 정보가 필요합니다.

❓ **{question}**

⚠️ 이 정보는 참고용이며, 정확한 진단을 위해서는 의료진과 상담하시기 바랍니다."""
    
    def _generate_followup_question_response(self, disease: DiseaseCandidate, 
                                           candidates: List[DiseaseCandidate], 
                                           question: str, session: IntegratedSession) -> str:
        """후속 질문 응답 생성"""
        symptoms = ", ".join(session.context.get("symptoms", []))
        question_count = session.context["questioning_state"]["question_count"]
        
        return f"""🔍 **진단 업데이트** (질문 {question_count}/3):
현재까지 파악된 증상: {symptoms}

가능성이 높은 질병: **{disease.disease_name}**

추가 확인이 필요한 사항이 있습니다.

❓ **{question}**"""
    
    def _generate_final_diagnosis_response(self, disease: DiseaseCandidate, session: IntegratedSession) -> str:
        """최종 진단 응답 생성 (EXAONE 활용)"""
        all_symptoms = ", ".join(session.context.get("symptoms", []))
        
        prompt = f"""
환자의 모든 증상: {all_symptoms}
최종 진단: {disease.disease_name}

위 정보를 바탕으로 최종 진단 결과를 친근하고 전문적으로 설명해주세요.
다음 내용을 포함해주세요:

1. 진단 결과 요약
2. 해당 질병의 일반적인 특징
3. 권장 사항 (휴식, 수분 섭취 등)
4. 의료진 상담 권유

**중요**: 확정 진단이 아닌 가능성에 대한 설명임을 명확히 해주세요.
"""
        
        system_prompt = """당신은 친근하고 전문적인 의료 상담 AI입니다. 
최종 진단 결과를 설명할 때는 확정이 아닌 가능성임을 강조하고, 의료진 상담을 권해주세요."""
        
        exaone_response = self.exaone.generate_response(prompt, system_prompt)
        
        if "⚠️" not in exaone_response:
            return exaone_response
        else:
            # EXAONE 실패 시 기본 응답
            return f"""✅ **최종 분석 결과**:
모든 증상을 종합한 결과 **{disease.disease_name}**의 가능성이 높습니다.

📋 **확인된 증상들**: {all_symptoms}

💡 **권장 사항**:
- 충분한 휴식과 수분 섭취
- 증상이 지속되거나 악화되면 의료진 상담
- 처방약 복용 시 용법 준수

⚠️ **중요**: 이는 증상 기반 분석 결과이며, 정확한 진단과 치료를 위해서는 반드시 의료진과 상담하시기 바랍니다."""
    
    def search_disease_info(self, query: str, session: IntegratedSession) -> str:
        """질병 정보 검색 (RAG 기반)"""
        logger.info(f"🔍 질병 정보 검색: {query}")
        
        # RAG 검색 수행
        search_results = self.rag_manager.search_combined(query, qa_top_k=3, medical_top_k=3)
        
        # 검색 결과 구성
        response_parts = [f"🔍 **'{query}' 검색 결과**:\n"]
        
        # Q&A 결과
        qa_results = search_results.get("qa_results", [])
        if qa_results:
            response_parts.append("📋 **Q&A 정보**:")
            for i, (doc, score) in enumerate(qa_results[:2], 1):
                content = doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
                response_parts.append(f"{i}. {content}")
            response_parts.append("")
        
        # 의료 문서 결과
        medical_results = search_results.get("medical_results", [])
        if medical_results:
            response_parts.append("📚 **의료 문서 정보**:")
            for i, (doc, score) in enumerate(medical_results[:2], 1):
                content = doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
                response_parts.append(f"{i}. {content}")
        
        if not qa_results and not medical_results:
            response_parts.append("검색된 정보가 없습니다. 다른 키워드로 시도해보세요.")
        
        response_parts.append("\n⚠️ 이 정보는 참고용이며, 정확한 진단을 위해서는 의료진과 상담하시기 바랍니다.")
        
        return "\n".join(response_parts)