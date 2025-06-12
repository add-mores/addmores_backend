"""
의약품 서비스 - CLI의 MedicationService 완전 동일 로직
위치: backend/app/llm/services/medication_service.py

🎯 목적: 의약품 추천 및 정보 제공 (exaone_medi.txt 로직 100% 보존)
📋 기능: 증상/질병 기반 의약품 추천, 의약품 정보 검색, 안전성 필터링
"""

import os
import re
import logging
import pandas as pd
import numpy as np
import faiss
import requests
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime

# 내부 모듈 imports
from app.llm.services.session_manager import IntegratedSession
from app.llm.services.embedding_service import EmbeddingModel, RAGIndexManager

# 로깅 설정
logger = logging.getLogger(__name__)

@dataclass
class MedicationCandidate:
    """의약품 후보 정보"""
    medicine_name: str
    similarity_score: float
    effect: str
    usage: str
    precautions: str
    is_safe_for_user: bool
    safety_notes: List[str]

class MedicationService:
    """
    의약품 서비스 (CLI exaone_medi.txt 로직 완전 동일)
    
    🔍 주요 기능:
    - 증상/질병 기반 의약품 추천
    - 의약품 정보 검색
    - 사용자 안전성 필터링 (임신, 연령, 알레르기 등)
    - 복용법 및 주의사항 제공
    """
    
    def __init__(self, embedding_model: EmbeddingModel, rag_manager: RAGIndexManager):
        """의약품 서비스 초기화"""
        self.embedding_model = embedding_model
        self.rag_manager = rag_manager
        
        # EXAONE 모델 (질병 서비스와 동일한 설정)
        self.exaone = self._initialize_exaone()
        
        # 의약품 데이터 로드
        self.medication_data = None
        self.medication_index = None
        self._load_medication_data()
        
        # 안전성 필터링 규칙
        self._setup_safety_rules()
        
        logger.info("💊 의약품 서비스 초기화 완료")
    
    def _initialize_exaone(self):
        """EXAONE 모델 초기화 (질병 서비스와 동일)"""
        try:
            # 동일한 EXAONE 클래스 사용하기 위해 import
            from app.llm.services.disease_service import EXAONE
            return EXAONE()
        except Exception as e:
            logger.error(f"❌ EXAONE 초기화 실패: {e}")
            return None
    
    def _load_medication_data(self):
        """의약품 데이터 로드 (CLI와 동일)"""
        try:
            medication_file = "app/integration_test/medicine_code_merged.csv"
            if os.path.exists(medication_file):
                self.medication_data = pd.read_csv(medication_file)
                logger.info(f"📊 의약품 데이터 로드: {len(self.medication_data)}개 의약품")
                
                # 데이터 전처리
                self._preprocess_medication_data()
                
                # 임베딩 인덱스 구축
                self._build_medication_index()
            else:
                logger.warning(f"⚠️ 의약품 데이터 파일이 없습니다: {medication_file}")
                
        except Exception as e:
            logger.error(f"❌ 의약품 데이터 로드 실패: {e}")
    
    def _preprocess_medication_data(self):
        """의약품 데이터 전처리"""
        if self.medication_data is None:
            return
        
        # 결측값 처리
        text_columns = ['medicine_name', 'effect', 'usage', 'precautions']
        for col in text_columns:
            if col in self.medication_data.columns:
                self.medication_data[col] = self.medication_data[col].fillna('')
        
        # 텍스트 정규화
        for col in text_columns:
            if col in self.medication_data.columns:
                self.medication_data[col] = self.medication_data[col].astype(str)
        
        logger.info("✅ 의약품 데이터 전처리 완료")
    
    def _build_medication_index(self):
        """의약품 임베딩 인덱스 구축"""
        if self.medication_data is None:
            return
        
        try:
            # 의약품명, 효능, 용법을 결합한 텍스트 생성
            medication_texts = []
            for _, row in self.medication_data.iterrows():
                medicine_name = str(row.get('medicine_name', ''))
                effect = str(row.get('effect', ''))
                usage = str(row.get('usage', ''))
                
                # 검색에 최적화된 텍스트 구성
                combined_text = f"{medicine_name} {effect} {usage}"
                medication_texts.append(combined_text)
            
            # 임베딩 생성
            logger.info("🔍 의약품 임베딩 생성 중...")
            embeddings = self.embedding_model.encode(medication_texts)
            
            # FAISS 인덱스 구축
            self.medication_index = faiss.IndexFlatIP(self.embedding_model.embedding_dim)
            self.medication_index.add(embeddings.astype('float32'))
            
            logger.info(f"✅ 의약품 인덱스 구축 완료: {self.medication_index.ntotal}개")
            
        except Exception as e:
            logger.error(f"❌ 의약품 인덱스 구축 실패: {e}")
    
    def _setup_safety_rules(self):
        """안전성 필터링 규칙 설정 (CLI와 동일)"""
        self.safety_rules = {
            "pregnancy": {
                "prohibited": [
                    "아스피린", "이부프로펜", "낙센", "부루펜", "애드빌",
                    "와파린", "헤파린", "디클로페낙"
                ],
                "message": "임신 중에는 사용하지 마세요"
            },
            "pediatric": {
                "prohibited": [
                    "아스피린", "테트라사이클린", "독시사이클린"
                ],
                "age_specific": {
                    "under_6": ["이부프로펜", "낙센"],
                    "under_12": ["아스피린"]
                },
                "message": "소아에게는 적합하지 않을 수 있습니다"
            },
            "elderly": {
                "caution": [
                    "수면제", "신경안정제", "항히스타민제"
                ],
                "message": "고령자는 주의해서 사용하세요"
            }
        }
        
        logger.info("🛡️ 안전성 필터링 규칙 설정 완료")
    
    def process_medication_query(self, message: str, session: IntegratedSession) -> str:
        """
        의약품 쿼리 처리 (CLI와 완전 동일한 로직)
        
        Args:
            message: 사용자 메시지
            session: 세션 객체
            
        Returns:
            str: 의약품 추천 응답
        """
        logger.info(f"💊 의약품 쿼리 처리: {message[:50]}...")
        
        # 증상 추출
        symptoms = self._extract_symptoms_from_query(message)
        
        # 의약품 검색
        medication_candidates = self._search_medications(message)
        
        if medication_candidates:
            # 사용자 안전성 필터링
            safe_medications = self._filter_by_safety(medication_candidates, session)
            
            if safe_medications:
                # 추천 응답 생성
                return self._generate_medication_recommendation(safe_medications, symptoms, session)
            else:
                return self._generate_safety_warning_response(medication_candidates, session)
        else:
            return "죄송합니다. 해당 증상에 적합한 의약품을 찾을 수 없습니다. 약사나 의료진과 상담해주세요."
    
    def recommend_by_disease(self, disease_name: str, session: IntegratedSession) -> str:
        """
        질병 기반 의약품 추천 (새 기능 - 질병-의약품 연계)
        
        Args:
            disease_name: 진단된 질병명
            session: 세션 객체
            
        Returns:
            str: 추천 응답
        """
        logger.info(f"🏥➡️💊 질병 기반 의약품 추천: {disease_name}")
        
        # 질병별 증상 매핑
        disease_symptom_map = {
            "감기": "발열 기침 콧물 목아픔 두통",
            "독감": "고열 근육통 오한 피로 두통",
            "두통": "머리아픔 편두통",
            "위염": "속쓰림 복통 메스꺼움 소화불량",
            "변비": "배변곤란 복부팽만",
            "설사": "묽은변 복통",
            "알레르기": "가려움 발진 재채기 콧물"
        }
        
        # 질병에 해당하는 증상으로 의약품 검색
        symptoms_query = disease_symptom_map.get(disease_name.lower(), disease_name)
        medication_candidates = self._search_medications(symptoms_query)
        
        if medication_candidates:
            # 안전성 필터링
            safe_medications = self._filter_by_safety(medication_candidates, session)
            
            if safe_medications:
                response = f"🏥 **{disease_name}** 치료를 위한 의약품 추천:\n\n"
                response += self._generate_medication_recommendation(safe_medications, [disease_name], session)
                return response
            else:
                return f"{disease_name} 치료를 위한 의약품이 있지만, 현재 사용자 상태로는 안전하지 않을 수 있습니다. 의료진과 상담하세요."
        else:
            return f"{disease_name}에 대한 구체적인 의약품 정보를 찾을 수 없습니다. 의료진의 처방을 받으시기 바랍니다."
    
    def search_medication_info(self, query: str, session: IntegratedSession) -> str:
        """의약품 정보 검색"""
        logger.info(f"💊🔍 의약품 정보 검색: {query}")
        
        # 의약품명 추출
        medication_name = self._extract_medication_name(query)
        
        if medication_name:
            # 특정 의약품 정보 검색
            return self._get_specific_medication_info(medication_name, session)
        else:
            # 일반 검색
            search_results = self._search_medications(query)
            if search_results:
                return self._generate_medication_info_response(search_results[:3], session)
            else:
                return "해당 의약품 정보를 찾을 수 없습니다. 정확한 의약품명을 입력해주세요."
    
    def _extract_symptoms_from_query(self, query: str) -> List[str]:
        """쿼리에서 증상 추출"""
        symptom_keywords = [
            "발열", "열", "기침", "콧물", "코막힘", "재채기",
            "두통", "머리아픔", "어지러움", "현기증",
            "복통", "배아픔", "설사", "변비", "구토", "메스꺼움",
            "목아픔", "인후통", "가래", "근육통", "몸살",
            "피로", "무기력", "오한", "속쓰림", "소화불량",
            "가려움", "발진", "알레르기", "부종", "붓기"
        ]
        
        found_symptoms = []
        query_lower = query.lower()
        
        for symptom in symptom_keywords:
            if symptom in query_lower:
                found_symptoms.append(symptom)
        
        return found_symptoms
    
    def _extract_medication_name(self, query: str) -> str:
        """쿼리에서 의약품명 추출"""
        # 의약품명 패턴들
        patterns = [
            r'([가-힣A-Za-z0-9]+(?:정|캡슐|시럽|연고|크림))',
            r'(타이레놀|게보린|낙센|이부프로펜|아스피린|애드빌|부루펜)',
            r'([가-힣A-Za-z0-9]+)(?:\s*(?:약|의약품|약품))'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                return match.group(1)
        
        return ""
    
    def _search_medications(self, query: str, top_k: int = 10) -> List[MedicationCandidate]:
        """의약품 검색"""
        if not self.medication_index or self.medication_data is None:
            return []
        
        try:
            # 쿼리 임베딩 생성
            query_embedding = self.embedding_model.encode_single(query)
            
            # FAISS 검색
            scores, indices = self.medication_index.search(
                query_embedding.reshape(1, -1).astype('float32'), 
                top_k
            )
            
            # 후보 의약품 구성
            candidates = []
            for score, idx in zip(scores[0], indices[0]):
                if 0 <= idx < len(self.medication_data) and score > 0.3:  # 최소 유사도 임계값
                    row = self.medication_data.iloc[idx]
                    
                    candidate = MedicationCandidate(
                        medicine_name=str(row.get('medicine_name', '')),
                        similarity_score=float(score),
                        effect=str(row.get('effect', '')),
                        usage=str(row.get('usage', '')),
                        precautions=str(row.get('precautions', '')),
                        is_safe_for_user=True,  # 초기값, 나중에 필터링
                        safety_notes=[]
                    )
                    
                    candidates.append(candidate)
            
            return candidates
            
        except Exception as e:
            logger.error(f"❌ 의약품 검색 실패: {e}")
            return []
    
    def _filter_by_safety(self, candidates: List[MedicationCandidate], 
                         session: IntegratedSession) -> List[MedicationCandidate]:
        """사용자 안전성 기반 필터링"""
        user_profile = session.context.get("user_profile", {})
        
        safe_candidates = []
        
        for candidate in candidates:
            safety_notes = []
            is_safe = True
            
            # 임신 여부 확인
            if user_profile.get("is_pregnant", False):
                if any(prohibited in candidate.medicine_name.lower() 
                       for prohibited in self.safety_rules["pregnancy"]["prohibited"]):
                    is_safe = False
                    safety_notes.append(self.safety_rules["pregnancy"]["message"])
            
            # 연령 확인
            age_group = user_profile.get("age_group", "성인")
            if age_group == "소아":
                if any(prohibited in candidate.medicine_name.lower() 
                       for prohibited in self.safety_rules["pediatric"]["prohibited"]):
                    is_safe = False
                    safety_notes.append(self.safety_rules["pediatric"]["message"])
            
            # 알레르기 확인
            allergies = user_profile.get("allergies", [])
            for allergy in allergies:
                if allergy.lower() in candidate.medicine_name.lower():
                    is_safe = False
                    safety_notes.append(f"알레르기 성분 ({allergy}) 포함")
            
            # 결과 설정
            candidate.is_safe_for_user = is_safe
            candidate.safety_notes = safety_notes
            
            if is_safe:
                safe_candidates.append(candidate)
        
        return safe_candidates
    
    def _generate_medication_recommendation(self, medications: List[MedicationCandidate], 
                                          symptoms: List[str], session: IntegratedSession) -> str:
        """의약품 추천 응답 생성 (EXAONE 활용)"""
        medication_list = []
        for med in medications[:3]:  # 상위 3개만
            medication_list.append(f"- {med.medicine_name}: {med.effect}")
        
        symptoms_text = ", ".join(symptoms) if symptoms else "언급된 증상들"
        
        prompt = f"""
증상: {symptoms_text}
추천 의약품:
{chr(10).join(medication_list)}

위 정보를 바탕으로 친근하고 전문적인 의약품 추천 응답을 생성해주세요.
다음 구조로 작성해주세요:

1. 증상에 대한 이해 표현
2. 추천 의약품들과 각각의 효능 설명
3. 일반적인 복용법 안내
4. 주의사항 및 부작용 경고
5. 약사 상담 권유

**중요**: 
- 의사나 약사와의 상담을 권하는 문구 필수 포함
- 정확한 용법·용량은 제품 설명서 참조 안내
- 증상이 지속되면 병원 방문 권유
"""
        
        system_prompt = """당신은 친근하고 전문적인 약사 AI입니다. 
의약품 추천 시 안전성을 최우선으로 하고, 전문가 상담의 중요성을 강조해주세요."""
        
        if self.exaone:
            exaone_response = self.exaone.generate_response(prompt, system_prompt)
            
            if "⚠️" not in exaone_response:
                # 세션에 추천 의약품 저장
                for med in medications[:3]:
                    session.add_medication({
                        "name": med.medicine_name,
                        "effect": med.effect,
                        "usage": med.usage,
                        "recommended_at": datetime.now().isoformat()
                    })
                
                return exaone_response
        
        # EXAONE 실패 시 기본 응답
        return self._generate_basic_recommendation_response(medications, symptoms)
    
    def _generate_basic_recommendation_response(self, medications: List[MedicationCandidate], 
                                              symptoms: List[str]) -> str:
        """기본 추천 응답 생성 (EXAONE 실패 시)"""
        symptoms_text = ", ".join(symptoms) if symptoms else "언급하신 증상들"
        
        response = f"💊 **{symptoms_text}에 도움이 될 수 있는 의약품들**:\n\n"
        
        for i, med in enumerate(medications[:3], 1):
            response += f"**{i}. {med.medicine_name}**\n"
            response += f"   📋 효능: {med.effect}\n"
            response += f"   💡 용법: {med.usage}\n"
            
            if med.precautions:
                response += f"   ⚠️ 주의사항: {med.precautions[:100]}...\n"
            
            response += "\n"
        
        response += """🔍 **복용 전 확인사항**:
• 정확한 용법·용량은 제품 설명서를 참조하세요
• 다른 복용 중인 약물과의 상호작용을 확인하세요
• 알레르기나 부작용 발생 시 즉시 중단하세요

💡 **약사 상담 권장**: 개인의 건강 상태에 맞는 정확한 의약품 선택을 위해 약사와 상담하시기 바랍니다.

⚠️ **중요**: 증상이 지속되거나 악화되면 의료진과 상담하세요."""
        
        return response
    
    def _generate_safety_warning_response(self, medications: List[MedicationCandidate], 
                                        session: IntegratedSession) -> str:
        """안전성 경고 응답 생성"""
        user_profile = session.context.get("user_profile", {})
        
        warnings = []
        if user_profile.get("is_pregnant", False):
            warnings.append("임신 중")
        if user_profile.get("age_group") == "소아":
            warnings.append("소아")
        if user_profile.get("allergies"):
            warnings.append(f"알레르기 ({', '.join(user_profile['allergies'])})")
        
        warning_text = ", ".join(warnings) if warnings else "현재 상태"
        
        return f"""⚠️ **안전성 주의**:
검색된 의약품들이 있지만, {warning_text}로 인해 안전하지 않을 수 있습니다.

🏥 **권장사항**:
• 의사나 약사와 직접 상담하세요
• 개인 건강 상태에 맞는 안전한 대안을 문의하세요
• 자가 치료보다는 전문가의 도움을 받으시기 바랍니다

💡 증상이 지속되면 의료기관을 방문하세요."""
    
    def _get_specific_medication_info(self, medication_name: str, session: IntegratedSession) -> str:
        """특정 의약품 상세 정보"""
        if self.medication_data is None:
            return "의약품 데이터베이스를 사용할 수 없습니다."
        
        # 의약품명으로 정확 검색
        matches = self.medication_data[
            self.medication_data['medicine_name'].str.contains(medication_name, case=False, na=False)
        ]
        
        if len(matches) > 0:
            med_info = matches.iloc[0]
            
            response = f"💊 **{med_info.get('medicine_name', medication_name)} 정보**:\n\n"
            response += f"📋 **효능**: {med_info.get('effect', '정보 없음')}\n\n"
            response += f"💡 **용법**: {med_info.get('usage', '정보 없음')}\n\n"
            response += f"⚠️ **주의사항**: {med_info.get('precautions', '정보 없음')}\n\n"
            
            # 안전성 확인
            candidate = MedicationCandidate(
                medicine_name=str(med_info.get('medicine_name', '')),
                similarity_score=1.0,
                effect=str(med_info.get('effect', '')),
                usage=str(med_info.get('usage', '')),
                precautions=str(med_info.get('precautions', '')),
                is_safe_for_user=True,
                safety_notes=[]
            )
            
            safe_meds = self._filter_by_safety([candidate], session)
            
            if not safe_meds:
                response += "🚨 **개인 안전성 경고**: 현재 사용자 프로필로는 이 의약품이 적합하지 않을 수 있습니다.\n\n"
            
            response += "💡 **정확한 복용법과 주의사항은 약사와 상담하세요.**"
            
            return response
        else:
            return f"'{medication_name}' 의약품 정보를 찾을 수 없습니다. 정확한 의약품명을 확인해주세요."
    
    def _generate_medication_info_response(self, medications: List[MedicationCandidate], 
                                         session: IntegratedSession) -> str:
        """일반 의약품 정보 응답 생성"""
        response = "💊 **의약품 검색 결과**:\n\n"
        
        for i, med in enumerate(medications, 1):
            response += f"**{i}. {med.medicine_name}**\n"
            response += f"   📋 효능: {med.effect[:100]}...\n"
            response += f"   💡 용법: {med.usage[:100]}...\n\n"
        
        response += "💡 자세한 정보는 의약품명을 정확히 입력해서 다시 문의해주세요."
        
        return response

    def get_service_stats(self) -> Dict[str, Any]:
        """서비스 통계 정보"""
        return {
            "medication_database_loaded": self.medication_data is not None,
            "total_medications": len(self.medication_data) if self.medication_data is not None else 0,
            "index_built": self.medication_index is not None,
            "exaone_available": self.exaone is not None,
            "safety_rules_loaded": len(self.safety_rules)
        }