# ~/code/backend/test_intent_disease_cli.py
"""
의도파악 + 질병예측 CLI 테스트 코드
exaon_v5.txt 기반 - 완전 동일한 로직 사용
"""

import os, sys
import requests
import pandas as pd
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional
import re
from dataclasses import dataclass
from enum import Enum
import logging
import traceback
from collections import Counter


# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# (0) RAG 관련 클래스 및 데이터 모델 - exaon_v5.txt 완전 동일
# =============================================================================

class RAGContentType(Enum):
    """RAG 컨텐츠 타입"""
    QA = "qa"
    MEDICAL_DOC = "medical_doc"

@dataclass
class RAGDocument:
    """RAG 문서 데이터 클래스"""
    doc_id: str
    content: str
    metadata: Dict
    content_type: RAGContentType
    embedding: Optional[np.ndarray] = None

class RAGIndexManager:
    """RAG 인덱스 관리 클래스 - exaon_v5.txt 완전 동일"""
    
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.qa_index = None
        self.medical_doc_index = None
        self.qa_documents = []
        self.medical_documents = []
        
    def load_rag_data(self):
        """6개 CSV 파일에서 RAG 데이터 로드 - exaon_v5.txt 완전 동일"""
        print("🔄 RAG 데이터 로딩 시작...")
        
        # Q&A 데이터 로드 (clean_51004.csv)
        self._load_qa_data()
        
        # 의료 문서 데이터 로드 (나머지 5개 clean 파일들)
        self._load_medical_documents()
        
        # 인덱스 구축
        self._build_indexes()
        
        print("✅ RAG 데이터 로딩 완료!")
        print("   - Q&A 문서: {}개".format(len(self.qa_documents)))
        print("   - 의료 문서: {}개".format(len(self.medical_documents)))
        
    def _load_qa_data(self):
        """Q&A 데이터 로드 (clean_51004.csv) - exaon_v5.txt 완전 동일"""
        try:
            df = pd.read_csv("clean_51004.csv", encoding="utf-8")
            print("📋 Q&A 데이터 로드: {}행".format(len(df)))
            
            for idx, row in df.iterrows():
                if pd.notna(row.get('question')) and pd.notna(row.get('answer')):
                    # Q&A 쌍을 하나의 문서로 구성
                    content = "질문: {}\n답변: {}".format(row['question'], row['answer'])
                    
                    doc = RAGDocument(
                        doc_id="qa_{}".format(row.get('doc_id', idx)),
                        content=content,
                        metadata={
                            "disease_name": row.get('disease_name', ''),
                            "section_title": row.get('section_title', ''),
                            "question": row['question'],
                            "answer": row['answer'],
                            "url": row.get('url', ''),
                            "type": row.get('type', '')
                        },
                        content_type=RAGContentType.QA
                    )
                    self.qa_documents.append(doc)
                    
        except Exception as e:
            print("❌ Q&A 데이터 로드 실패: {}".format(e))
    
    def _load_medical_documents(self):
        """의료 문서 데이터 로드 (나머지 5개 clean 파일들) - exaon_v5.txt 완전 동일"""
        medical_files = [
            "clean_55588.csv", "clean_56763.csv", "clean_58572.csv", 
            "clean_66149.csv", "clean_63166.csv"
        ]
        
        for filename in medical_files:
            try:
                if not os.path.exists(filename):
                    print("⚠️ 파일 없음: {}".format(filename))
                    continue
                    
                df = pd.read_csv(filename, encoding="utf-8")
                print("📋 의료 문서 로드: {} - {}행".format(filename, len(df)))
                
                for idx, row in df.iterrows():
                    if pd.notna(row.get('content')) and len(str(row['content']).strip()) > 10:
                        # 의료 문서 구성
                        content_parts = []
                        
                        if pd.notna(row.get('disease_name')):
                            content_parts.append("질병: {}".format(row['disease_name']))
                        
                        if pd.notna(row.get('section_title')):
                            content_parts.append("섹션: {}".format(row['section_title']))
                            
                        content_parts.append("내용: {}".format(row['content']))
                        
                        content = "\n".join(content_parts)
                        
                        doc = RAGDocument(
                            doc_id="doc_{}_{}".format(filename, row.get('doc_id', idx)),
                            content=content,
                            metadata={
                                "disease_name": row.get('disease_name', ''),
                                "section_title": row.get('section_title', ''),
                                "content_length": row.get('content_length', 0),
                                "url": row.get('url', ''),
                                "status": row.get('status', ''),
                                "source_file": filename
                            },
                            content_type=RAGContentType.MEDICAL_DOC
                        )
                        self.medical_documents.append(doc)
                        
            except Exception as e:
                print("❌ {} 로드 실패: {}".format(filename, e))
    
    def _create_content_from_row(self, row, file_name: str) -> str:
        """행 데이터를 검색 가능한 텍스트로 변환"""
        content_parts = []
        
        # clean_ 파일들 처리
        if file_name.startswith('clean_'):
            # Q&A 데이터가 아닌 정제된 의료 데이터
            for col, val in row.items():
                if pd.notna(val) and str(val).strip():
                    content_parts.append(f"{col}: {str(val)}")
        
        # 질병 관련 파일
        elif any(keyword in file_name.lower() for keyword in ['disease', 'symptom']):
            disease = row.get('disease', row.get('disnm_ko', ''))
            symptoms = row.get('symptoms', row.get('sym', ''))
            department = row.get('department', row.get('dep', ''))
            
            if disease:
                content_parts.append(f"질병: {disease}")
            if symptoms:
                content_parts.append(f"증상: {symptoms}")
            if department:
                content_parts.append(f"진료과: {department}")
        
        # 의약품 관련 파일
        elif any(keyword in file_name.lower() for keyword in ['medicine', 'drug', 'pharmacy']):
            medicine = row.get('itemName', row.get('medicine', ''))
            effect = row.get('efcyQesitm', row.get('effect', ''))
            
            if medicine:
                content_parts.append(f"의약품: {medicine}")
            if effect:
                content_parts.append(f"효과: {effect}")
        
        # 기타 파일
        else:
            for col, val in row.items():
                if pd.notna(val) and str(val).strip():
                    content_parts.append(f"{col}: {str(val)}")
        
        return " ".join(content_parts)
    
    def _build_indexes(self):
        """FAISS 인덱스 구축 - exaon_v5.txt 완전 동일"""
        print("🔄 RAG 인덱스 구축 중...")
        
        # Q&A 인덱스 구축
        if self.qa_documents:
            qa_embeddings = []
            for doc in self.qa_documents:
                embedding = self.embedding_model.encode([doc.content])[0]
                qa_embeddings.append(embedding)
                doc.embedding = embedding
            
            qa_matrix = np.vstack(qa_embeddings)
            faiss.normalize_L2(qa_matrix)
            
            self.qa_index = faiss.IndexFlatIP(qa_matrix.shape[1])
            self.qa_index.add(qa_matrix)
        
        # 의료 문서 인덱스 구축
        if self.medical_documents:
            doc_embeddings = []
            for doc in self.medical_documents:
                embedding = self.embedding_model.encode([doc.content])[0]
                doc_embeddings.append(embedding)
                doc.embedding = embedding
            
            doc_matrix = np.vstack(doc_embeddings)
            faiss.normalize_L2(doc_matrix)
            
            self.medical_doc_index = faiss.IndexFlatIP(doc_matrix.shape[1])
            self.medical_doc_index.add(doc_matrix)
        
        print("✅ RAG 인덱스 구축 완료!")
    
    def search_qa(self, query: str, top_k: int = 3) -> List[RAGDocument]:
        """Q&A 검색 - exaon_v5.txt 완전 동일"""
        if not self.qa_index or not self.qa_documents:
            return []
        
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.qa_index.search(query_embedding, top_k)
        
        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.qa_documents):
                results.append(self.qa_documents[idx])
        
        return results
    
    def search_medical_docs(self, query: str, top_k: int = 3) -> List[RAGDocument]:
        """의료 문서 검색 - exaon_v5.txt 완전 동일"""
        if not self.medical_doc_index or not self.medical_documents:
            return []
        
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.medical_doc_index.search(query_embedding, top_k)
        
        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.medical_documents):
                results.append(self.medical_documents[idx])
        
        return results

# =============================================================================
# (1) 세션 상태 및 EXAONE 클래스 - exaon_v5.txt 완전 동일
# =============================================================================

# 세션 상태 정의
session_state = {
    "history": [],
    "last_intent": None,
    "last_entity": None,
    "last_disease": None,
    "last_final_diagnosis": None,
    "last_medications": None,
    "last_department": None
}

class EXAONE:
    """EXAONE LLM 서비스 클래스 - exaon_v5.txt 완전 동일"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "exaone3.5:7.8b"):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.exaone_config = {
            "temperature": 0.1,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "num_predict": 3000,  # 🔧 토큰 수 증가 (1000 → 2000)
            "stop": ["사용자:", "환자:", "Human:", "Assistant:"]
        }

        self.endpoint = None
        if self._check_endpoint("generate"):
            self.endpoint = "generate"
        elif self._check_endpoint("chat"):
            self.endpoint = "chat"
        else:
            print("⚠️ Ollama 서버에 연결할 수 없습니다. 기본 응답 모드로 실행합니다.")
            self.endpoint = None

    def _check_endpoint(self, name: str) -> bool:
        try:
            r = requests.options(f"{self.base_url}/api/{name}", timeout=2)
            return r.status_code in (200, 204, 405)
        except Exception:
            return False

    def generate_response(self, prompt: str, timeout: int = 180) -> str:
        if self.endpoint == "generate":
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": self.exaone_config
            }
            url = f"{self.base_url}/api/generate"
            try:
                resp = requests.post(url, json=payload, timeout=timeout)
                resp.raise_for_status()
                data = resp.json()

                if "choices" in data and isinstance(data["choices"], list) and len(data["choices"]) > 0:
                    return data["choices"][0].get("text", "").strip()
                if "response" in data:
                    return data["response"].strip()
                return "ERROR: EXAONE generate 응답 파싱 실패"
            except Exception as e:
                return f"ERROR: EXAONE generate failed - {str(e)}"

        elif self.endpoint == "chat":
            chat_payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": "당신은 의료 상담 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                "stream": False
            }
            url = f"{self.base_url}/api/chat"
            try:
                resp = requests.post(url, json=chat_payload, timeout=timeout)
                resp.raise_for_status()
                data = resp.json()

                if "choices" in data and isinstance(data["choices"], list) and len(data["choices"]) > 0:
                    return data["choices"][0].get("message", {}).get("content", "").strip()
                if "response" in data:
                    return data["response"].strip()
                return "ERROR: EXAONE chat 응답 파싱 실패"
            except Exception as e:
                return f"ERROR: EXAONE chat failed - {str(e)}"
        else:
            return "EXAONE 서버에 연결할 수 없어 기본 의료 조언을 제공합니다. 정확한 진단을 위해 의료진과 상담하세요."

# =============================================================================
# (2) CSV 파일 탐색 및 컬럼 감지 함수들 - exaon_v5.txt 완전 동일
# =============================================================================

def detect_columns(df: pd.DataFrame, data_type: str) -> Dict[str, str]:
    """데이터프레임에서 해당 타입의 컬럼들을 자동 감지"""
    detected = {}
    
    if data_type == "disease":
        # 질병 관련 컬럼 패턴
        patterns = {
            "disease_name": ["disease", "disnm_ko", "병명", "질병명"],
            "symptoms": ["symptoms", "sym", "증상"],
            "symptoms_key": ["symptoms_key", "sym_k", "핵심증상"],
            "department": ["department", "dep", "진료과"]
        }
    elif data_type == "medication":
        # 의약품 관련 컬럼 패턴
        patterns = {
            "itemName": ["itemName", "약품명", "의약품명"],
            "efcyQesitm": ["efcyQesitm", "효능효과", "효과"]
        }
    elif data_type == "hospital":
        # 병원 관련 컬럼 패턴
        patterns = {
            "hospital_name": ["hospital_name", "병원명"],
            "address": ["address", "주소"],
            "treatment_departments": ["treatment", "진료과목"]
        }
    else:
        return detected
    
    for field, candidates in patterns.items():
        for col in df.columns:
            col_lower = col.lower()
            if any(candidate.lower() in col_lower for candidate in candidates):
                detected[field] = col
                break
    
    return detected

def discover_csv_files() -> Tuple[List[str], List[str], List[str]]:
    """현재 디렉토리에서 CSV 파일들을 자동 탐색하고 타입별로 분류"""
    files = [f for f in os.listdir('.') 
             if f.lower().endswith(".csv") and not f.startswith("clean_")]
    
    disease_files = []
    medication_files = []
    hospital_files = []

    for fname in files:
        try:
            df = pd.read_csv(fname, encoding="utf-8", low_memory=False)
        except Exception:
            continue

        d_cols = detect_columns(df, "disease")
        if "disease_name" in d_cols and (d_cols.get("symptoms") or d_cols.get("symptoms_key")):
            disease_files.append(fname)
            continue

        m_cols = detect_columns(df, "medication")
        if "itemName" in m_cols and "efcyQesitm" in m_cols:
            medication_files.append(fname)
            continue

        h_cols = detect_columns(df, "hospital")
        if "hospital_name" in h_cols and "address" in h_cols and "treatment_departments" in h_cols:
            hospital_files.append(fname)
            continue

    return disease_files, medication_files, hospital_files

# =============================================================================
# (3) KM-BERT 임베딩 모델 클래스 - exaon_v5.txt 완전 동일
# =============================================================================

class EmbeddingModel:
    """KM-BERT 임베딩 모델 클래스 - exaon_v5.txt 완전 동일"""
    
    def __init__(self, model_name: str = "madatnlp/km-bert"):
        print(f"🔄 임베딩 모델 로딩 중: {model_name}")
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print(f"✅ KM-BERT 임베딩 모델 로드 완료 (Device: {self.device})")

    def encode(self, texts: List[str]) -> np.ndarray:
        """텍스트를 벡터로 인코딩 - exaon_v5.txt 완전 동일"""
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**encodings)
            last_hidden = outputs.last_hidden_state
            attention_mask = encodings.attention_mask.unsqueeze(-1)
            masked_hidden = last_hidden * attention_mask
            sum_hidden = masked_hidden.sum(dim=1)
            lengths = attention_mask.sum(dim=1)
            sentence_embeddings = sum_hidden / lengths.clamp(min=1e-9)
            return sentence_embeddings.cpu().numpy()

# =============================================================================
# (4) 질병 데이터 로드 및 인덱스 구축 함수들 - exaon_v5.txt 완전 동일
# =============================================================================

def load_and_build_disease_indexes(
    csv_paths: List[str],
    embedding_model: EmbeddingModel
) -> Tuple[faiss.IndexFlatIP, faiss.IndexFlatIP, List[Dict]]:
    """질병 데이터 로드 및 FAISS 인덱스 구축 - exaon_v5.txt 완전 동일"""
    
    all_key_embs = []
    all_full_embs = []
    all_docs_meta = []

    for path in csv_paths:
        df = pd.read_csv(path, encoding="utf-8", low_memory=False)
        detected = detect_columns(df, "disease")
        if "disease_name" not in detected or not (detected.get("symptoms") or detected.get("symptoms_key")):
            continue

        for _, row in df.iterrows():
            disease_name = str(row.get(detected["disease_name"], "")).strip()
            if not disease_name:
                continue

            # 핵심 증상과 전체 증상 추출
            key_symptoms = ""
            if detected.get("symptoms_key"):
                key_symptoms = str(row.get(detected["symptoms_key"], "")).strip()
            
            full_symptoms = ""
            if detected.get("symptoms"):
                full_symptoms = str(row.get(detected["symptoms"], "")).strip()
            elif key_symptoms:
                full_symptoms = key_symptoms

            if not full_symptoms:
                continue

            # 진료과 정보
            department = ""
            if detected.get("department"):
                department = str(row.get(detected["department"], "")).strip()

            # 메타데이터 구성
            meta = {
                "disease": disease_name,
                "key_symptoms": key_symptoms,
                "symptoms": full_symptoms,
                "department": department,
                "source_file": path
            }
            all_docs_meta.append(meta)

    if not all_docs_meta:
        raise ValueError("질병 데이터를 찾을 수 없습니다.")

    print(f"✅ 질병 데이터 로드 완료: {len(all_docs_meta)}개")

    # 임베딩 생성
    key_texts = [meta["key_symptoms"] or meta["symptoms"] for meta in all_docs_meta]
    full_texts = [meta["symptoms"] for meta in all_docs_meta]

    key_embeddings = embedding_model.encode(key_texts)
    full_embeddings = embedding_model.encode(full_texts)

    # FAISS 인덱스 구축
    dimension = key_embeddings.shape[1]
    index_key = faiss.IndexFlatIP(dimension)
    index_full = faiss.IndexFlatIP(dimension)

    faiss.normalize_L2(key_embeddings)
    faiss.normalize_L2(full_embeddings)

    index_key.add(key_embeddings)
    index_full.add(full_embeddings)

    return index_key, index_full, all_docs_meta

# =============================================================================
# (5) 의도 파악 함수 - "처음으로" 기능 추가
# =============================================================================

def detect_intent_with_rag(user_text: str, rag_manager: RAGIndexManager) -> str:
    """RAG 강화된 의도 파악 + 처음으로 기능 추가"""
    
    user_text = user_text.strip()
    
    # 🆕 1) "처음으로" 최우선 체크
    reset_patterns = [
        "처음으로", "처음부터", "다시", "리셋", "reset",
        "새로 시작", "초기화", "돌아가", "그만", "취소", "나가기"
    ]
    
    user_lower = user_text.lower()
    for pattern in reset_patterns:
        if pattern in user_lower:
            print(f"🔄 리셋 패턴 감지: '{pattern}' → 세션 초기화")
            return "reset"
    
    # 🆕 2) 일상적인 증상 표현 우선 체크 (가장 중요!)
    symptom_patterns = [
        "아프", "아파", "저리", "저려", "쑤시", "쑤셔", "따끔", "콕콕", "쿵쿵",
        "묵직", "무겁", "어지럽", "메스껍", "토할", "열나", "오한", "기침",
        "가래", "콧물", "코막힘", "목아픔", "속쓰림", "설사", "변비", "몸살",
        "피곤", "무력감", "답답", "숨차", "두근", "가슴", "배", "등", "허리",
        "목", "어깨", "팔", "다리", "무릎", "발", "머리", "눈", "귀"
    ]
    
    if any(pattern in user_lower for pattern in symptom_patterns):
        print(f"🩺 증상 표현 감지: '{user_text}' → disease_diagnosis")
        return "disease_diagnosis"
    
    # 3) 기존 의약품 관련 체크
    if any(keyword in user_lower for keyword in ["약", "의약품", "처방", "복용", "먹을", "드실", "추천"]):
        if "뭐" in user_lower or "어떤" in user_lower or "무슨" in user_lower:
            return "medication_recommend"
    
    # 4) 특정 의약품명 감지
    for keyword in ["타이레놀", "애드빌", "부루펜", "낙센", "게보린", "판콜드", "서방정"]:
        if keyword in user_lower:
            session_state["last_entity"] = keyword
            return "medication_info"
    
    # 5) 질병 정보 vs 진단 구분
    if "설명" in user_lower or "뭐" in user_lower or "대해" in user_lower or "이란" in user_lower:
        # 전역 변수 사용 (안전하게 처리)
        if 'global_meds_names' in globals():
            for med in global_meds_names:
                med_clean = med.replace("-", "").replace(" ", "").lower()
                if med_clean in user_lower.replace("-", "").replace(" ", ""):
                    session_state["last_entity"] = med
                    return "medication_info"
        
        if 'global_disease_names' in globals():
            for disease in global_disease_names:
                disease_clean = disease.replace("-", "").replace(" ", "").lower()
                if disease_clean in user_lower.replace("-", "").replace(" ", ""):
                    session_state["last_entity"] = disease
                    return "disease_info"
        
        return "disease_info"
    
    if any(keyword in user_lower for keyword in ["병원", "주변", "근처"]):
        return "hospital_search"
    
    # 6) RAG 검색을 통한 Intent 강화
    qa_results = rag_manager.search_qa(user_text, top_k=2)
    
    if qa_results:
        for qa_doc in qa_results:
            answer = qa_doc.metadata.get('answer', '').lower()
            if any(keyword in answer for keyword in ['약', '의약품', '복용', '처방']):
                return "medication_recommend"
            if any(keyword in answer for keyword in ['진단', '질병', '증상', '검사']):
                return "disease_diagnosis"
    
    # 4) LLM Fallback - 개선된 프롬프트
    prompt = (
        "다음 사용자 문장의 의도를 정확히 분류해주세요. 다음 5가지 중 하나로만 답하세요:\n\n"
        "1) disease_diagnosis (질병 진단) - 증상 호소, 몸이 아프다는 표현\n"
        "   예: '머리가 아파요', '손가락이 저려요', '기침이 나요', '열이 나요'\n\n"
        "2) disease_info (질병 정보 조회) - 특정 질병에 대한 설명 요청\n"
        "   예: '감기란 무엇인가요', '당뇨병에 대해 설명해주세요'\n\n"
        "3) medication_recommend (의약품 추천) - 증상에 맞는 약 추천 요청\n"
        "   예: '두통에 좋은 약이 뭐가 있나요', '감기약 추천해주세요'\n\n"
        "4) medication_info (의약품 정보 조회) - 특정 약에 대한 정보\n"
        "   예: '타이레놀이 뭐에 좋나요', '애드빌 부작용이 있나요'\n\n"
        "5) hospital_search (병원 검색) - 병원 찾기\n"
        "   예: '근처 병원 찾아주세요', '내과 병원 어디 있나요'\n\n"
        f"사용자 문장: '{user_text.strip()}'\n"
        "답변: "
    )
    
    exaone = EXAONE()
    llm_intent = exaone.generate_response(prompt).strip().lower()
    
    print(f"🤖 LLM 의도 분석: '{user_text}' → '{llm_intent}'")
    
    # 더 견고한 파싱
    if "disease_diagnosis" in llm_intent or "질병 진단" in llm_intent or "진단" in llm_intent:
        return "disease_diagnosis"
    elif "medication_recommend" in llm_intent or "의약품 추천" in llm_intent:
        return "medication_recommend"
    elif "medication_info" in llm_intent or "의약품 정보" in llm_intent:
        return "medication_info"
    elif "disease_info" in llm_intent or "질병 정보" in llm_intent:
        return "disease_info"
    elif "hospital_search" in llm_intent or "병원" in llm_intent:
        return "hospital_search"
    
    # 기본값: 증상처럼 보이면 진단으로
    print(f"⚠️ LLM 결과 애매함 → 기본값: disease_diagnosis")
    return "disease_diagnosis"

# =============================================================================
# (6) LLM 백업 해석 함수 - exaon_v5.txt 완전 동일
# =============================================================================

def interpret_yes_no(user_reply: str) -> bool:
    """LLM 백업을 활용한 사용자 답변 해석 - exaon_v5.txt 완전 동일"""
    
    user_reply = user_reply.strip()
    
    # 1차: 직접 매칭
    positive_keywords = ["네", "예", "맞아", "있어", "그래", "응", "어", "있습니다", "느껴져", "심해", "많이"]
    negative_keywords = ["아니", "없어", "안", "아니야", "별로", "없습니다", "안 느껴져"]
    
    user_lower = user_reply.lower()
    
    if any(pos in user_lower for pos in positive_keywords):
        print(f"   ✅ 직접 매칭: '{user_reply}' → 있음")
        return True
    if any(neg in user_lower for neg in negative_keywords):
        print(f"   ❌ 직접 매칭: '{user_reply}' → 없음")
        return False
    
    # 2차: LLM 백업 해석
    print(f"   🤖 애매한 답변, LLM 해석 중: '{user_reply}'")
    
    prompt = (
        "다음 환자의 답변이 '예(있음)'인지 '아니오(없음)'인지만 판단해주세요.\n"
        "간단히 '예' 또는 '아니오'로만 답하세요.\n"
        f"환자 답변: {user_reply}"
    )
    
    exaone = EXAONE()
    result = exaone.generate_response(prompt).strip().lower()
    
    print(f"   🤖 LLM 해석: '{user_reply}' → '{result}'")
    
    if "예" in result or "yes" in result or "긍정" in result:
        print("   🔍 최종 해석: ✅ 있음")
        return True
    elif "아니오" in result or "no" in result or "부정" in result:
        print("   🔍 최종 해석: ❌ 없음")
        return False
    else:
        print("   ⚠️ 애매한 응답 → 기본값: ❌ 없음")
        return False

# =============================================================================
# (7) RAG 강화된 질병 진단 엔진 - exaon_v5.txt 기반 + 설명 강화
# =============================================================================

class DiseaseInferenceEngineWithRAG:
    """RAG 기능이 강화된 질병 진단 엔진 - exaon_v5.txt 기반 + 설명 강화"""
    
    def __init__(self, index_key: faiss.IndexFlatIP, index_full: faiss.IndexFlatIP, 
                 all_docs_meta: List[Dict], embedding_model: EmbeddingModel,
                 rag_manager: RAGIndexManager, alpha: float = 0.7, beta: float = 0.3):
        self.index_key = index_key
        self.index_full = index_full
        self.all_docs_meta = all_docs_meta
        self.embedding_model = embedding_model
        self.rag_manager = rag_manager
        self.alpha = alpha
        self.beta = beta

        # 기존 증상 키워드 어휘집 구축
        self.sym_k_vocab = set()
        for meta in all_docs_meta:
            ks = meta.get("key_symptoms")
            if isinstance(ks, str) and ks.strip():
                for tk in ks.split():
                    if len(tk) > 1 and not tk.endswith(("가", "은", "는", "이", "서")):
                        self.sym_k_vocab.add(tk)

    def _check_reset_intent(self, user_input: str) -> bool:
        """진단 중에도 리셋 의도 체크"""
        reset_patterns = [
            "처음으로", "처음부터", "다시", "리셋", "reset",
            "새로 시작", "초기화", "돌아가", "그만", "취소", "나가기"
        ]
        
        user_lower = user_input.lower().strip()
        for pattern in reset_patterns:
            if pattern in user_lower:
                return True
        return False

    def extract_user_key_symptoms(self, user_text: str) -> List[str]:
        return [tk for tk in self.sym_k_vocab if tk in user_text]

    def get_candidate_diseases(self, user_text: str, topk: int = 10) -> List[Tuple[Dict, float]]:
        """기존 벡터 검색 + RAG 검색 결합 - exaon_v5.txt 완전 동일"""
        
        keys = self.extract_user_key_symptoms(user_text)
        query_key = " ".join(keys) if keys else user_text

        # 1) 기존 방식: 핵심 증상 기반 검색
        qk_emb = self.embedding_model.encode([query_key])
        faiss.normalize_L2(qk_emb)
        Dk, Ik = self.index_key.search(qk_emb, topk)
        cand_idxs = Ik[0]
        key_scores = Dk[0]

        uf_emb = self.embedding_model.encode([user_text])
        faiss.normalize_L2(uf_emb)

        results = []
        for rank, idx in enumerate(cand_idxs):
            if idx < 0 or idx >= len(self.all_docs_meta):
                continue
            meta = self.all_docs_meta[idx]
            df_emb = self.embedding_model.encode([meta["symptoms"]])
            faiss.normalize_L2(df_emb)
            sim_full = float(cosine_similarity(uf_emb, df_emb)[0][0])
            sim_key = float(key_scores[rank])
            final_score = self.alpha * sim_key + self.beta * sim_full
            results.append((meta, final_score))

        # 2) RAG 검색으로 보너스 점수 부여
        rag_qa_results = self.rag_manager.search_qa(user_text, top_k=3)
        rag_mentioned_diseases = set()
        for qa_doc in rag_qa_results:
            disease_name = qa_doc.metadata.get('disease_name', '')
            if disease_name:
                rag_mentioned_diseases.add(disease_name)

        # RAG에서 언급된 질병에 보너스 점수
        for i, (meta, score) in enumerate(results):
            if meta['disease'] in rag_mentioned_diseases:
                results[i] = (meta, score + 0.2)  # +20% 보너스

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:topk]

    def _filter_candidates_by_answers(self, candidates: List[Tuple[Dict, float]], 
                                    user_answers: Dict[str, bool]) -> List[Tuple[Dict, float]]:
        """사용자 답변에 따른 후보 질병 점수 조정 - exaon_v5.txt 완전 동일"""
        
        updated_results = []
        
        for meta, base_score in candidates:
            disease_name = meta['disease']
            key_symptoms = [s.strip() for s in meta.get("key_symptoms", "").split(",") if s.strip()]
            symptoms = [s.strip() for s in meta.get("symptoms", "").split(",") if s.strip()]
            
            adjusted_score = base_score
            negative_key_count = 0  # 부정된 핵심 증상 개수
            
            for symptom_asked, has_symptom in user_answers.items():
                if has_symptom:
                    # 핵심 증상 확인 시 +30% 보너스
                    if any(symptom_asked.lower() in key_symptom.lower() for key_symptom in key_symptoms):
                        adjusted_score *= 1.3
                        print(f"      + {disease_name}: 핵심증상 '{symptom_asked}' 확인 → +30%")
                    # 일반 증상 확인 시 +10% 보너스
                    elif any(symptom_asked.lower() in symptom.lower() for symptom in symptoms):
                        adjusted_score *= 1.1
                        print(f"      + {disease_name}: 일반증상 '{symptom_asked}' 확인 → +10%")
                else:
                    # 핵심 증상 부정 시 -80% 페널티
                    if any(symptom_asked.lower() in key_symptom.lower() for key_symptom in key_symptoms):
                        adjusted_score *= 0.2
                        negative_key_count += 1
                        print(f"      - {disease_name}: 핵심증상 '{symptom_asked}' 부정 → -80%")
                    # 일반 증상 부정 시 -50% 페널티
                    elif any(symptom_asked.lower() in symptom.lower() for symptom in symptoms):
                        adjusted_score *= 0.5
                        print(f"      - {disease_name}: 일반증상 '{symptom_asked}' 부정 → -50%")
            
            # 핵심 증상 2개 이상 부정 시 추가 -90% 페널티
            if negative_key_count >= 2:
                adjusted_score *= 0.1
                print(f"      -- {disease_name}: 핵심증상 {negative_key_count}개 부정 → 추가 -90%")
            
            updated_results.append((meta, adjusted_score))
        
        # 점수순 정렬
        updated_results.sort(key=lambda x: x[1], reverse=True)
        return updated_results

    def _should_ask_more_questions(self, filtered_candidates: List[Tuple[Dict, float]], 
                                 user_answers: Dict[str, bool]) -> bool:
        """추가 질문 필요성 판단 - exaon_v5.txt 완전 동일"""
        
        if len(filtered_candidates) < 2:
            return False
        
        # 1-2위 점수 차이가 적으면 추가 질문 필요
        if len(filtered_candidates) >= 2:
            first_score = filtered_candidates[0][1]
            second_score = filtered_candidates[1][1]
            
            print("🔍 추가 질문 필요성 검토:")
            print(f"   1위: {filtered_candidates[0][0]['disease']} ({first_score:.3f}점)")
            print(f"   2위: {filtered_candidates[1][0]['disease']} ({second_score:.3f}점)")
            print(f"   점수 차이: {abs(first_score - second_score):.3f}")
            
            if first_score < 0.5:
                print("   → 1위 점수가 낮아 추가 질문 필요")
                return True
            
            if abs(first_score - second_score) < 0.3:
                print("   → 점수 차이가 적어 추가 질문 필요")
                return True
        
        if filtered_candidates[0][1] < 0.6:
            print("   → 1위 신뢰도가 낮아 추가 질문 필요")
            return True
        
        print("   → 충분한 차별화 완료, 추가 질문 불필요")
        return False

    def _generate_targeted_questions(self, filtered_candidates: List[Tuple[Dict, float]], 
                                   asked_questions: List[str]) -> List[str]:
        """상위 질병들의 특징적 증상으로 차별화 질문 생성 - exaon_v5.txt 완전 동일"""
        
        if len(filtered_candidates) < 2:
            return []
        
        top_candidates = filtered_candidates[:3]
        print("🎯 상위 후보들의 특징 증상 분석 중...")
        
        disease_symptoms = {}
        for meta, score in top_candidates:
            disease_name = meta['disease']
            key_symptoms = [s.strip() for s in meta.get("key_symptoms", "").split(",") if s.strip()]
            disease_symptoms[disease_name] = key_symptoms
            print(f"   - {disease_name}: {', '.join(key_symptoms)}")
        
        # 차별화 가능한 증상 찾기
        all_symptoms = set()
        for symptoms in disease_symptoms.values():
            all_symptoms.update([s.lower() for s in symptoms])
        
        remaining_symptoms = [s for s in all_symptoms if s not in [q.lower() for q in asked_questions]]
        
        differential_symptoms = []
        for symptom in remaining_symptoms:
            count = sum(1 for symptoms in disease_symptoms.values() 
                       if any(symptom in s.lower() for s in symptoms))
            
            if 0 < count < len(top_candidates):
                differential_symptoms.append((symptom, count))
        
        differential_symptoms.sort(key=lambda x: x[1])
        selected_questions = [symptom for symptom, _ in differential_symptoms[:2]]
        
        print(f"🔍 선택된 차별화 질문: {', '.join(selected_questions)}")
        return selected_questions

    def run_diagnosis(self, user_text: str) -> None:
        """RAG 강화된 진단 실행 - 설명 강화 버전"""
        global session_state

        print("🔍 RAG 강화된 질병 진단을 시작합니다...")
        
        # 기본 후보 추출 (RAG 보강됨)
        candidates = self.get_candidate_diseases(user_text, topk=5)
        if not candidates:
            msg = "관련된 질병 후보를 찾지 못했습니다. 의료 전문가에게 문의하세요."
            print(f"챗봇> {msg}")
            session_state["last_disease"] = msg
            session_state["last_intent"] = "disease_diagnosis"
            return

        print(f"\n=== 🔍 후보 질병 ({len(candidates)}개) ===")
        for idx, (meta, score) in enumerate(candidates, 1):
            print(f"{idx}. {meta['disease']} (유사도: {score:.3f})")
        
        # 나이 정보 수집 - exaon_v5.txt 완전 동일
        patient_age = None
        age_attempts = 0
        while patient_age is None and age_attempts < 3:
            age_input = input("\n챗봇> 더 정확한 진단을 위해 연령대를 알려주시겠어요? (예: 20대, 30대, 또는 구체적인 나이): ").strip()
            
            # 🆕 나이 입력에서도 "처음으로" 체크
            if age_input and self._check_reset_intent(age_input):
                print("🔄 세션 리셋 요청 감지 → 진단 중단")
                reset_session()
                return
            
            if not age_input:
                print("챗봇> 연령 정보 없이 진단을 계속하겠습니다.")
                break
            
            age_match = re.search(r'(\d+)', age_input)
            if age_match:
                patient_age = int(age_match.group(1))
                print(f"챗봇> 연령 정보가 확인되었습니다: {patient_age}세")
                break
            else:
                age_attempts += 1
                if age_attempts < 3:
                    print("챗봇> 숫자로 된 연령을 입력해주세요. (예: 25, 30대)")

        # 추가 증상 수집 - exaon_v5.txt 완전 동일
        print("\n챗봇> 혹시 다른 증상도 있으시다면 추가로 말씀해주세요 (없으면 엔터):")
        additional_symptoms = input("사용자> ").strip()
        
        # 🆕 추가 증상 입력에서도 "처음으로" 체크
        if additional_symptoms and self._check_reset_intent(additional_symptoms):
            print("🔄 세션 리셋 요청 감지 → 진단 중단")
            reset_session()
            return
        
        if additional_symptoms:
            print(f"\n🔄 추가 증상 반영 중: {additional_symptoms}")
            combined_text = f"{user_text} {additional_symptoms}"
            candidates = self.get_candidate_diseases(combined_text, topk=5)
            
            print(f"=== 🔍 업데이트된 후보 질병 ({len(candidates)}개) ===")
            for idx, (meta, score) in enumerate(candidates, 1):
                print(f"{idx}. {meta['disease']} (유사도: {score:.3f})")

        # 차별화 질문 시작 - exaon_v5.txt 완전 동일
        user_answers = {}
        all_asked_questions = []

        if len(candidates) > 1:
            print("\n=== 💬 증상 확인 질문 ===")
            
            # 각 후보 질병의 특징적 증상들로 질문 생성
            all_key_symptoms = set()
            for meta, score in candidates:
                key_symptoms = [s.strip() for s in meta.get("key_symptoms", "").split(",") if s.strip()]
                all_key_symptoms.update(key_symptoms)
            
            # 상위 3개 증상만 질문
            selected_symptoms = list(all_key_symptoms)[:3]
            
            for symptom in selected_symptoms:
                print(f"챗봇> '{symptom}' 증상이 있으신가요?")
                reply = input("사용자> ").strip()
                
                # 🆕 차별화 질문에서도 "처음으로" 체크
                if self._check_reset_intent(reply):
                    print("🔄 세션 리셋 요청 감지 → 진단 중단")
                    reset_session()
                    return
                
                has_symptom = interpret_yes_no(reply)
                user_answers[symptom] = has_symptom
                all_asked_questions.append(symptom)
                
                answer_text = "예" if has_symptom else "아니오"
                session_state["history"].append(("user", f"{symptom}: {answer_text}"))
                print()

        # 후보 재평가 - exaon_v5.txt 완전 동일
        print("\n=== 후보 질병 재평가 중 ===")
        filtered_candidates = self._filter_candidates_by_answers(candidates, user_answers)
        
        # 추가 질문이 필요한지 판단 - exaon_v5.txt 완전 동일
        if self._should_ask_more_questions(filtered_candidates, user_answers) and len(all_asked_questions) < 6:
            print("\n=== 상위 후보 차별화를 위한 추가 질문 ===")
            additional_questions = self._generate_targeted_questions(filtered_candidates, all_asked_questions)
            
            for question in additional_questions:
                print(f"챗봇> '{question}' 증상이 있으신가요?")
                reply = input("사용자> ").strip()
                
                # 🆕 추가 질문에서도 "처음으로" 체크
                if self._check_reset_intent(reply):
                    print("🔄 세션 리셋 요청 감지 → 진단 중단")
                    reset_session()
                    return
                
                has_symptom = interpret_yes_no(reply)
                user_answers[question] = has_symptom
                all_asked_questions.append(question)
                
                answer_text = "예" if has_symptom else "아니오"
                session_state["history"].append(("user", f"{question}: {answer_text}"))
                print()
            
            # 다시 재평가
            print("\n=== 최종 후보 질병 재평가 ===")
            filtered_candidates = self._filter_candidates_by_answers(candidates, user_answers)

        # 최종 후보 출력
        print("\n=== 🎯 최종 후보 질병 순위 ===")
        for idx, (meta, score) in enumerate(filtered_candidates[:5], 1):
            print(f"{idx}. {meta['disease']} (신뢰도: {score:.3f})")

        # 🆕 설명 강화된 최종 진단 생성
        final_diagnosis = self._generate_enhanced_final_diagnosis(filtered_candidates, user_answers, user_text)
        
        print(f"\n💊 최종 진단 결과:\n{final_diagnosis}")
        
        # 세션 상태 업데이트
        if filtered_candidates:
            session_state["last_disease"] = filtered_candidates[0][0]['disease']
            session_state["last_final_diagnosis"] = final_diagnosis
        
        session_state["last_intent"] = "disease_diagnosis"

    def _generate_enhanced_final_diagnosis(self, filtered_candidates: List[Tuple[Dict, float]], 
                                         user_answers: Dict[str, bool], original_symptoms: str) -> str:
        """🆕 설명 강화된 최종 진단 생성 (기존 구조 유지 + 프롬프트만 강화 + 코로나19 주의사항)"""
        
        if not filtered_candidates:
            return "충분한 정보가 없어 정확한 진단을 내릴 수 없습니다. 의료 전문가와 상담하시기 바랍니다."
        
        # 상위 3개 후보
        top_candidates = filtered_candidates[:3]
        
        # 🆕 코로나19 주의사항이 필요한 질병들 체크 (더 포괄적으로)
        covid_similar_diseases = ['감기', '상기도', '독감', '인플루엔자', '급성기관지염', '폐렴', '기관지염', '비염', '인후염', '급성 상기도 감염', '만성 폐렴', '만성 기침']
        top_disease_lower = top_candidates[0][0]['disease'].lower()
        needs_covid_warning = any(disease in top_disease_lower for disease in covid_similar_diseases)
        
        print(f"🔍 코로나19 주의사항 체크: '{top_disease_lower}' → {needs_covid_warning}")
        
        # 대화 히스토리 정리
        qa_history = []
        qa_history.append(f"초기 증상: {original_symptoms}")
        for symptom, has_symptom in user_answers.items():
            answer = "있음" if has_symptom else "없음"
            qa_history.append(f"{symptom}: {answer}")
        
        history_text = "\n".join(qa_history)
        
        # 🆕 설명 강화된 프롬프트 (기존 구조 유지하되 설명 강화)
        enhanced_prompt = f"""의료 전문가로서 다음 정보를 종합하여 근거 기반 진단을 제시하세요.

환자 증상 및 대화 내용:
{history_text}

가장 가능성 높은 질병들 (신뢰도 순):
{chr(10).join([f"{i+1}. {meta['disease']} (신뢰도: {score:.3f})" for i, (meta, score) in enumerate(top_candidates)])}

다음 구조로 답변하세요:

1. 🎯 **가장 가능성 높은 진단** (1-2개)
   - 진단명과 선택 근거를 명확히 제시

2. 🔍 **의학적 판단 근거**
   - 증상 패턴이 진단과 일치하는 이유
   - 다른 후보 질병들과의 차이점

3. 🏥 **권장 진료과**

4. 💡 **즉시 대처법**
   - 구체적인 조치사항과 생활 관리법

5. ⚠️ **주의사항**
   - 증상 악화 시 응급 상황 판단 기준

6. 📅 **병원 방문 시점**
   - 즉시 방문 vs 경과 관찰 후 방문

**중요:** 이 분석은 의료진의 정확한 진단을 대체할 수 없습니다."""

        # 🆕 코로나19 주의사항 추가 (더 강조)
        if needs_covid_warning:
            enhanced_prompt += f"""

🚨🚨 **COVID-19 감별 필수 주의사항** 🚨🚨
진단된 질병의 증상은 COVID-19와 거의 동일합니다!
- 즉시 코로나19 검사(신속검사/PCR)를 받으세요
- 검사 결과 나올 때까지 자가격리하세요
- 마스크 착용하고 타인과의 접촉을 피하세요
- 호흡곤란 시 즉시 응급실로 가세요"""

        try:
            exaone = EXAONE()
            diagnosis_result = exaone.generate_response(enhanced_prompt)
            return diagnosis_result
        except Exception as e:
            return f"진단 처리 중 오류가 발생했습니다: {str(e)}\n의료진과 직접 상담하시기 바랍니다."

# =============================================================================
# (8) 질병 정보 검색 핸들러 - exaon_v5.txt 기반
# =============================================================================

def handle_disease_info_with_rag(user_text: str, rag_manager: RAGIndexManager):
    """RAG 검색이 강화된 질병 정보 조회 - exaon_v5.txt 완전 동일"""
    global session_state, all_docs_meta
    
    # 1) 기존 DB에서 직접 매칭 시도
    matched = find_disease_by_name_in_input(user_text, all_docs_meta)
    
    if matched:
        print("챗봇> '{}'에 대한 정보입니다:".format(matched['disease']))
        symptoms = matched.get("symptoms", "").strip()
        department = matched.get("department", "").strip()
        
        if symptoms:
            print(" - 증상: {}".format(symptoms))
        if department:
            print(" - 진료과: {}".format(department))
    
    # 2) RAG 검색으로 추가 정보 제공
    disease_name = matched['disease'] if matched else user_text.strip()
    
    # Q&A 검색
    qa_results = rag_manager.search_qa(disease_name, top_k=2)
    if qa_results:
        print("\n📚 '{}'에 대한 관련 상담 사례:".format(disease_name))
        for i, qa_doc in enumerate(qa_results, 1):
            question = qa_doc.metadata.get('question', '')[:100]
            answer = qa_doc.metadata.get('answer', '')[:200]
            print("  {}. Q: {}...".format(i, question))
            print("     A: {}...".format(answer))
    
    # 의료 문서 검색
    doc_results = rag_manager.search_medical_docs(disease_name, top_k=2)
    if doc_results:
        print("\n📖 '{}'에 대한 상세 의료 정보:".format(disease_name))
        for i, doc in enumerate(doc_results, 1):
            content = doc.content[:200]
            print("  {}. {}...".format(i, content))
    
    # 3) 매칭 실패 시 LLM fallback + RAG 컨텍스트
    if not matched:
        # RAG 검색 결과를 컨텍스트로 활용
        context_info = ""
        if doc_results:
            context_info = "\n참고 정보:\n"
            for doc in doc_results:
                context_info += "- {}...\n".format(doc.content[:150])
        
        prompt = "질병 '{}'에 대해 간단히 정의, 증상, 치료법을 알려주세요.{}".format(user_text.strip(), context_info)
        exaone = EXAONE()
        llm_answer = exaone.generate_response(prompt)
        print("챗봇> '{}'에 대한 설명입니다:\n{}".format(user_text.strip(), llm_answer))
    
    session_state["last_intent"] = "disease_info"
    session_state["last_entity"] = matched["disease"] if matched else user_text.strip()

def find_disease_by_name_in_input(user_text: str, all_docs_meta: List[Dict]) -> Optional[Dict]:
    """사용자 입력에서 질병명 찾기"""
    input_norm = normalize_text(user_text)
    for meta in all_docs_meta:
        disease_norm = normalize_text(meta["disease"])
        if disease_norm and disease_norm in input_norm:
            return meta
    return None

def normalize_text(text: str) -> str:
    """텍스트 정규화"""
    return "".join(ch for ch in text if ch.isalnum()).lower()

# =============================================================================
# (9) 전역 변수 및 초기화 함수
# =============================================================================

# 전역 변수 선언 - exaon_v5.txt 완전 동일
all_docs_meta = []
global_disease_names = []
global_meds_names = []
DiseaseInferenceEngineWithRAG_instance = None

def reset_session():
    """세션 상태 초기화"""
    global session_state
    session_state = {
        "history": [],
        "last_intent": None,
        "last_entity": None,
        "last_disease": None,
        "last_final_diagnosis": None,
        "last_medications": None,
        "last_department": None
    }
    print("🔄 세션이 초기화되었습니다. 처음부터 다시 시작합니다.")

def initialize_system():
    """시스템 초기화 - exaon_v5.txt 기반"""
    global all_docs_meta, global_disease_names, global_meds_names, DiseaseInferenceEngineWithRAG_instance

    print("🚀 의도파악 + 질병예측 시스템 초기화 중...")

    # 1) CSV 파일 탐색
    disease_files, medication_files, hospital_files = discover_csv_files()

    print("📋 발견된 데이터 파일:")
    print(f"   - 질병 데이터: {disease_files}")
    print(f"   - 의약품 데이터: {medication_files}")
    print(f"   - 병원 데이터: {hospital_files}")

    if not disease_files:
        print("❌ 질병 데이터 파일을 찾을 수 없습니다.")
        print("💡 CSV 파일들을 현재 디렉토리에 준비해주세요.")
        return None, None

    # 2) 임베딩 모델 초기화
    emb_model = EmbeddingModel()

    # 3) RAG 매니저 초기화
    rag_manager = RAGIndexManager(emb_model)
    rag_manager.load_rag_data()

    # 4) 질병 데이터 로드 및 인덱스 구축
    try:
        index_key, index_full, all_docs_meta = load_and_build_disease_indexes(disease_files, emb_model)
        print(f"✅ 질병 데이터 로드 완료: {len(all_docs_meta)}개")
    except ValueError as e:
        print(f"❌ 질병 데이터 로드 오류: {e}")
        return None, None

    # 5) RAG 강화된 질병 진단 엔진 초기화
    DiseaseInferenceEngineWithRAG_instance = DiseaseInferenceEngineWithRAG(
        index_key=index_key,
        index_full=index_full,
        all_docs_meta=all_docs_meta,
        embedding_model=emb_model,
        rag_manager=rag_manager,
        alpha=0.7,
        beta=0.3
    )

    # 6) 글로벌 리스트 초기화 (Intent 파악용)
    global_disease_names = [meta["disease"] for meta in all_docs_meta if meta.get("disease")]
    global_meds_names = []  # 의약품 데이터가 없어도 동작하도록

    print("✅ 시스템 초기화 완료!")
    print(f"📊 로드된 데이터: 질병 {len(all_docs_meta)}개, Q&A {len(rag_manager.qa_documents)}개")

    # ─────────────────────────────────────────────────────────────────────
    # 👉 메모리에서 로드된 FAISS 인덱스를 로컬에 덤프하기
    from faiss_manager import FAISSIndexSaver

    saver = FAISSIndexSaver(index_dir="indexes")
    success = saver.save_all_indexes(
        rag_manager=rag_manager,
        disease_key_index=index_key,
        disease_full_index=index_full,
        disease_metadata=all_docs_meta,
        medication_index=None,         # 의약품 인덱스가 없으면 None
        medication_metadata=[],        # 빈 리스트
        hospital_data={}               # 빈 dict 또는 빈 리스트
    )

    if success:
        print("✅ FAISS 인덱스 로컬 덤프 완료: ndexes/ 아래 파일들을 확인하세요.")
    else:
        print("❌ FAISS 덤프 실패—로그를 확인해주세요.")
    # ─────────────────────────────────────────────────────────────────────

    return DiseaseInferenceEngineWithRAG_instance, rag_manager


# =============================================================================
# (9) 메인 CLI 함수
# =============================================================================

def main():
    """메인 CLI 루프"""
    print("="*80)
    print("🏥 의도파악 + 질병예측 CLI 테스트")
    print("📚 exaon_v5.txt 기반 완전 동일 로직")
    print("="*80)
    
    # 시스템 초기화
    diagnosis_engine, rag_manager = initialize_system()
    
    if not diagnosis_engine:
        print("❌ 시스템 초기화 실패")
        return
    
    print("\n💡 기능 테스트:")
    print("1. 의도 파악 (처음으로 기능 포함)")
    print("2. 질병 진단 (설명 강화 + 차별화 질문 + 코로나19 주의사항)")
    print("3. 질병 정보 검색 (RAG 기반)")
    print("4. '처음으로', '리셋', '다시' 등으로 세션 초기화")
    print("5. 'exit'로 종료")
    print("\n**중요:** 의료진과의 상담을 대체할 수 없습니다.")
    print("\n🚀 대화를 시작하세요!")
    print("📝 예시: '머리가 아파요', '코로나19에 대해 설명해줘'\n")
    
    while True:
        try:
            user_input = input("사용자> ").strip()
            
            if user_input.lower() in ["exit", "quit"]:
                print("챗봇> 시스템을 종료합니다.")
                break
            
            if not user_input:
                continue
            
            # 의도 파악
            intent = detect_intent_with_rag(user_input, rag_manager)
            print(f"🔍 감지된 의도: {intent}")
            
            # 의도별 처리
            if intent == "reset":
                reset_session()
                continue
            
            elif intent == "disease_diagnosis":
                print("\n🩺 질병 진단 모드 시작")
                diagnosis_engine.run_diagnosis(user_input)
            
            elif intent == "disease_info":
                print("\n📖 질병 정보 검색 모드")
                handle_disease_info_with_rag(user_input, rag_manager)
            
            else:
                print(f"챗봇> '{intent}' 의도가 감지되었지만, 현재는 질병 진단과 질병 정보 검색만 테스트 중입니다.")
                print("챗봇> 증상을 말씀해주시거나 특정 질병에 대해 문의해주세요.")
            
            print("\n" + "-"*50 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n챗봇> 시스템을 종료합니다.")
            break
        except Exception as e:
            print(f"❌ 오류 발생: {str(e)}")
            continue

if __name__ == "__main__":
    main()