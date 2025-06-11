"""
수정된 통합 의료 챗봇 v6 - 스마트한 차별화 질문 시스템
주요 수정사항:
1. 중복 질문 방지 - 이미 언급한 증상은 다시 묻지 않음
2. 세션 상태 개선 - 초기 증상 정보 유지
3. 필터링 로직 강화 - 빈 결과 방지
4. 종료 조건 개선 - 적절한 차별화 질문 수
"""

import os
import sys
import requests
import pandas as pd
import numpy as np
import faiss
import torch
import pickle
import json
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Tuple, Optional, Any, Set
import re
import logging
from dataclasses import dataclass
from enum import Enum
import traceback
from datetime import datetime, timedelta

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# 🚀 FAISS 인덱스 로더 클래스 - 기존과 동일
# =============================================================================

class PreBuiltIndexLoader:
    """사전 생성된 FAISS 인덱스 로더"""
    
    INDEX_DIR = "faiss_indexes"
    
    INDEX_FILES = {
        "rag_qa": "rag_qa_index.index",
        "rag_medical": "rag_medical_index.index", 
        "disease_key": "disease_key_index.index",
        "disease_full": "disease_full_index.index",
        "medication": "medication_index.index"
    }
    
    METADATA_FILES = {
        "rag_qa": "rag_qa_documents.pkl",
        "rag_medical": "rag_medical_documents.pkl",
        "disease": "disease_metadata.pkl",
        "medication": "medication_metadata.pkl"
    }
    
    CONFIG_FILE = "index_config.json"
    
    @classmethod
    def check_indexes_available(cls) -> bool:
        """사전 생성된 인덱스 사용 가능 여부 확인"""
        if not os.path.exists(cls.INDEX_DIR):
            return False
        
        # 필수 파일들 존재 확인
        essential_files = [
            cls.INDEX_FILES["rag_qa"],
            cls.INDEX_FILES["rag_medical"],
            cls.METADATA_FILES["rag_qa"], 
            cls.METADATA_FILES["rag_medical"]
        ]
        
        for filename in essential_files:
            if not os.path.exists(os.path.join(cls.INDEX_DIR, filename)):
                logger.warning(f"⚠️ 필수 인덱스 파일 누락: {filename}")
                return False
        
        logger.info("✅ 사전 생성된 인덱스 사용 가능")
        return True
    
    @classmethod
    def load_rag_indexes(cls) -> Tuple[Optional[faiss.Index], Optional[faiss.Index], List, List]:
        """RAG 인덱스 로드"""
        try:
            logger.info("🔄 사전 생성된 RAG 인덱스 로딩 중...")
            
            # Q&A 인덱스 및 문서 로드
            qa_index = None
            qa_documents = []
            
            qa_index_path = os.path.join(cls.INDEX_DIR, cls.INDEX_FILES["rag_qa"])
            qa_docs_path = os.path.join(cls.INDEX_DIR, cls.METADATA_FILES["rag_qa"])
            
            if os.path.exists(qa_index_path) and os.path.exists(qa_docs_path):
                qa_index = faiss.read_index(qa_index_path)
                with open(qa_docs_path, 'rb') as f:
                    qa_documents = pickle.load(f)
                logger.info(f"✅ RAG Q&A 인덱스 로드: {len(qa_documents)}개 문서")
            
            # 의료 문서 인덱스 및 문서 로드
            medical_index = None
            medical_documents = []
            
            medical_index_path = os.path.join(cls.INDEX_DIR, cls.INDEX_FILES["rag_medical"])
            medical_docs_path = os.path.join(cls.INDEX_DIR, cls.METADATA_FILES["rag_medical"])
            
            if os.path.exists(medical_index_path) and os.path.exists(medical_docs_path):
                medical_index = faiss.read_index(medical_index_path)
                with open(medical_docs_path, 'rb') as f:
                    medical_documents = pickle.load(f)
                logger.info(f"✅ RAG 의료문서 인덱스 로드: {len(medical_documents)}개 문서")
            
            return qa_index, medical_index, qa_documents, medical_documents
            
        except Exception as e:
            logger.error(f"❌ RAG 인덱스 로드 실패: {e}")
            return None, None, [], []
    
    @classmethod
    def load_disease_indexes(cls) -> Tuple[Optional[faiss.Index], Optional[faiss.Index], List]:
        """질병 인덱스 로드"""
        try:
            logger.info("🔄 사전 생성된 질병 인덱스 로딩 중...")
            
            disease_key_index = None
            disease_full_index = None
            disease_metadata = []
            
            # 인덱스 파일 경로
            key_index_path = os.path.join(cls.INDEX_DIR, cls.INDEX_FILES["disease_key"])
            full_index_path = os.path.join(cls.INDEX_DIR, cls.INDEX_FILES["disease_full"])
            metadata_path = os.path.join(cls.INDEX_DIR, cls.METADATA_FILES["disease"])
            
            # 인덱스 로드
            if os.path.exists(key_index_path):
                disease_key_index = faiss.read_index(key_index_path)
                logger.info("✅ 질병 Key 인덱스 로드 완료")
            
            if os.path.exists(full_index_path):
                disease_full_index = faiss.read_index(full_index_path)
                logger.info("✅ 질병 Full 인덱스 로드 완료")
            
            # 메타데이터 로드
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    disease_metadata = pickle.load(f)
                logger.info(f"✅ 질병 메타데이터 로드: {len(disease_metadata)}개")
            
            return disease_key_index, disease_full_index, disease_metadata
            
        except Exception as e:
            logger.error(f"❌ 질병 인덱스 로드 실패: {e}")
            return None, None, []
    
    @classmethod
    def load_medication_index(cls) -> Tuple[Optional[faiss.Index], List]:
        """의약품 인덱스 로드"""
        try:
            logger.info("🔄 사전 생성된 의약품 인덱스 로딩 중...")
            
            medication_index = None
            medication_metadata = []
            
            # 파일 경로
            index_path = os.path.join(cls.INDEX_DIR, cls.INDEX_FILES["medication"])
            metadata_path = os.path.join(cls.INDEX_DIR, cls.METADATA_FILES["medication"])
            
            # 인덱스 로드
            if os.path.exists(index_path):
                medication_index = faiss.read_index(index_path)
                logger.info("✅ 의약품 인덱스 로드 완료")
            
            # 메타데이터 로드
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    medication_metadata = pickle.load(f)
                logger.info(f"✅ 의약품 메타데이터 로드: {len(medication_metadata)}개")
            
            return medication_index, medication_metadata
            
        except Exception as e:
            logger.error(f"❌ 의약품 인덱스 로드 실패: {e}")
            return None, []
    
    @classmethod
    def get_index_info(cls) -> Dict:
        """인덱스 정보 반환"""
        config_path = os.path.join(cls.INDEX_DIR, cls.CONFIG_FILE)
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"설정 파일 로드 실패: {e}")
        
        return {}

# =============================================================================
# RAG 관련 클래스 - 기존과 동일
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

class OptimizedRAGIndexManager:
    """🚀 최적화된 RAG 인덱스 관리 클래스"""
    
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.qa_index = None
        self.medical_doc_index = None
        self.qa_documents = []
        self.medical_documents = []
        self.use_prebuilt = False
        
    def load_rag_data(self):
        """🚀 RAG 데이터 로딩 - 사전 생성된 인덱스 우선 사용"""
        start_time = datetime.now()
        
        # 1. 사전 생성된 인덱스 사용 시도
        if PreBuiltIndexLoader.check_indexes_available():
            logger.info("🚀 사전 생성된 RAG 인덱스 사용")
            
            qa_index, medical_index, qa_docs, medical_docs = PreBuiltIndexLoader.load_rag_indexes()
            
            if qa_index and medical_index:
                self.qa_index = qa_index
                self.medical_doc_index = medical_index
                self.qa_documents = qa_docs
                self.medical_documents = medical_docs
                self.use_prebuilt = True
                
                load_time = datetime.now() - start_time
                logger.info(f"✅ 사전 생성된 인덱스 로딩 완료! 소요시간: {load_time}")
                logger.info(f"   - Q&A 문서: {len(self.qa_documents)}개")
                logger.info(f"   - 의료 문서: {len(self.medical_documents)}개")
                return
        
        # 2. 백업 모드: 실시간 생성
        logger.info("🔄 백업 모드: RAG 인덱스 실시간 생성")
        self._load_rag_data_realtime()
        
        load_time = datetime.now() - start_time
        logger.info(f"⚠️ 실시간 생성 완료. 소요시간: {load_time}")
        logger.info("💡 다음에는 generate_faiss_indexes.py를 실행하여 인덱스를 사전 생성하세요!")
        
    def _load_rag_data_realtime(self):
        """실시간 RAG 데이터 로딩"""
        logger.info("🔄 RAG 데이터 실시간 로딩 시작...")
        
        # Q&A 데이터 로드 (clean_51004.csv)
        self._load_qa_data()
        
        # 의료 문서 데이터 로드 (나머지 5개 clean_ 파일들)
        self._load_medical_documents()
        
        # 인덱스 구축
        self._build_indexes()
        
        logger.info("✅ RAG 데이터 실시간 로딩 완료!")
        
    def _load_qa_data(self):
        """Q&A 데이터 로드"""
        try:
            if not os.path.exists("clean_51004.csv"):
                logger.warning("⚠️ clean_51004.csv 파일이 없습니다.")
                return
                
            df = pd.read_csv("clean_51004.csv", encoding="utf-8")
            
            for idx, row in df.iterrows():
                try:
                    question = str(row.get('question', ''))
                    answer = str(row.get('answer', ''))
                    
                    if question and answer:
                        content = f"Q: {question}\nA: {answer}"
                        
                        doc = RAGDocument(
                            doc_id=f"qa_{idx}",
                            content=content,
                            metadata={
                                'question': question,
                                'answer': answer,
                                'source': 'clean_51004'
                            },
                            content_type=RAGContentType.QA
                        )
                        self.qa_documents.append(doc)
                        
                except Exception as e:
                    logger.error(f"Q&A 데이터 처리 오류 (행 {idx}): {e}")
                    
            logger.info(f"✅ Q&A 데이터 로드 완료: {len(self.qa_documents)}개")
            
        except Exception as e:
            logger.error(f"❌ Q&A 데이터 로드 실패: {e}")
    
    def _load_medical_documents(self):
        """의료 문서 데이터 로드"""
        clean_files = [
            "clean_55588.csv", "clean_56763.csv", "clean_58572.csv", 
            "clean_63166.csv", "clean_66149.csv"
        ]
        
        for file_path in clean_files:
            try:
                if not os.path.exists(file_path):
                    logger.warning(f"⚠️ {file_path} 파일이 없습니다.")
                    continue
                    
                df = pd.read_csv(file_path, encoding="utf-8")
                
                for idx, row in df.iterrows():
                    try:
                        content_parts = []
                        
                        for col in df.columns:
                            value = str(row.get(col, '')).strip()
                            if value and value != 'nan' and len(value) > 5:
                                content_parts.append(f"{col}: {value}")
                        
                        if content_parts:
                            content = "\n".join(content_parts)
                            
                            doc = RAGDocument(
                                doc_id=f"med_{file_path}_{idx}",
                                content=content,
                                metadata={
                                    'source': file_path,
                                    'original_data': row.to_dict()
                                },
                                content_type=RAGContentType.MEDICAL_DOC
                            )
                            self.medical_documents.append(doc)
                            
                    except Exception as e:
                        logger.error(f"의료 문서 처리 오류 {file_path} (행 {idx}): {e}")
                        
                logger.info(f"✅ {file_path} 로드 완료")
                
            except Exception as e:
                logger.error(f"❌ {file_path} 로드 실패: {e}")
        
        logger.info(f"✅ 의료 문서 로드 완료: {len(self.medical_documents)}개")
    
    def _build_indexes(self):
        """FAISS 인덱스 구축"""
        try:
            # Q&A 인덱스 구축
            if self.qa_documents:
                qa_texts = [doc.content for doc in self.qa_documents]
                qa_embeddings = self.embedding_model.encode(qa_texts)
                faiss.normalize_L2(qa_embeddings)
                
                self.qa_index = faiss.IndexFlatIP(qa_embeddings.shape[1])
                self.qa_index.add(qa_embeddings)
                logger.info(f"✅ Q&A 인덱스 구축 완료: {len(self.qa_documents)}개")
            
            # 의료 문서 인덱스 구축
            if self.medical_documents:
                med_texts = [doc.content for doc in self.medical_documents]
                med_embeddings = self.embedding_model.encode(med_texts)
                faiss.normalize_L2(med_embeddings)
                
                self.medical_doc_index = faiss.IndexFlatIP(med_embeddings.shape[1])
                self.medical_doc_index.add(med_embeddings)
                logger.info(f"✅ 의료 문서 인덱스 구축 완료: {len(self.medical_documents)}개")
                
        except Exception as e:
            logger.error(f"❌ 인덱스 구축 실패: {e}")
    
    def get_relevant_context(self, query: str, top_k: int = 3) -> str:
        """쿼리에 대한 관련 컨텍스트 검색"""
        contexts = []
        
        # Q&A 검색
        qa_results = self.search_qa(query, top_k)
        for doc in qa_results:
            contexts.append(f"[Q&A] {doc.content[:200]}...")
        
        # 의료 문서 검색
        med_results = self.search_medical_docs(query, top_k)
        for doc in med_results:
            contexts.append(f"[의료문서] {doc.content[:200]}...")
        
        return "\n".join(contexts[:5])  # 최대 5개
    
    def search_qa(self, query: str, top_k: int = 3) -> List[RAGDocument]:
        """Q&A 검색"""
        if not self.qa_index or not self.qa_documents:
            return []
        
        try:
            query_embedding = self.embedding_model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            scores, indices = self.qa_index.search(query_embedding, top_k)
            
            results = []
            for idx in indices[0]:
                if 0 <= idx < len(self.qa_documents):
                    results.append(self.qa_documents[idx])
            
            return results
        except Exception as e:
            logger.error(f"Q&A 검색 오류: {e}")
            return []
    
    def search_medical_docs(self, query: str, top_k: int = 3) -> List[RAGDocument]:
        """의료 문서 검색"""
        if not self.medical_doc_index or not self.medical_documents:
            return []
        
        try:
            query_embedding = self.embedding_model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            scores, indices = self.medical_doc_index.search(query_embedding, top_k)
            
            results = []
            for idx in indices[0]:
                if 0 <= idx < len(self.medical_documents):
                    results.append(self.medical_documents[idx])
            
            return results
        except Exception as e:
            logger.error(f"의료 문서 검색 오류: {e}")
            return []

# =============================================================================
# 임베딩 모델 클래스 - 기존과 동일
# =============================================================================

class EmbeddingModel:
    """KM-BERT 임베딩 모델 클래스"""
    
    def __init__(self, model_name: str = "madatnlp/km-bert"):
        logger.info(f"🔄 임베딩 모델 로딩 중: {model_name}")
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        logger.info(f"✅ KM-BERT 임베딩 모델 로드 완료 (Device: {self.device})")

    def encode(self, texts: List[str]) -> np.ndarray:
        """텍스트를 벡터로 인코딩"""
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
# EXAONE LLM 클래스 - 기존과 동일
# =============================================================================

class EXAONE:
    """EXAONE LLM 서비스 클래스"""
    
    def __init__(self, model_name: str = "exaone3.5:7.8b"):
        self.model_name = model_name
        self.base_url = "http://localhost:11434"
        self.endpoint = None
        
        self.exaone_config = {
            "temperature": 0.3,
            "top_p": 0.8,
            "max_tokens": 2048,
            "repeat_penalty": 1.1
        }
        
        logger.info(f"🔧 EXAONE 초기화: {model_name}")
        
        if self._check_endpoint("chat"):
            self.endpoint = "chat"
            logger.info("✅ EXAONE chat 엔드포인트 사용")
        elif self._check_endpoint("generate"):
            self.endpoint = "generate"
            logger.info("✅ EXAONE generate 엔드포인트 사용")
        else:
            logger.warning("⚠️ EXAONE 서버 연결 실패. 기본 응답 모드로 실행합니다.")

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

# =============================================================================
# 기타 로딩 함수들 - 기존과 동일
# =============================================================================

def detect_columns(df: pd.DataFrame, data_type: str) -> Dict[str, str]:
    """데이터 타입별 컬럼 감지"""
    columns = df.columns.tolist()
    detected = {}
    
    if data_type == "disease":
        column_mappings = {
            'disease_name': ['disnm_ko', 'disnm_en', 'disease', '질병', '병명', 'disease_name'],
            'symptoms': ['sym', 'symptoms', '증상', 'symptom'],
            'symptoms_key': ['sym_k', 'symptoms_key', '핵심증상', 'key_symptoms']
        }
        
        for target_type, possible_names in column_mappings.items():
            for col in columns:
                if col in possible_names:
                    detected[target_type] = col
                    break
                col_lower = col.lower()
                for possible in possible_names:
                    if possible.lower() in col_lower or col_lower in possible.lower():
                        detected[target_type] = col
                        break
                if target_type in detected:
                    break
    
    elif data_type == "medication":
        for col in columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['item', '품목', '약품명']):
                detected['itemName'] = col
            elif any(keyword in col_lower for keyword in ['efcy', '효능', '효과']):
                detected['efcyQesitm'] = col
    
    return detected

def discover_csv_files() -> Tuple[List[str], List[str]]:
    """CSV 파일 자동 탐지"""
    files = [f for f in os.listdir('.') if f.lower().endswith(".csv")]
    
    disease_files = []
    medication_files = []

    for fname in files:
        try:
            df = pd.read_csv(fname, encoding="utf-8", low_memory=False, nrows=5)
        except Exception:
            continue

        d_cols = detect_columns(df, "disease")
        if "disease_name" in d_cols and (d_cols.get("symptoms") or d_cols.get("symptoms_key")):
            disease_files.append(fname)
            logger.info(f"🏥 질병 파일 발견: {fname} (컬럼: {d_cols})")
            continue

        m_cols = detect_columns(df, "medication")
        if "itemName" in m_cols and "efcyQesitm" in m_cols:
            medication_files.append(fname)
            logger.info(f"💊 의약품 파일 발견: {fname} (컬럼: {m_cols})")
            continue

    return disease_files, medication_files

def optimized_load_disease_indexes(
    csv_paths: List[str],
    embedding_model: EmbeddingModel
) -> Tuple[faiss.IndexFlatIP, faiss.IndexFlatIP, List[Dict]]:
    """🚀 최적화된 질병 데이터 로드"""
    
    # 1. 사전 생성된 인덱스 사용 시도
    if PreBuiltIndexLoader.check_indexes_available():
        logger.info("🚀 사전 생성된 질병 인덱스 사용")
        
        disease_key_index, disease_full_index, disease_metadata = PreBuiltIndexLoader.load_disease_indexes()
        
        if disease_key_index and disease_full_index and disease_metadata:
            logger.info(f"✅ 질병 인덱스 로딩 완료: {len(disease_metadata)}개")
            return disease_key_index, disease_full_index, disease_metadata
    
    # 2. 백업 모드: 실시간 생성
    logger.info("🔄 백업 모드: 질병 인덱스 실시간 생성")
    return load_and_build_disease_indexes(csv_paths, embedding_model)

def optimized_load_medication_index(
    csv_paths: List[str],
    embedding_model: EmbeddingModel
) -> Tuple[faiss.IndexFlatIP, List[Dict]]:
    """🚀 최적화된 의약품 데이터 로드"""
    
    # 1. 사전 생성된 인덱스 사용 시도
    if PreBuiltIndexLoader.check_indexes_available():
        logger.info("🚀 사전 생성된 의약품 인덱스 사용")
        
        medication_index, medication_metadata = PreBuiltIndexLoader.load_medication_index()
        
        if medication_index and medication_metadata:
            logger.info(f"✅ 의약품 인덱스 로딩 완료: {len(medication_metadata)}개")
            return medication_index, medication_metadata
    
    # 2. 백업 모드: 실시간 생성
    logger.info("🔄 백업 모드: 의약품 인덱스 실시간 생성")
    return load_and_build_medication_index(csv_paths, embedding_model)

def load_and_build_disease_indexes(
    csv_paths: List[str],
    embedding_model: EmbeddingModel
) -> Tuple[faiss.IndexFlatIP, faiss.IndexFlatIP, List[Dict]]:
    """질병 데이터 로드 및 FAISS 인덱스 구축"""
    
    all_key_embs = []
    all_full_embs = []
    all_docs_meta = []

    for path in csv_paths:
        logger.info(f"📂 질병 데이터 로드 중: {path}")
        
        try:
            df = pd.read_csv(path, encoding="utf-8", low_memory=False)
            logger.info(f"🔍 {path} 컬럼 확인: {list(df.columns)}")
            
            detected = detect_columns(df, "disease")
            logger.info(f"🔍 감지된 질병 컬럼: {detected}")
            
            disease_col = detected.get("disease_name")
            symptoms_col = detected.get("symptoms")
            symptoms_key_col = detected.get("symptoms_key")
            
            if not disease_col:
                logger.warning(f"⚠️ {path}: 질병명 컬럼을 찾을 수 없습니다.")
                continue
            
            valid_rows = 0
            for _, row in df.iterrows():
                try:
                    disease_name = str(row.get(disease_col, "")).strip()
                    if not disease_name or disease_name == "nan":
                        continue
                    
                    # 증상 정보 수집
                    symptoms_full = ""
                    symptoms_key = ""
                    
                    if symptoms_col:
                        symptoms_full = str(row.get(symptoms_col, "")).strip()
                    if symptoms_key_col:
                        symptoms_key = str(row.get(symptoms_key_col, "")).strip()
                    
                    # 기타 컬럼들도 수집
                    additional_info = []
                    for col in ['def', 'therapy', 'diag', 'guide', 'pvt']:
                        if col in df.columns:
                            value = str(row.get(col, "")).strip()
                            if value and value != "nan":
                                additional_info.append(value)
                    
                    # 메타데이터 구성
                    doc_meta = {
                        "disease": disease_name,
                        "symptoms": symptoms_full,
                        "symptoms_key": symptoms_key,
                        "additional_info": " ".join(additional_info),
                        "source_file": path,
                        "original_data": row.to_dict()
                    }
                    
                    # 임베딩용 텍스트 구성
                    key_text = f"{disease_name} {symptoms_key}".strip()
                    full_text = f"{disease_name} {symptoms_full} {symptoms_key} {' '.join(additional_info)}".strip()
                    
                    all_docs_meta.append(doc_meta)
                    all_key_embs.append(key_text)
                    all_full_embs.append(full_text)
                    valid_rows += 1
                    
                except Exception as e:
                    logger.error(f"행 처리 오류 {path}: {e}")
                    continue
            
            logger.info(f"✅ {path}: {valid_rows}개 질병 데이터 로드 완료")
            
        except Exception as e:
            logger.error(f"❌ {path} 로드 실패: {e}")
            continue

    if not all_docs_meta:
        raise ValueError("유효한 질병 데이터가 없습니다.")

    logger.info(f"🔄 질병 임베딩 생성 중: {len(all_docs_meta)}건")

    # 임베딩 생성
    key_embeddings = embedding_model.encode(all_key_embs)
    full_embeddings = embedding_model.encode(all_full_embs)

    # FAISS 인덱스 구축  
    faiss.normalize_L2(key_embeddings)
    faiss.normalize_L2(full_embeddings)
    
    index_key = faiss.IndexFlatIP(key_embeddings.shape[1])
    index_full = faiss.IndexFlatIP(full_embeddings.shape[1])
    
    index_key.add(key_embeddings)
    index_full.add(full_embeddings)

    logger.info(f"✅ 질병 FAISS 인덱스 구축 완료: {len(all_docs_meta)}건")
    return index_key, index_full, all_docs_meta

def load_and_build_medication_index(
    csv_paths: List[str],
    embedding_model: EmbeddingModel
) -> Tuple[faiss.IndexFlatIP, List[Dict]]:
    """의약품 데이터 로드 및 FAISS 인덱스 구축"""
    
    all_med_docs = []
    all_med_meta = []

    for path in csv_paths:
        logger.info(f"📂 의약품 데이터 로드 중: {path}")
        
        try:
            df = pd.read_csv(path, encoding="utf-8", low_memory=False)
            detected = detect_columns(df, "medication")
            
            item_col = detected.get("itemName")
            efcy_col = detected.get("efcyQesitm")
            
            if not item_col or not efcy_col:
                logger.warning(f"⚠️ {path}: 필요한 의약품 컬럼을 찾을 수 없습니다.")
                continue
            
            valid_rows = 0
            for _, row in df.iterrows():
                try:
                    item_name = str(row.get(item_col, "")).strip()
                    efficacy = str(row.get(efcy_col, "")).strip()
                    
                    if not item_name or not efficacy or item_name == "nan" or efficacy == "nan":
                        continue
                    
                    # 문서 텍스트 구성
                    doc_text = f"의약품명: {item_name}\n효능: {efficacy}"
                    
                    # 메타데이터 구성
                    doc_meta = {
                        "name": item_name,
                        "efficacy": efficacy,
                        "doc_text": doc_text,
                        "source_file": path,
                        "original_data": row.to_dict()
                    }
                    
                    all_med_docs.append(doc_text)
                    all_med_meta.append(doc_meta)
                    valid_rows += 1
                    
                except Exception as e:
                    logger.error(f"의약품 행 처리 오류 {path}: {e}")
                    continue
            
            logger.info(f"✅ {path}: {valid_rows}개 의약품 데이터 로드 완료")
            
        except Exception as e:
            logger.error(f"❌ {path} 로드 실패: {e}")
            continue

    if not all_med_docs:
        logger.warning("⚠️ 유효한 의약품 데이터가 없습니다.")
        return None, []

    logger.info(f"🔄 의약품 임베딩 생성 중: {len(all_med_docs)}건")

    # 임베딩 생성
    med_embeddings = embedding_model.encode(all_med_docs)

    # FAISS 인덱스 구축  
    faiss.normalize_L2(med_embeddings)
    med_index = faiss.IndexFlatIP(med_embeddings.shape[1])
    med_index.add(med_embeddings)

    logger.info(f"✅ 의약품 FAISS 인덱스 구축 완료: {len(all_med_docs)}건")
    return med_index, all_med_meta

# =============================================================================
# 세션 관리 클래스 - 기존과 동일
# =============================================================================

class IntegratedSession:
    """🔥 완전한 통합 세션 관리 클래스"""
    
    def __init__(self):
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.history = []
        self.context = {
            "last_intent": None,
            "last_entity": None, 
            "last_disease": None,
            "last_final_diagnosis": None,
            "diagnosed_diseases": [],
            "detected_symptoms": [],
            "diagnosis_time": None,
            "disease_symptoms_mapping": {},
            "last_medications": [],
            "recommended_medications": [],
            "medication_queries": [],
            "user_medication_profile": {
                "age_group": None,
                "is_pregnant": None,
                "chronic_conditions": []
            },
            # 🔥 수정된 차별화 질문 상태
            "questioning_state": {
                "is_questioning": False,
                "current_candidates": [],
                "asked_questions": [],
                "user_answers": {},
                "current_question_index": 0,
                "max_questions": 3
            },
            # 🔥 새로 추가된 필드들
            "initial_symptoms_text": "",  # 초기 증상 텍스트 저장
            "mentioned_symptoms": []       # 사용자가 언급한 증상들
        }
    
    def reset_session(self):
        """세션 초기화"""
        old_session_id = self.session_id
        self.__init__()
        logger.info(f"🔄 세션 초기화: {old_session_id} → {self.session_id}")
    
    def add_message(self, user_message: str, bot_response: str, intent: str):
        """대화 히스토리에 메시지 추가"""
        self.history.append({
            "timestamp": datetime.now(),
            "user_message": user_message,
            "bot_response": bot_response,
            "intent": intent
        })
        
        if len(self.history) > 50:
            self.history = self.history[-50:]
    
    def get_recent_diagnosis(self) -> Optional[str]:
        """최근 진단된 질병 반환 (30분 이내)"""
        if not self.context["last_final_diagnosis"] or not self.context["diagnosis_time"]:
            return None
        
        time_diff = datetime.now() - self.context["diagnosis_time"]
        if time_diff > timedelta(minutes=30):
            return None
        
        return self.context["last_final_diagnosis"]
    
    def update_disease_context(self, disease: str, symptoms: List[str], intent: str):
        """질병 컨텍스트 업데이트"""
        self.context["last_disease"] = disease
        self.context["last_intent"] = intent
        self.context["diagnosis_time"] = datetime.now()
        
        if disease:
            self.context["diagnosed_diseases"].append({
                "disease": disease,
                "symptoms": symptoms,
                "timestamp": datetime.now()
            })
        
        if symptoms:
            self.context["detected_symptoms"].extend(symptoms)
            self.context["disease_symptoms_mapping"][disease] = symptoms
    
    def get_disease_symptoms(self, disease: str) -> List[str]:
        """특정 질병의 증상 목록 반환"""
        return self.context["disease_symptoms_mapping"].get(disease, [])
    
    def update_medication_context(self, medications: List[Dict]):
        """의약품 컨텍스트 업데이트"""
        self.context["last_medications"] = medications
        self.context["recommended_medications"].extend(medications)
        
        if len(self.context["recommended_medications"]) > 10:
            self.context["recommended_medications"] = self.context["recommended_medications"][-10:]
    
    def start_questioning(self, candidates: List[Dict], questions: List[str]):
        """차별화 질문 모드 시작"""
        state = self.context["questioning_state"]
        state["is_questioning"] = True
        state["current_candidates"] = candidates
        state["asked_questions"] = questions
        state["user_answers"] = {}
        state["current_question_index"] = 0
        logger.info(f"🔬 차별화 질문 모드 시작: {len(candidates)}개 후보, {len(questions)}개 질문")
    
    def get_current_question(self) -> Optional[str]:
        """현재 질문 반환"""
        state = self.context["questioning_state"]
        if (state["is_questioning"] and 
            state["current_question_index"] < len(state["asked_questions"])):
            return state["asked_questions"][state["current_question_index"]]
        return None
    
    def add_answer(self, question: str, answer: str):
        """질문에 대한 답변 추가"""
        self.context["questioning_state"]["user_answers"][question] = answer
    
    def next_question(self):
        """다음 질문으로 이동"""
        self.context["questioning_state"]["current_question_index"] += 1
    
    def should_continue_questioning(self) -> bool:
        """질문을 계속해야 하는지 판단"""
        state = self.context["questioning_state"]
        return (state["is_questioning"] and 
                state["current_question_index"] < len(state["asked_questions"]) and
                state["current_question_index"] < state["max_questions"])
    
    def finish_questioning(self):
        """질문 모드 종료"""
        self.context["questioning_state"]["is_questioning"] = False
        logger.info("🔬 차별화 질문 모드 종료")

# =============================================================================
# 🔥 수정된 강화된 질병 서비스 클래스
# =============================================================================

class EnhancedDiseaseService:
    """🔥 수정된 강화된 질병 서비스 - 스마트한 차별화 질문"""
    
    def __init__(self, exaone_service, disease_index_key, disease_index_full, 
                 disease_meta, embedding_model, rag_manager):
        self.exaone = exaone_service
        self.disease_index_key = disease_index_key
        self.disease_index_full = disease_index_full
        self.disease_meta = disease_meta
        self.embedding_model = embedding_model
        self.rag_manager = rag_manager
        
        # 🔥 증상 키워드 매핑 (개선)
        self.symptom_keywords = {
            "발열": ["열", "발열", "체온", "뜨거워", "후끈", "미열", "고열"],
            "두통": ["머리", "두통", "머리아", "정수리", "관자놀이", "머리가아"],
            "기침": ["기침", "켁켁", "컴컴", "마른기침", "가래기침"],
            "콧물": ["콧물", "코막힘", "코가막", "재채기", "코감기"],
            "인후통": ["목아", "목이아", "인후통", "목따가", "삼키기", "목이따"],
            "근육통": ["몸살", "근육통", "온몸", "아프다", "쑤시", "결림"],
            "피로": ["피로", "무기력", "기운없", "졸려", "힘들어"]
        }
        
        # 🔥 감기/코로나 우선 진단을 위한 증상 가중치
        self.symptom_weights = {
            "기침": 0.3, "콧물": 0.25, "인후통": 0.25, "발열": 0.35,
            "두통": 0.2, "근육통": 0.2, "피로": 0.15
        }
        
    def extract_mentioned_symptoms(self, symptoms_text: str) -> Set[str]:
        """🔥 새로 추가: 사용자가 이미 언급한 증상들 추출"""
        symptoms_lower = symptoms_text.lower()
        mentioned_symptoms = set()
        
        for symptom_type, keywords in self.symptom_keywords.items():
            for keyword in keywords:
                if keyword in symptoms_lower:
                    mentioned_symptoms.add(symptom_type)
                    break
        
        logger.info(f"🔍 감지된 기존 증상: {mentioned_symptoms}")
        return mentioned_symptoms
        
    def diagnose_disease(self, symptoms_text: str, session: IntegratedSession) -> str:
        """🔥 수정된 질병 진단 - 개선된 차별화 질문"""
        
        # 🔥 초기 증상을 세션에 저장 (중요!)
        session.context["initial_symptoms_text"] = symptoms_text
        session.context["mentioned_symptoms"] = list(self.extract_mentioned_symptoms(symptoms_text))
        
        # 1. 벡터 검색으로 유사 질병 찾기
        similar_diseases = self._search_similar_diseases(symptoms_text, top_k=8)
        
        # 2. 증상 기반 가중치 계산 및 재정렬
        weighted_diseases = self._apply_symptom_weights(symptoms_text, similar_diseases)
        
        # 3. 감기/코로나 우선 진단 로직
        prioritized_diseases = self._prioritize_common_diseases(symptoms_text, weighted_diseases)
        
        # 4. 차별화 질문 필요성 판단 (조건 완화)
        if len(prioritized_diseases) > 1 and prioritized_diseases[0].get('confidence', 0) < 0.85:
            # 🔥 스마트한 질문 생성 (이미 알고 있는 증상 제외)
            questions = self._generate_smart_differential_questions(symptoms_text, prioritized_diseases)
            if questions:
                session.start_questioning(prioritized_diseases, questions)
                first_question = session.get_current_question()
                
                return f"""🔍 **초기 분석 결과**:
증상을 분석한 결과, {len(prioritized_diseases)}가지 질병 가능성이 있습니다:
{self._format_disease_list(prioritized_diseases[:3])}

📚 **관련 의료 정보**:
{self._get_rag_context(symptoms_text)}

🔬 **정확한 진단을 위한 추가 질문**:
{first_question}

💡 위 질문에 답변해주시면 더 정확한 진단을 도와드릴 수 있습니다."""
        
        # 5. 최종 진단 생성 (차별화 질문 없이 바로 진단)
        return self._generate_enhanced_final_diagnosis(prioritized_diseases[:3], symptoms_text, session)
    
    def _generate_smart_differential_questions(self, symptoms_text: str, diseases: List[Dict]) -> List[str]:
        """🔥 스마트한 차별화 질문 생성 - 이미 알고 있는 증상은 제외"""
        mentioned_symptoms = self.extract_mentioned_symptoms(symptoms_text)
        
        # 가능한 모든 질문들
        all_questions = [
            ("발열", "현재 발열(열)이 있으신가요?"),
            ("기침", "기침이나 가래 증상이 있으신가요?"),
            ("인후통", "목의 통증이나 따가움이 있으신가요?"),
            ("근육통", "근육통이나 몸살 기운이 있으신가요?"),
            ("콧물", "코막힘이나 콧물 증상이 있으신가요?"),
            ("두통", "두통이나 머리 아픈 증상이 있으신가요?"),
            ("피로", "피로감이나 무기력함을 느끼시나요?")
        ]
        
        # 🔥 이미 언급한 증상은 질문에서 제외
        smart_questions = []
        for symptom_type, question in all_questions:
            if symptom_type not in mentioned_symptoms:
                smart_questions.append(question)
        
        # 최대 3개 질문만 선택
        selected_questions = smart_questions[:3]
        
        logger.info(f"🔥 스마트 질문 생성: 기존 증상 {mentioned_symptoms} 제외, {len(selected_questions)}개 질문 생성")
        return selected_questions
    
    def process_followup_answer(self, answer: str, session: IntegratedSession) -> str:
        """🔥 개선된 후속 답변 처리"""
        current_question = session.get_current_question()
        if not current_question:
            return "질문 처리 중 오류가 발생했습니다."
        
        # 답변 기록
        session.add_answer(current_question, answer)
        
        # 🔥 개선된 후보 필터링
        state = session.context["questioning_state"]
        filtered_candidates = self._improved_filter_candidates(
            state["current_candidates"], current_question, answer, session
        )
        
        # 🔥 빈 결과 방지 - 필터링이 너무 강하면 원래 후보 유지
        if len(filtered_candidates) == 0:
            logger.warning("⚠️ 필터링 결과가 비어있음 - 원래 후보 유지")
            filtered_candidates = state["current_candidates"]
        
        state["current_candidates"] = filtered_candidates
        
        # 다음 질문으로 이동
        session.next_question()
        
        # 🔥 종료 조건 개선
        should_finish = (
            not session.should_continue_questioning() or 
            len(filtered_candidates) <= 1 or
            (len(filtered_candidates) <= 2 and state["current_question_index"] >= 2)
        )
        
        if should_finish:
            session.finish_questioning()
            # 🔥 초기 증상 정보를 함께 전달
            initial_symptoms = session.context.get("initial_symptoms_text", "")
            return self._generate_enhanced_final_diagnosis(filtered_candidates, initial_symptoms, session)
        
        # 다음 질문 계속
        next_question = session.get_current_question()
        return f"✅ 답변 감사합니다.\n\n❓ {next_question}"
    
    def _improved_filter_candidates(self, candidates: List[Dict], question: str, answer: str, session: IntegratedSession) -> List[Dict]:
        """🔥 개선된 후보 필터링 로직"""
        answer_lower = answer.lower()
        is_positive = any(word in answer_lower for word in ["예", "있", "네", "맞", "그래", "심해", "많이"])
        is_negative = any(word in answer_lower for word in ["아니", "없", "안", "아직", "별로"])
        
        # 🔥 질문 유형별 정교한 필터링
        if "발열" in question or "열" in question:
            if is_positive:
                # 발열이 있으면 감기/독감/코로나 관련 질병 우선
                fever_diseases = []
                other_diseases = []
                
                for candidate in candidates:
                    disease_name = candidate.get('disease', '').lower()
                    symptoms = candidate.get('symptoms', '').lower()
                    
                    if any(keyword in disease_name or keyword in symptoms 
                          for keyword in ['감기', '독감', '인플루엔자', '상기도', '발열', '열']):
                        candidate['confidence'] = candidate.get('confidence', 0) + 0.2
                        fever_diseases.append(candidate)
                    else:
                        other_diseases.append(candidate)
                
                return fever_diseases + other_diseases
        
        elif "기침" in question:
            if is_positive:
                # 기침이 있으면 호흡기 질환 우선
                cough_diseases = [c for c in candidates 
                                if any(keyword in c.get('disease', '').lower() or keyword in c.get('symptoms', '').lower()
                                      for keyword in ['기침', '호흡', '폐', '기관지', '천식'])]
                other_diseases = [c for c in candidates if c not in cough_diseases]
                return cough_diseases + other_diseases
        
        elif "목" in question or "인후통" in question:
            if is_positive:
                # 인후통이 있으면 상기도 감염 우선
                throat_diseases = [c for c in candidates 
                                 if any(keyword in c.get('disease', '').lower() or keyword in c.get('symptoms', '').lower()
                                       for keyword in ['인후', '목', '상기도', '편도'])]
                other_diseases = [c for c in candidates if c not in throat_diseases]
                return throat_diseases + other_diseases
        
        # 🔥 기본적으로는 모든 후보 유지 (너무 강한 필터링 방지)
        return candidates
    
    def _apply_symptom_weights(self, symptoms_text: str, diseases: List[Dict]) -> List[Dict]:
        """증상 가중치 적용하여 질병 재정렬"""
        symptoms_lower = symptoms_text.lower()
        
        for disease in diseases:
            base_score = disease.get('score', 0)
            symptom_boost = 0
            
            # 핵심 증상 매칭 계산
            disease_symptoms = disease.get('symptoms', '').lower()
            
            for symptom, weight in self.symptom_weights.items():
                if symptom in symptoms_lower and symptom in disease_symptoms:
                    symptom_boost += weight
            
            # 최종 점수 계산
            disease['confidence'] = min(base_score + symptom_boost, 1.0)
        
        # 신뢰도 기준으로 재정렬
        return sorted(diseases, key=lambda x: x.get('confidence', 0), reverse=True)
    
    def _prioritize_common_diseases(self, symptoms_text: str, diseases: List[Dict]) -> List[Dict]:
        """🔥 감기/코로나 등 일반적 질병 우선 진단"""
        symptoms_lower = symptoms_text.lower()
        
        # 감기/독감 관련 증상 체크
        cold_symptoms = ["기침", "콧물", "인후통", "발열", "두통", "열", "머리"]
        cold_match_count = sum(1 for symptom in cold_symptoms if symptom in symptoms_lower)
        
        # 감기 증상이 2개 이상이면 감기/독감을 상위로 올림
        if cold_match_count >= 2:
            cold_diseases = []
            other_diseases = []
            
            for disease in diseases:
                disease_name = disease.get('disease', '').lower()
                if any(keyword in disease_name for keyword in ['감기', '독감', '인플루엔자', '상기도','상기도 감염','만성 기침','냉방병','급성기관지염','폐렴', '기관지염','인후염', '급성 상기도 감염']):
                    # 감기/독감 질병에 추가 가중치
                    disease['confidence'] = disease.get('confidence', 0) + 0.3
                    cold_diseases.append(disease)
                else:
                    other_diseases.append(disease)
            
            # 감기 질병을 앞으로 정렬
            prioritized = sorted(cold_diseases, key=lambda x: x.get('confidence', 0), reverse=True)
            prioritized.extend(sorted(other_diseases, key=lambda x: x.get('confidence', 0), reverse=True))
            
            logger.info(f"🔥 감기 증상 {cold_match_count}개 감지 - 감기/독감 우선 진단")
            return prioritized
        
        return diseases
    
    def _generate_enhanced_final_diagnosis(self, candidates: List[Dict], symptoms_text: str, session: IntegratedSession) -> str:
        """🔥 강화된 최종 진단 - 초기 증상 정보 활용"""
        
        if not candidates:
            return "⚠️ 관련된 질병을 찾을 수 없습니다. 의료 전문가에게 상담을 받으시기 바랍니다."
        
        top_candidate = candidates[0]
        disease_name = top_candidate.get('disease', '')
        confidence = top_candidate.get('confidence', 0)
        
        # 🔥 초기 증상 정보 활용
        initial_symptoms = session.context.get("initial_symptoms_text", symptoms_text)
        
        # 🔥 코로나19 주의사항 판단
        covid_similar_diseases = ['감기', '상기도', '독감', '인플루엔자', '기관지염', '폐렴','만성 기침','냉방병','급성기관지염','폐렴', '기관지염','인후염', '급성 상기도 감염']
        needs_covid_warning = any(keyword in disease_name.lower() for keyword in covid_similar_diseases)
        
        # 🔥 강화된 진단 프롬프트
        enhanced_prompt = f"""당신은 전문 의료진입니다. 다음 정보를 바탕으로 정확한 진단을 내려주세요.

환자 증상: {initial_symptoms}
가장 가능성 높은 질병: {disease_name} (신뢰도: {confidence:.2f})
기타 후보: {', '.join([c.get('disease', '') for c in candidates[1:3]])}

다음 형식으로 답변해주세요:

🎯 **최종 진단**: {disease_name}

📋 **진단 근거**:
- 환자분께서 보고하신 증상들이 이 질병의 특징적인 증상과 일치합니다.
- [구체적인 증상 매칭 설명]

📖 **질병 설명**:
- [질병에 대한 간단하고 이해하기 쉬운 설명]

💡 **관리 방법**:
- [생활 관리 및 대처 방법]

🏥 **병원 진료 권유**:
- [병원 방문 권유 및 응급상황 판단]

**참고**: 이 진단은 증상 분석을 바탕으로 한 참고용 정보입니다."""

        # 🔥 코로나19 주의사항 추가
        if needs_covid_warning:
            enhanced_prompt += f"""

🚨 **COVID-19 감별 주의사항** 🚨
진단된 질병의 증상은 COVID-19와 유사할 수 있습니다.
- 가능하면 코로나19 검사를 받으시기 바랍니다
- 타인과의 접촉을 최소화하고 마스크를 착용하세요
- 호흡곤란이나 고열 지속 시 즉시 병원에 방문하세요"""

        try:
            response = self.exaone.generate_response(enhanced_prompt)
            
            # 세션 컨텍스트 업데이트 (진단된 질병과 사용자 증상 저장)
            session.context["last_final_diagnosis"] = disease_name
            session.update_disease_context(disease_name, [initial_symptoms], "disease_diagnosis")
            logger.info(f"🏥 강화된 진단 완료: {disease_name} (신뢰도: {confidence:.2f})")
            
            return response
            
        except Exception as e:
            logger.error(f"강화된 진단 생성 오류: {e}")
            return f"⚠️ 진단 생성 중 오류가 발생했습니다: {str(e)}"
    
    # 나머지 메서드들은 기존과 동일...
    def _search_similar_diseases(self, query: str, top_k: int = 8) -> List[Dict]:
        """벡터 검색으로 유사한 질병 찾기"""
        try:
            query_embedding = self.embedding_model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            scores, indices = self.disease_index_full.search(query_embedding, top_k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if 0 <= idx < len(self.disease_meta):
                    disease_data = self.disease_meta[idx].copy()
                    disease_data['score'] = float(scores[0][i])
                    results.append(disease_data)
            
            return results
        except Exception as e:
            logger.error(f"질병 벡터 검색 오류: {e}")
            return []
    
    def _get_rag_context(self, query: str) -> str:
        """RAG 컨텍스트 가져오기"""
        try:
            if self.rag_manager:
                context = self.rag_manager.get_relevant_context(query, top_k=2)
                return context[:200] + "..."
            return "추가 의료 정보를 찾을 수 없습니다."
        except Exception as e:
            logger.error(f"RAG 컨텍스트 오류: {e}")
            return "관련 의료 정보를 찾을 수 없습니다."
    
    def _format_disease_list(self, diseases: List[Dict]) -> str:
        """질병 목록 포맷팅"""
        formatted = []
        for i, disease in enumerate(diseases[:3], 1):
            name = disease.get('disease', '알 수 없음')
            confidence = disease.get('confidence', 0)
            formatted.append(f"{i}. {name} (신뢰도: {confidence:.2f})")
        return "\n".join(formatted)
    
    def get_disease_info(self, disease_query: str) -> str:
        """질병 정보 검색"""
        try:
            rag_context = self._get_rag_context(disease_query)
            similar_diseases = self._search_similar_diseases(disease_query, top_k=3)
            
            system_prompt = """당신은 의료 정보 전문가입니다. 
질병에 대한 정확하고 이해하기 쉬운 정보를 제공해주세요."""
            
            prompt = f"""질병명: {disease_query}

관련 의료 정보:
{rag_context}

다음 형식으로 답변해주세요:

📖 **질병 개요**:
- [질병에 대한 간단한 설명]

🔍 **주요 증상**:
- [대표적인 증상들]

🎯 **원인**:
- [발생 원인]

💡 **예방 및 관리**:
- [예방 방법과 관리법]

🏥 **치료**:
- [치료 방법]

**참고**: 정확한 진단과 치료는 의료진과 상담하세요."""
            
            response = self.exaone.generate_response(prompt, system_prompt)
            return response
            
        except Exception as e:
            logger.error(f"질병 정보 검색 오류: {e}")
            return f"⚠️ 질병 정보 검색 중 오류가 발생했습니다: {str(e)}"

# =============================================================================
# 나머지 클래스들 - 기존과 동일
# =============================================================================

class EnhancedIntentClassifier:
    """강화된 의도 분류기"""
    
    def __init__(self, exaone_service):
        self.exaone = exaone_service
        
        self.intent_categories = {
            "symptom_medication": [
                "아픈데 약", "아프면 약", "무슨 약", "어떤 약", "약 추천", 
                "먹어야", "복용", "치료제", "두통약", "감기약", "해열제",
                "머리 아픈데", "배 아픈데", "열 나는데", "기침 나는데"
            ],
            "disease_diagnosis": [
                "진단", "병", "질병", "무슨 병", "어떤 병", "증상",
                "아프다", "아픈", "통증", "열", "기침", "감기"
            ],
            "disease_info": [
                "질병에 대해", "병에 대해", "원인", "치료법", "예방법", 
                "알려줘", "설명", "이란", "무엇"
            ],
            "medication_recommend": [
                "약 추천", "처방", "의약품", "치료제", "약물"
            ],
            "medication_info": [
                "약 정보", "부작용", "효능", "성분", "용법", "용량", 
                "타이레놀", "게보린", "낙센"
            ],
            "disease_to_medication": [
                "약", "처방", "치료제", "먹으면", "복용"
            ],
            "reset": [
                "처음으로", "처음부터", "다시", "리셋", "reset", 
                "초기화", "새로 시작", "그만"
            ],
            "general": [
                "안녕", "감사", "고마워", "bye", "안녕하세요"
            ]
        }
        
        self.compound_patterns = [
            r'([가-힣]+(?:아프|아픈|통증)).*(?:약|약품|먹어야|복용)',
            r'(?:머리|두통|배|복통|열|기침).*(?:약|약품|치료제)',
            r'(?:감기|독감|몸살).*(?:약|약품|먹으면)',
            r'([가-힣]+(?:나는데|나서)).*(?:약|약품|어떤)'
        ]
    
    def classify_intent(self, message: str, session: IntegratedSession) -> str:
        """강화된 의도 분류"""
        message_lower = message.lower()
        
        # 1. 차별화 질문 모드 우선 확인
        if session.context["questioning_state"]["is_questioning"]:
            logger.info("🔬 차별화 질문 응답 모드")
            return "diagnosis_followup"
        
        # 2. 세션 초기화 요청 우선 처리
        if any(keyword in message_lower for keyword in self.intent_categories["reset"]):
            logger.info("🔄 세션 초기화 요청 감지")
            return "reset"
        
        # 3. 복합 의도 패턴 검사
        for pattern in self.compound_patterns:
            if re.search(pattern, message):
                logger.info(f"💊 복합 의도 감지: 증상 기반 의약품 추천 - {message}")
                return "symptom_medication"
        
        # 4. 증상 키워드 + 약품 키워드 동시 존재 검사
        symptom_words = ["아프", "아픈", "통증", "열", "기침", "두통", "복통", "머리", "배"]
        medication_words = ["약", "약품", "먹어야", "복용", "치료제", "처방"]
        
        has_symptom = any(word in message_lower for word in symptom_words)
        has_medication = any(word in message_lower for word in medication_words)
        
        if has_symptom and has_medication:
            logger.info(f"💊 증상+약품 키워드 동시 감지 - {message}")
            return "symptom_medication"
        
        # 5. 세션 컨텍스트 기반 질병-의약품 연계 검사
        recent_disease = session.get_recent_diagnosis()
        if recent_disease:
            medication_keywords = ["약", "약품", "처방", "추천", "복용", "먹어야", "치료제"]
            if any(keyword in message_lower for keyword in medication_keywords):
                logger.info(f"🔗 질병-의약품 연계 의도: {recent_disease} -> 약품")
                return "disease_to_medication"
        
        # 6. 기본 키워드 매칭
        for intent, keywords in self.intent_categories.items():
            if intent in ["symptom_medication", "disease_to_medication", "reset"]:
                continue
                
            if any(keyword in message_lower for keyword in keywords):
                return intent
        
        # 7. EXAONE 기반 고급 분류
        return self._classify_with_exaone(message)
    
    def _classify_with_exaone(self, message: str) -> str:
        """EXAONE 기반 고급 의도 분류"""
        system_prompt = """당신은 의료 챗봇의 의도 분류 전문가입니다.
사용자 메시지를 다음 중 하나로 분류하세요:

1. symptom_medication: 증상을 설명하면서 약 추천을 요청
2. disease_diagnosis: 증상만 설명하며 질병 진단 요청
3. disease_info: 특정 질병 정보 요청
4. medication_recommend: 일반적인 약품 추천 요청
5. medication_info: 특정 약품 정보 요청  
6. general: 일반적인 인사나 대화

분류 결과만 답변하세요."""

        prompt = f"사용자 메시지: '{message}'\n분류 결과:"
        
        try:
            response = self.exaone.generate_response(prompt, system_prompt)
            valid_intents = ["symptom_medication", "disease_diagnosis", "disease_info", 
                           "medication_recommend", "medication_info", "general"]
            
            for intent in valid_intents:
                if intent in response.lower():
                    return intent
            return "general"
            
        except Exception as e:
            logger.error(f"EXAONE 의도 분류 오류: {e}")
            return "general"

class MedicationService:
    """의약품 서비스"""
    
    WARNING_MSG = "\n\n⚠️ **중요한 안전 정보**:\n- 이 정보는 일반적인 참고용입니다\n- 복용 전 반드시 의사나 약사와 상담하세요\n- 알레르기나 다른 약물과의 상호작용을 확인하세요\n- 증상이 지속되면 즉시 의료진에게 연락하세요"
    
    def __init__(self, exaone_service: EXAONE, med_index: faiss.IndexFlatIP, 
                 med_meta: List[Dict], embedding_model: EmbeddingModel):
        self.exaone = exaone_service
        self.med_index = med_index
        self.med_meta = med_meta
        self.embedding_model = embedding_model
    
    def recommend_medication_by_symptoms(self, symptoms_text: str, session: IntegratedSession) -> str:
        """증상 기반 의약품 추천"""
        
        # 1. 사용자 정보 수집
        self._collect_user_medication_info(session)
        profile = session.context["user_medication_profile"]
        
        # 2. 벡터 검색으로 관련 의약품 찾기
        similar_meds = self._search_similar_medications(symptoms_text, top_k=5)
        
        # 3. 검색 결과를 컨텍스트로 활용
        context_info = ""
        if similar_meds:
            context_info = "\n💊 관련 의약품 데이터:\n"
            for i, med in enumerate(similar_meds, 1):
                name = med.get('name', '')
                efficacy = med.get('efficacy', '')
                context_info += f"{i}. {name}: {efficacy[:100]}...\n"
        
        # 4. EXAONE을 이용한 추천
        system_prompt = """당신은 숙련된 약사이며, 일반의약품에 대한 전문가입니다.

사용자의 증상에 대해 적절한 일반의약품을 최대 3개까지 추천하십시오.
관련 의약품 데이터가 있다면 이를 참고하되, 사용자 조건에 맞는 안전한 약품을 추천해주세요.
주의사항과 부작용은 사용자 안전을 위해 자세히 설명하십시오.

각 추천 약에 대해 다음 정보를 포함하십시오:
1. 약 이름  
2. 추천 이유  
3. 복용법  
4. 주의사항  
5. 부작용"""

        prompt = f"""
사용자 정보:
- 연령대: {profile.get('age_group', '성인')}
- 임신 여부: {"예" if profile.get('is_pregnant') else "아니오"}
- 기저질환: {', '.join(profile.get('chronic_conditions', [])) if profile.get('chronic_conditions') else '없음'}

사용자 증상: {symptoms_text}
{context_info}

위 조건을 고려하여 안전하고 적절한 일반의약품을 추천해주세요."""

        try:
            response = self.exaone.generate_response(prompt, system_prompt)
            
            # 추천된 약품 정보 추출하여 세션에 저장
            medications = self._extract_medications_from_response(response)
            if medications:
                session.update_medication_context(medications)
                logger.info(f"💊 의약품 추천 완료: {len(medications)}개")
            
            return response + self.WARNING_MSG
            
        except Exception as e:
            logger.error(f"의약품 추천 오류: {e}")
            return "⚠️ 의약품 추천 중 오류가 발생했습니다. 다시 시도해주세요."
    
    def get_medication_info(self, medication_name: str) -> str:
        """의약품 정보 제공"""
        
        # 벡터 검색으로 관련 의약품 찾기
        related_meds = self._search_similar_medications(medication_name, top_k=2)
        
        context_info = ""
        if related_meds:
            context_info = "\n💊 데이터베이스 정보:\n"
            for med in related_meds:
                name = med.get('name', '')
                efficacy = med.get('efficacy', '')
                context_info += f"- {name}: {efficacy}\n"
        
        system_prompt = """당신은 전문 약사입니다.

사용자가 질문한 약품을 중심으로 정확하고 상세한 정보를 제공해주세요.
데이터베이스 정보가 있다면 이를 참고하되, 포괄적인 정보를 제공해주세요.

각 약품에 대해 다음 정보를 포함하십시오:
1. 약 이름과 주요 성분
2. 효능/효과
3. 용법/용량  
4. 주의사항
5. 부작용
6. 상호작용"""

        prompt = f"""
사용자가 질문한 의약품: {medication_name}
{context_info}

위 의약품에 대한 상세한 정보를 제공해주세요."""

        try:
            response = self.exaone.generate_response(prompt, system_prompt)
            return response + self.WARNING_MSG
            
        except Exception as e:
            logger.error(f"의약품 정보 조회 오류: {e}")
            return f"⚠️ {medication_name}에 대한 정보를 가져오는 중 오류가 발생했습니다."
    
    def recommend_by_disease_symptoms(self, disease: str, symptoms: List[str], session: IntegratedSession) -> str:
        """질병-증상 기반 의약품 추천"""
        
        # 질병과 증상을 조합한 쿼리 생성
        symptoms_text = ", ".join(symptoms) if symptoms else ""
        combined_query = f"{disease} 증상: {symptoms_text}"
        
        logger.info(f"🔗 질병-의약품 연계 추천: {combined_query}")
        
        # 기존 로직을 그대로 호출
        return self.recommend_medication_by_symptoms(combined_query, session)
    
    def _collect_user_medication_info(self, session: IntegratedSession):
        """사용자 의약품 정보 수집"""
        profile = session.context["user_medication_profile"]
        
        # CLI 환경에서는 기본값 사용
        if not profile.get("age_group"):
            profile["age_group"] = "성인"
        if profile.get("is_pregnant") is None:
            profile["is_pregnant"] = False
        if not profile.get("chronic_conditions"):
            profile["chronic_conditions"] = []
    
    def _search_similar_medications(self, query: str, top_k: int = 3) -> List[Dict]:
        """벡터 검색으로 유사한 의약품 찾기"""
        try:
            query_embedding = self.embedding_model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            scores, indices = self.med_index.search(query_embedding, top_k)
            
            results = []
            for idx in indices[0]:
                if 0 <= idx < len(self.med_meta):
                    results.append(self.med_meta[idx])
            
            return results
        except Exception as e:
            logger.error(f"의약품 벡터 검색 오류: {e}")
            return []
    
    def _extract_medications_from_response(self, response: str) -> List[Dict]:
        """응답에서 의약품 정보 추출"""
        medications = []
        
        # 의약품명 패턴 매칭 시도
        med_patterns = [
            r'1\.\s*약\s*이름[:\s]*([가-힣A-Za-z0-9\s]+)',
            r'약\s*이름[:\s]*([가-힣A-Za-z0-9\s]+)',
            r'추천\s*약[:\s]*([가-힣A-Za-z0-9\s]+)',
            r'(\w+정|\w+캡슐|\w+시럽)',
            r'(타이레놀|게보린|낙센|이부프로펜|아스피린|애드빌|부루펜)'
        ]
        
        for pattern in med_patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                med_name = match.group(1).strip()
                if med_name and len(med_name) > 1:
                    medications.append({
                        "name": med_name,
                        "source": "llm_recommendation",
                        "response_context": response[:200]
                    })
        
        # 중복 제거
        seen_names = set()
        unique_medications = []
        for med in medications:
            if med["name"] not in seen_names:
                seen_names.add(med["name"])
                unique_medications.append(med)
        
        return unique_medications[:3]

# =============================================================================
# 🚀 수정된 통합 채팅 서비스 v6
# =============================================================================

class OptimizedIntegratedChatServiceV6:
    """🚀 수정된 통합 채팅 서비스 v6 - 스마트한 차별화 질문"""
    
    def __init__(self):
        """서비스 초기화"""
        start_time = datetime.now()
        logger.info("🚀 수정된 통합 의료 챗봇 서비스 v6 초기화 시작...")
        
        # 1. 임베딩 모델 초기화
        self.embedding_model = EmbeddingModel()
        
        # 2. EXAONE LLM 초기화
        self.exaone = EXAONE()
        
        # 3. 최적화된 RAG 매니저 초기화
        self.rag_manager = OptimizedRAGIndexManager(self.embedding_model)
        self.rag_manager.load_rag_data()
        
        # 4. 질병 데이터 로드
        disease_files, medication_files = discover_csv_files()
        
        if disease_files:
            self.disease_index_key, self.disease_index_full, self.disease_meta = optimized_load_disease_indexes(
                disease_files, self.embedding_model
            )
            self.disease_service = EnhancedDiseaseService(
                self.exaone, self.disease_index_key, self.disease_index_full,
                self.disease_meta, self.embedding_model, self.rag_manager
            )
        else:
            self.disease_service = None
            logger.warning("⚠️ 질병 데이터가 없습니다.")
        
        # 5. 의약품 데이터 로드
        if medication_files:
            self.med_index, self.med_meta = optimized_load_medication_index(
                medication_files, self.embedding_model
            )
            self.medication_service = MedicationService(
                self.exaone, self.med_index, self.med_meta, self.embedding_model
            )
        else:
            self.medication_service = None
            logger.warning("⚠️ 의약품 데이터가 없습니다.")
        
        # 6. 강화된 의도 분류기 초기화
        self.intent_classifier = EnhancedIntentClassifier(self.exaone)
        
        # 초기화 완료 시간 계산
        init_time = datetime.now() - start_time
        
        # 사전 생성된 인덱스 사용 여부 확인
        using_prebuilt = self.rag_manager.use_prebuilt and PreBuiltIndexLoader.check_indexes_available()
        
        logger.info("✅ 수정된 통합 의료 챗봇 서비스 v6 초기화 완료!")
        logger.info(f"⏱️ 초기화 시간: {init_time}")
        logger.info(f"🚀 사전 생성 인덱스 사용: {'예' if using_prebuilt else '아니오'}")
        
        if using_prebuilt:
            index_info = PreBuiltIndexLoader.get_index_info()
            if index_info:
                logger.info(f"📊 인덱스 생성일: {index_info.get('created_at', 'Unknown')}")
                logger.info(f"📊 인덱스 버전: {index_info.get('version', 'Unknown')}")
        
        logger.info(f"📊 로드된 데이터:")
        logger.info(f"   - RAG Q&A: {len(self.rag_manager.qa_documents)}개")
        logger.info(f"   - RAG 의료문서: {len(self.rag_manager.medical_documents)}개")
        if self.disease_service:
            logger.info(f"   - 질병 데이터: {len(self.disease_meta)}개")
        if self.medication_service:
            logger.info(f"   - 의약품 데이터: {len(self.med_meta)}개")
    
    def process_message(self, message: str, session: IntegratedSession) -> str:
        """🔥 수정된 메시지 처리"""
        
        try:
            # 1. 강화된 의도 분류
            intent = self.intent_classifier.classify_intent(message, session)
            logger.info(f"🎯 의도 분류 결과: {intent}")
            
            # 2. 세션 초기화 처리
            if intent == "reset":
                session.reset_session()
                return "🔄 세션이 초기화되었습니다. 처음부터 다시 시작해주세요!"
            
            # 3. 차별화 질문 후속 답변 처리
            elif intent == "diagnosis_followup":
                if self.disease_service:
                    response = self.disease_service.process_followup_answer(message, session)
                else:
                    response = "⚠️ 질병 진단 서비스를 사용할 수 없습니다."
            
            # 4. 증상 기반 의약품 추천
            elif intent == "symptom_medication":
                if self.medication_service:
                    response = self.medication_service.recommend_medication_by_symptoms(message, session)
                else:
                    response = "⚠️ 의약품 추천 서비스를 사용할 수 없습니다."
            
            # 5. 기존 의도들 처리
            elif intent == "disease_diagnosis":
                if self.disease_service:
                    response = self.disease_service.diagnose_disease(message, session)
                else:
                    response = "⚠️ 질병 진단 서비스를 사용할 수 없습니다."
            
            elif intent == "disease_info":
                if self.disease_service:
                    disease_name = self._extract_disease_name(message)
                    response = self.disease_service.get_disease_info(disease_name or message)
                else:
                    response = "⚠️ 질병 정보 서비스를 사용할 수 없습니다."
            
            elif intent == "medication_recommend":
                if self.medication_service:
                    response = self.medication_service.recommend_medication_by_symptoms(message, session)
                else:
                    response = "⚠️ 의약품 추천 서비스를 사용할 수 없습니다."
            
            elif intent == "medication_info":
                if self.medication_service:
                    med_name = self._extract_medication_name(message)
                    response = self.medication_service.get_medication_info(med_name or message)
                else:
                    response = "⚠️ 의약품 정보 서비스를 사용할 수 없습니다."
            
            elif intent == "disease_to_medication":
                recent_disease = session.get_recent_diagnosis()
                if recent_disease and self.medication_service:
                    symptoms = session.get_disease_symptoms(recent_disease)
                    response = self.medication_service.recommend_by_disease_symptoms(
                        recent_disease, symptoms, session
                    )
                else:
                    response = "먼저 증상을 알려주시면 질병을 진단한 후 적절한 약품을 추천해드릴게요!"
            
            else:  # general
                response = self._handle_general_message(message)
            
            # 6. 대화 히스토리에 추가
            session.add_message(message, response, intent)
            
            return response
            
        except Exception as e:
            logger.error(f"메시지 처리 오류: {e}")
            traceback.print_exc()
            return "⚠️ 죄송합니다. 처리 중 오류가 발생했습니다. 다시 시도해주세요."
    
    def _extract_disease_name(self, message: str) -> str:
        """메시지에서 질병명 추출"""
        patterns = [
            r'([가-힣]+(?:병|염|증|암))',
            r'([가-힣]+에 대해)',
            r'([가-힣]+이란)',
            r'([가-힣]+설명)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message)
            if match:
                return match.group(1).replace('에 대해', '').replace('이란', '').replace('설명', '')
        
        return ""
    
    def _extract_medication_name(self, message: str) -> str:
        """메시지에서 의약품명 추출"""
        patterns = [
            r'([가-힣A-Za-z0-9]+(?:정|캡슐|시럽))',
            r'(타이레놀|게보린|낙센|이부프로펜|아스피린|애드빌|부루펜)',
            r'([가-힣A-Za-z0-9]+)(?:\s*(?:약|의약품|약품))'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message)
            if match:
                return match.group(1)
        
        return ""
    
    def _handle_general_message(self, message: str) -> str:
        """일반 메시지 처리"""
        greetings = ["안녕", "hello", "hi", "안녕하세요"]
        thanks = ["감사", "고마워", "thank"]
        
        message_lower = message.lower()
        
        # 사전 생성된 인덱스 사용 여부 확인
        using_prebuilt = self.rag_manager.use_prebuilt and PreBuiltIndexLoader.check_indexes_available()
        speed_info = "🚀 사전 생성된 인덱스 사용 (5초 로딩)" if using_prebuilt else "⚠️ 실시간 인덱스 생성 (16분 로딩)"
        
        if any(greet in message_lower for greet in greetings):
            return f"""안녕하세요! 수정된 의료 챗봇 v6입니다. 

🔍 **이용 가능한 기능**:
• 증상 설명 → 질병 진단 (🔥 스마트한 차별화 질문)
• 질병 진단 후 "어떤 약?" → 의약품 추천  
• 질병 정보 검색 (RAG 기반)
• 의약품 정보 검색
• "처음으로" → 세션 초기화

🔥 **수정된 기능**:
• 중복 질문 방지 - 이미 언급한 증상은 다시 묻지 않음
• 세션 상태 개선 - 초기 증상 정보 유지
• 필터링 로직 강화 - 빈 결과 방지

📚 **지식 베이스**: 6개 clean_ 파일 RAG + 질병/의약품 벡터 DB
🧠 **AI 모델**: EXAONE 3.5:7.8b + KM-BERT
{speed_info}

어떤 증상이 있으신가요?"""
        elif any(thank in message_lower for thank in thanks):
            return "도움이 되셨다니 기쁩니다! 다른 궁금한 점이 있으시면 언제든 말씀해주세요."
        else:
            return "죄송합니다. 잘 이해하지 못했습니다. 증상을 설명해주시거나 질병/의약품에 대해 구체적으로 문의해주세요."

# =============================================================================
# CLI 테스트용 메인 함수
# =============================================================================

def main():
    """수정된 CLI 테스트 메인 함수"""
    print("="*80)
    print("🚀 수정된 통합 의료 챗봇 v6 - 스마트한 차별화 질문 시스템")
    print("🔥 주요 수정사항:")
    print("   • 중복 질문 방지 - 이미 언급한 증상은 다시 묻지 않음")
    print("   • 세션 상태 개선 - 초기 증상 정보 유지")
    print("   • 필터링 로직 강화 - 빈 결과 방지")
    print("   • 종료 조건 개선 - 적절한 차별화 질문 수")
    print("="*80)
    
    # 인덱스 사용 가능 여부 확인
    if PreBuiltIndexLoader.check_indexes_available():
        print("🚀 사전 생성된 인덱스 발견! 빠른 로딩이 가능합니다.")
        index_info = PreBuiltIndexLoader.get_index_info()
        if index_info:
            print(f"📊 인덱스 생성일: {index_info.get('created_at', 'Unknown')}")
    else:
        print("⚠️ 사전 생성된 인덱스가 없습니다. 실시간 생성 모드로 실행됩니다.")
        print("💡 성능 향상을 위해 먼저 generate_faiss_indexes.py를 실행하세요!")
    
    try:
        # 🚀 수정된 통합 채팅 서비스 초기화
        chat_service = OptimizedIntegratedChatServiceV6()
        
        # 📝 세션 생성
        session = IntegratedSession()
        
        print("\n💡 테스트 예시:")
        print("1. '머리가 아프고 열이 나요' (질병 진단 → 스마트한 차별화 질문)")
        print("2. '어떤 약 먹어야 해?' (질병 진단 후 → 의약품 추천)")
        print("3. '머리 아픈데 무슨 약?' (복합 의도 → 의약품 추천)")
        print("4. '감기에 대해 알려줘' (질병 정보 → RAG 기반 검색)")
        print("5. '타이레놀 부작용이 뭐야?' (의약품 정보)")
        print("6. '처음으로' (세션 초기화)")
        print("7. 'exit' (종료)")
        print("\n**중요:** 의료진과의 상담을 대체할 수 없습니다.")
        print("\n🚀 대화를 시작하세요!")
        
        while True:
            user_input = input("\n사용자> ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['exit', 'quit', '종료']:
                print("의료 챗봇을 종료합니다.")
                break
            
            # 🔥 메시지 처리
            response = chat_service.process_message(user_input, session)
            print(f"\n챗봇> {response}")
            
            # 🔍 디버그 정보 (선택적)
            if user_input.lower() == "debug":
                print(f"\n🔬 디버그 정보:")
                print(f"   - 세션 ID: {session.session_id}")
                print(f"   - 최근 진단: {session.get_recent_diagnosis()}")
                print(f"   - 질문 모드: {session.context['questioning_state']['is_questioning']}")
                print(f"   - 초기 증상: {session.context.get('initial_symptoms_text', 'None')}")
                print(f"   - 언급된 증상: {session.context.get('mentioned_symptoms', [])}")
                print(f"   - 대화 수: {len(session.history)}")
                print(f"   - 사전 생성 인덱스 사용: {chat_service.rag_manager.use_prebuilt}")
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()