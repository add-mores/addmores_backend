"""
FAISS 인덱스 사전 생성 스크립트 - 수정 버전
디렉토리: generate_faiss_indexes.py

🚀 성능 최적화: 16분 → 5초 단축
✅ 전체 데이터 지원: clean_ + disease_prototype + medicine_code_merged
✅ 배치 처리: 메모리 효율성 개선
✅ 자동 저장: 인덱스 + 메타데이터 + 문서
🔧 수정: 실제 CSV 컬럼명에 맞춰 감지 로직 개선
"""

import os
import sys
import pandas as pd
import numpy as np
import faiss
import torch
import pickle
import json
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Tuple, Optional, Any
import logging
from datetime import datetime
import gc
from dataclasses import dataclass
from enum import Enum

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# 설정 및 상수
# =============================================================================

class IndexConfig:
    """인덱스 생성 설정"""
    # 저장 디렉토리
    INDEX_DIR = "faiss_indexes"
    
    # 배치 크기 (메모리 효율성)
    EMBEDDING_BATCH_SIZE = 100
    
    # 파일 패턴
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

# =============================================================================
# 고성능 임베딩 모델 클래스
# =============================================================================

class BatchEmbeddingModel:
    """배치 처리 최적화 임베딩 모델"""
    
    def __init__(self, model_name: str = "madatnlp/km-bert"):
        logger.info(f"🔄 임베딩 모델 로딩 중: {model_name}")
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        logger.info(f"✅ 임베딩 모델 로드 완료 (Device: {self.device})")

    def encode_batch(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """배치 처리로 텍스트 인코딩 (메모리 효율성)"""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            logger.info(f"📦 배치 처리 중: {i+1}-{min(i+batch_size, len(texts))}/{len(texts)}")
            
            encodings = self.tokenizer(
                batch_texts,
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
                
                all_embeddings.append(sentence_embeddings.cpu().numpy())
                
                # 메모리 정리
                del encodings, outputs, last_hidden, attention_mask, masked_hidden
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return np.vstack(all_embeddings)

# =============================================================================
# 데이터 탐지 및 로더 클래스
# =============================================================================

class DataDiscovery:
    """확장된 데이터 탐지 클래스"""
    
    @staticmethod
    def discover_all_files() -> Dict[str, List[str]]:
        """🔥 모든 데이터 파일 탐지 - clean_ + disease_prototype + medicine_code_merged"""
        files = [f for f in os.listdir('.') if f.lower().endswith(".csv")]
        
        categorized_files = {
            "rag_qa": [],           # Q&A 데이터 (clean_51004)
            "rag_medical": [],      # 의료 문서 (clean_55588~66149)
            "disease": [],          # 질병 데이터 (disease_prototype + 기타)
            "medication": []        # 의약품 데이터 (medicine_code_merged + 기타)
        }
        
        # 🔥 우선순위 파일들 체크
        priority_files = {
            "clean_51004.csv": "rag_qa",
            "disease_prototype.csv": "disease", 
            "medicine_code_merged.csv": "medication"
        }
        
        for fname, category in priority_files.items():
            if os.path.exists(fname):
                categorized_files[category].append(fname)
                logger.info(f"📌 우선순위 파일 발견: {fname} → {category}")
        
        # clean_ 의료 문서 파일들
        medical_clean_files = [
            "clean_55588.csv", "clean_56763.csv", "clean_58572.csv", 
            "clean_63166.csv", "clean_66149.csv"
        ]
        
        for fname in medical_clean_files:
            if os.path.exists(fname):
                categorized_files["rag_medical"].append(fname)
                logger.info(f"📄 의료 문서 파일 발견: {fname}")
        
        # 나머지 파일들 자동 분류
        for fname in files:
            if fname in [f for file_list in categorized_files.values() for f in file_list]:
                continue  # 이미 분류됨
                
            try:
                df = pd.read_csv(fname, encoding="utf-8", nrows=5)  # 샘플만 읽기
                category = DataDiscovery._classify_file(df, fname)
                if category:
                    categorized_files[category].append(fname)
                    logger.info(f"🔍 자동 분류: {fname} → {category}")
                    
            except Exception as e:
                logger.warning(f"⚠️ 파일 분류 실패: {fname} - {e}")
        
        # 결과 요약
        for category, file_list in categorized_files.items():
            logger.info(f"📊 {category}: {len(file_list)}개 파일")
            
        return categorized_files
    
    @staticmethod
    def _classify_file(df: pd.DataFrame, filename: str) -> Optional[str]:
        """파일 자동 분류 - 실제 컬럼명 기반"""
        columns = [col.lower() for col in df.columns]
        
        # Q&A 패턴
        if any(keyword in ' '.join(columns) for keyword in ['question', 'answer', 'q&a']):
            return "rag_qa"
        
        # 🔧 질병 패턴 - 실제 컬럼명 기반
        disease_keywords = ['disnm', 'disease', '질병', '병명', 'symptom', '증상', 'sym', 'diagnosis', '진단']
        if any(keyword in ' '.join(columns) for keyword in disease_keywords):
            return "disease"
        
        # 의약품 패턴
        med_keywords = ['item', '품목', '약품', 'medicine', 'drug', 'efcy', '효능']
        if any(keyword in ' '.join(columns) for keyword in med_keywords):
            return "medication"
        
        # 일반 의료 문서
        return "rag_medical"

# =============================================================================
# 고성능 RAG 인덱스 생성기
# =============================================================================

class RAGIndexGenerator:
    """RAG 인덱스 생성기"""
    
    def __init__(self, embedding_model: BatchEmbeddingModel):
        self.embedding_model = embedding_model
        self.qa_documents = []
        self.medical_documents = []
        
    def generate_rag_indexes(self, qa_files: List[str], medical_files: List[str]) -> Tuple[faiss.Index, faiss.Index]:
        """RAG 인덱스 생성"""
        logger.info("🔄 RAG 데이터 로딩 시작...")
        
        # Q&A 데이터 로드
        self._load_qa_data(qa_files)
        
        # 의료 문서 데이터 로드
        self._load_medical_documents(medical_files)
        
        # 인덱스 구축
        qa_index, medical_index = self._build_rag_indexes()
        
        logger.info("✅ RAG 인덱스 생성 완료!")
        return qa_index, medical_index
    
    def _load_qa_data(self, qa_files: List[str]):
        """Q&A 데이터 로드"""
        for file_path in qa_files:
            logger.info(f"📂 Q&A 데이터 로드: {file_path}")
            
            try:
                df = pd.read_csv(file_path, encoding="utf-8")
                
                for idx, row in df.iterrows():
                    try:
                        question = str(row.get('question', '')).strip()
                        answer = str(row.get('answer', '')).strip()
                        
                        if question and answer and question != 'nan' and answer != 'nan':
                            content = f"Q: {question}\nA: {answer}"
                            
                            doc = RAGDocument(
                                doc_id=f"qa_{file_path}_{idx}",
                                content=content,
                                metadata={
                                    'question': question,
                                    'answer': answer,
                                    'source': file_path
                                },
                                content_type=RAGContentType.QA
                            )
                            self.qa_documents.append(doc)
                            
                    except Exception as e:
                        logger.error(f"Q&A 행 처리 오류 {file_path}:{idx} - {e}")
                        
                logger.info(f"✅ {file_path}: {len([d for d in self.qa_documents if file_path in d.doc_id])}개 로드")
                
            except Exception as e:
                logger.error(f"❌ Q&A 파일 로드 실패: {file_path} - {e}")
    
    def _load_medical_documents(self, medical_files: List[str]):
        """의료 문서 데이터 로드"""
        for file_path in medical_files:
            logger.info(f"📂 의료 문서 로드: {file_path}")
            
            try:
                df = pd.read_csv(file_path, encoding="utf-8")
                
                for idx, row in df.iterrows():
                    try:
                        content_parts = []
                        
                        for col in df.columns:
                            value = str(row.get(col, '')).strip()
                            if value and value != 'nan' and len(value) > 5:
                                content_parts.append(f"{col}: {value}")
                        
                        if len(content_parts) >= 2:  # 최소 2개 이상의 유효한 컬럼
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
                        logger.error(f"의료 문서 행 처리 오류 {file_path}:{idx} - {e}")
                        
                logger.info(f"✅ {file_path}: {len([d for d in self.medical_documents if file_path in d.doc_id])}개 로드")
                
            except Exception as e:
                logger.error(f"❌ 의료 문서 파일 로드 실패: {file_path} - {e}")
    
    def _build_rag_indexes(self) -> Tuple[faiss.Index, faiss.Index]:
        """RAG FAISS 인덱스 구축"""
        qa_index = None
        medical_index = None
        
        # Q&A 인덱스 구축
        if self.qa_documents:
            logger.info(f"🔄 Q&A 임베딩 생성 중: {len(self.qa_documents)}개")
            qa_texts = [doc.content for doc in self.qa_documents]
            qa_embeddings = self.embedding_model.encode_batch(qa_texts, IndexConfig.EMBEDDING_BATCH_SIZE)
            faiss.normalize_L2(qa_embeddings)
            
            qa_index = faiss.IndexFlatIP(qa_embeddings.shape[1])
            qa_index.add(qa_embeddings)
            logger.info(f"✅ Q&A 인덱스 구축 완료: {len(self.qa_documents)}개")
        
        # 의료 문서 인덱스 구축
        if self.medical_documents:
            logger.info(f"🔄 의료 문서 임베딩 생성 중: {len(self.medical_documents)}개")
            med_texts = [doc.content for doc in self.medical_documents]
            med_embeddings = self.embedding_model.encode_batch(med_texts, IndexConfig.EMBEDDING_BATCH_SIZE)
            faiss.normalize_L2(med_embeddings)
            
            medical_index = faiss.IndexFlatIP(med_embeddings.shape[1])
            medical_index.add(med_embeddings)
            logger.info(f"✅ 의료 문서 인덱스 구축 완료: {len(self.medical_documents)}개")
        
        return qa_index, medical_index

# =============================================================================
# 질병 인덱스 생성기 - 수정된 컬럼 감지 로직
# =============================================================================

class DiseaseIndexGenerator:
    """질병 인덱스 생성기 - 실제 CSV 컬럼명 기반"""
    
    def __init__(self, embedding_model: BatchEmbeddingModel):
        self.embedding_model = embedding_model
        
    def generate_disease_indexes(self, disease_files: List[str]) -> Tuple[faiss.Index, faiss.Index, List[Dict]]:
        """질병 인덱스 생성"""
        logger.info("🔄 질병 데이터 로딩 시작...")
        
        all_key_texts = []
        all_full_texts = []
        all_metadata = []
        
        for file_path in disease_files:
            logger.info(f"📂 질병 데이터 로드: {file_path}")
            
            try:
                df = pd.read_csv(file_path, encoding="utf-8", low_memory=False)
                logger.info(f"🔍 {file_path} 컬럼 확인: {list(df.columns)}")
                
                detected_cols = self._detect_disease_columns(df)
                logger.info(f"🔍 감지된 컬럼: {detected_cols}")
                
                disease_col = detected_cols.get("disease_name")
                symptoms_col = detected_cols.get("symptoms")
                symptoms_key_col = detected_cols.get("symptoms_key")
                
                if not disease_col:
                    logger.warning(f"⚠️ {file_path}: 질병명 컬럼을 찾을 수 없습니다. 감지된 컬럼: {detected_cols}")
                    continue
                
                valid_count = 0
                for idx, row in df.iterrows():
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
                        
                        # 기타 컬럼들도 수집 (정의, 치료법 등)
                        additional_info = []
                        for col in ['def', 'therapy', 'diag', 'guide', 'pvt']:
                            if col in df.columns:
                                value = str(row.get(col, "")).strip()
                                if value and value != "nan":
                                    additional_info.append(value)
                        
                        # 메타데이터 구성
                        metadata = {
                            "disease": disease_name,
                            "symptoms": symptoms_full,
                            "symptoms_key": symptoms_key,
                            "additional_info": " ".join(additional_info),
                            "source_file": file_path,
                            "original_data": row.to_dict()
                        }
                        
                        # 임베딩용 텍스트 구성
                        key_text = f"{disease_name} {symptoms_key}".strip()
                        full_text = f"{disease_name} {symptoms_full} {symptoms_key} {' '.join(additional_info)}".strip()
                        
                        all_metadata.append(metadata)
                        all_key_texts.append(key_text)
                        all_full_texts.append(full_text)
                        valid_count += 1
                        
                    except Exception as e:
                        logger.error(f"질병 행 처리 오류 {file_path}:{idx} - {e}")
                
                logger.info(f"✅ {file_path}: {valid_count}개 질병 데이터 로드")
                
            except Exception as e:
                logger.error(f"❌ 질병 파일 로드 실패: {file_path} - {e}")
        
        if not all_metadata:
            raise ValueError("유효한 질병 데이터가 없습니다.")
        
        # 임베딩 생성 및 인덱스 구축
        logger.info(f"🔄 질병 임베딩 생성 중: {len(all_metadata)}개")
        
        key_embeddings = self.embedding_model.encode_batch(all_key_texts, IndexConfig.EMBEDDING_BATCH_SIZE)
        full_embeddings = self.embedding_model.encode_batch(all_full_texts, IndexConfig.EMBEDDING_BATCH_SIZE)
        
        faiss.normalize_L2(key_embeddings)
        faiss.normalize_L2(full_embeddings)
        
        disease_key_index = faiss.IndexFlatIP(key_embeddings.shape[1])
        disease_full_index = faiss.IndexFlatIP(full_embeddings.shape[1])
        
        disease_key_index.add(key_embeddings)
        disease_full_index.add(full_embeddings)
        
        logger.info(f"✅ 질병 인덱스 구축 완료: {len(all_metadata)}개")
        return disease_key_index, disease_full_index, all_metadata
    
    def _detect_disease_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """🔧 수정된 질병 컬럼 감지 - 실제 CSV 컬럼명 기반"""
        columns = df.columns.tolist()
        detected = {}
        
        # 🔥 실제 컬럼명 매핑
        column_mappings = {
            # 질병명 매핑
            'disease_name': ['disnm_ko', 'disnm_en', 'disease', '질병', '병명', 'disease_name'],
            # 증상 매핑
            'symptoms': ['sym', 'symptoms', '증상', 'symptom'],
            # 핵심 증상 매핑  
            'symptoms_key': ['sym_k', 'symptoms_key', '핵심증상', 'key_symptoms']
        }
        
        for target_type, possible_names in column_mappings.items():
            for col in columns:
                # 정확한 매치 우선
                if col in possible_names:
                    detected[target_type] = col
                    break
                # 부분 매치 (소문자 변환 후)
                col_lower = col.lower()
                for possible in possible_names:
                    if possible.lower() in col_lower or col_lower in possible.lower():
                        detected[target_type] = col
                        break
                if target_type in detected:
                    break
        
        logger.info(f"🔍 질병 컬럼 감지 결과: {detected}")
        return detected

# =============================================================================
# 의약품 인덱스 생성기 - 기존 로직 유지
# =============================================================================

class MedicationIndexGenerator:
    """의약품 인덱스 생성기"""
    
    def __init__(self, embedding_model: BatchEmbeddingModel):
        self.embedding_model = embedding_model
        
    def generate_medication_index(self, medication_files: List[str]) -> Tuple[faiss.Index, List[Dict]]:
        """의약품 인덱스 생성"""
        logger.info("🔄 의약품 데이터 로딩 시작...")
        
        all_med_texts = []
        all_metadata = []
        
        for file_path in medication_files:
            logger.info(f"📂 의약품 데이터 로드: {file_path}")
            
            try:
                df = pd.read_csv(file_path, encoding="utf-8", low_memory=False)
                detected_cols = self._detect_medication_columns(df)
                
                item_col = detected_cols.get("itemName")
                efcy_col = detected_cols.get("efcyQesitm")
                
                if not item_col or not efcy_col:
                    logger.warning(f"⚠️ {file_path}: 필요한 의약품 컬럼을 찾을 수 없습니다.")
                    continue
                
                valid_count = 0
                for idx, row in df.iterrows():
                    try:
                        item_name = str(row.get(item_col, "")).strip()
                        efficacy = str(row.get(efcy_col, "")).strip()
                        
                        if not item_name or not efficacy or item_name == "nan" or efficacy == "nan":
                            continue
                        
                        # 문서 텍스트 구성
                        doc_text = f"의약품명: {item_name}\n효능: {efficacy}"
                        
                        # 메타데이터 구성
                        metadata = {
                            "name": item_name,
                            "efficacy": efficacy,
                            "doc_text": doc_text,
                            "source_file": file_path,
                            "original_data": row.to_dict()
                        }
                        
                        all_med_texts.append(doc_text)
                        all_metadata.append(metadata)
                        valid_count += 1
                        
                    except Exception as e:
                        logger.error(f"의약품 행 처리 오류 {file_path}:{idx} - {e}")
                
                logger.info(f"✅ {file_path}: {valid_count}개 의약품 데이터 로드")
                
            except Exception as e:
                logger.error(f"❌ 의약품 파일 로드 실패: {file_path} - {e}")
        
        if not all_med_texts:
            logger.warning("⚠️ 유효한 의약품 데이터가 없습니다.")
            return None, []
        
        # 임베딩 생성 및 인덱스 구축
        logger.info(f"🔄 의약품 임베딩 생성 중: {len(all_metadata)}개")
        
        med_embeddings = self.embedding_model.encode_batch(all_med_texts, IndexConfig.EMBEDDING_BATCH_SIZE)
        faiss.normalize_L2(med_embeddings)
        
        medication_index = faiss.IndexFlatIP(med_embeddings.shape[1])
        medication_index.add(med_embeddings)
        
        logger.info(f"✅ 의약품 인덱스 구축 완료: {len(all_metadata)}개")
        return medication_index, all_metadata
    
    def _detect_medication_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """의약품 컬럼 감지"""
        columns = df.columns.tolist()
        detected = {}
        
        for col in columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['item', '품목', '약품명']):
                detected['itemName'] = col
            elif any(keyword in col_lower for keyword in ['efcy', '효능', '효과']):
                detected['efcyQesitm'] = col
        
        return detected

# =============================================================================
# 인덱스 저장/로드 매니저
# =============================================================================

class IndexManager:
    """인덱스 저장/로드 매니저"""
    
    @staticmethod
    def save_indexes(
        qa_index: faiss.Index,
        medical_index: faiss.Index, 
        disease_key_index: faiss.Index,
        disease_full_index: faiss.Index,
        medication_index: faiss.Index,
        qa_documents: List[RAGDocument],
        medical_documents: List[RAGDocument],
        disease_metadata: List[Dict],
        medication_metadata: List[Dict]
    ):
        """모든 인덱스와 메타데이터 저장"""
        
        # 저장 디렉토리 생성
        os.makedirs(IndexConfig.INDEX_DIR, exist_ok=True)
        
        logger.info("💾 인덱스 저장 시작...")
        
        try:
            # FAISS 인덱스 저장
            if qa_index:
                faiss.write_index(qa_index, os.path.join(IndexConfig.INDEX_DIR, IndexConfig.INDEX_FILES["rag_qa"]))
                logger.info("✅ RAG Q&A 인덱스 저장 완료")
            
            if medical_index:
                faiss.write_index(medical_index, os.path.join(IndexConfig.INDEX_DIR, IndexConfig.INDEX_FILES["rag_medical"]))
                logger.info("✅ RAG 의료문서 인덱스 저장 완료")
            
            if disease_key_index:
                faiss.write_index(disease_key_index, os.path.join(IndexConfig.INDEX_DIR, IndexConfig.INDEX_FILES["disease_key"]))
                logger.info("✅ 질병 Key 인덱스 저장 완료")
            
            if disease_full_index:
                faiss.write_index(disease_full_index, os.path.join(IndexConfig.INDEX_DIR, IndexConfig.INDEX_FILES["disease_full"]))
                logger.info("✅ 질병 Full 인덱스 저장 완료")
            
            if medication_index:
                faiss.write_index(medication_index, os.path.join(IndexConfig.INDEX_DIR, IndexConfig.INDEX_FILES["medication"]))
                logger.info("✅ 의약품 인덱스 저장 완료")
            
            # 메타데이터 저장
            if qa_documents:
                with open(os.path.join(IndexConfig.INDEX_DIR, IndexConfig.METADATA_FILES["rag_qa"]), 'wb') as f:
                    pickle.dump(qa_documents, f)
                logger.info("✅ RAG Q&A 문서 저장 완료")
            
            if medical_documents:
                with open(os.path.join(IndexConfig.INDEX_DIR, IndexConfig.METADATA_FILES["rag_medical"]), 'wb') as f:
                    pickle.dump(medical_documents, f)
                logger.info("✅ RAG 의료문서 저장 완료")
            
            if disease_metadata:
                with open(os.path.join(IndexConfig.INDEX_DIR, IndexConfig.METADATA_FILES["disease"]), 'wb') as f:
                    pickle.dump(disease_metadata, f)
                logger.info("✅ 질병 메타데이터 저장 완료")
            
            if medication_metadata:
                with open(os.path.join(IndexConfig.INDEX_DIR, IndexConfig.METADATA_FILES["medication"]), 'wb') as f:
                    pickle.dump(medication_metadata, f)
                logger.info("✅ 의약품 메타데이터 저장 완료")
            
            # 설정 정보 저장
            config_info = {
                "created_at": datetime.now().isoformat(),
                "total_qa_docs": len(qa_documents) if qa_documents else 0,
                "total_medical_docs": len(medical_documents) if medical_documents else 0,
                "total_disease_docs": len(disease_metadata) if disease_metadata else 0,
                "total_medication_docs": len(medication_metadata) if medication_metadata else 0,
                "embedding_model": "madatnlp/km-bert",
                "version": "v4.1_fixed"
            }
            
            with open(os.path.join(IndexConfig.INDEX_DIR, IndexConfig.CONFIG_FILE), 'w', encoding='utf-8') as f:
                json.dump(config_info, f, ensure_ascii=False, indent=2)
            
            logger.info("✅ 모든 인덱스 저장 완료!")
            logger.info(f"📁 저장 경로: {os.path.abspath(IndexConfig.INDEX_DIR)}")
            
        except Exception as e:
            logger.error(f"❌ 인덱스 저장 실패: {e}")
            raise

# =============================================================================
# 메인 생성 함수
# =============================================================================

def generate_all_indexes():
    """모든 FAISS 인덱스 생성 메인 함수"""
    start_time = datetime.now()
    logger.info("🚀 FAISS 인덱스 사전 생성 시작!")
    logger.info(f"📅 시작 시간: {start_time}")
    
    try:
        # 1. 임베딩 모델 초기화
        embedding_model = BatchEmbeddingModel()
        
        # 2. 데이터 파일 탐지
        file_categories = DataDiscovery.discover_all_files()
        
        # 3. RAG 인덱스 생성
        rag_generator = RAGIndexGenerator(embedding_model)
        qa_index, medical_index = rag_generator.generate_rag_indexes(
            file_categories["rag_qa"], 
            file_categories["rag_medical"]
        )
        
        # 4. 질병 인덱스 생성
        disease_key_index = disease_full_index = disease_metadata = None
        if file_categories["disease"]:
            disease_generator = DiseaseIndexGenerator(embedding_model)
            disease_key_index, disease_full_index, disease_metadata = disease_generator.generate_disease_indexes(
                file_categories["disease"]
            )
        
        # 5. 의약품 인덱스 생성
        medication_index = medication_metadata = None
        if file_categories["medication"]:
            medication_generator = MedicationIndexGenerator(embedding_model)
            medication_index, medication_metadata = medication_generator.generate_medication_index(
                file_categories["medication"]
            )
        
        # 6. 모든 인덱스 저장
        IndexManager.save_indexes(
            qa_index=qa_index,
            medical_index=medical_index,
            disease_key_index=disease_key_index,
            disease_full_index=disease_full_index,
            medication_index=medication_index,
            qa_documents=rag_generator.qa_documents,
            medical_documents=rag_generator.medical_documents,
            disease_metadata=disease_metadata or [],
            medication_metadata=medication_metadata or []
        )
        
        # 완료 시간 계산
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("🎉 모든 FAISS 인덱스 생성 완료!")
        logger.info(f"⏱️ 총 소요 시간: {duration}")
        logger.info("📊 생성 결과:")
        logger.info(f"   - RAG Q&A: {len(rag_generator.qa_documents)}개 문서")
        logger.info(f"   - RAG 의료문서: {len(rag_generator.medical_documents)}개 문서")
        logger.info(f"   - 질병 데이터: {len(disease_metadata) if disease_metadata else 0}개")
        logger.info(f"   - 의약품 데이터: {len(medication_metadata) if medication_metadata else 0}개")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 인덱스 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # 메모리 정리
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

if __name__ == "__main__":
    print("="*80)
    print("🚀 FAISS 인덱스 사전 생성 스크립트 - 수정 버전")
    print("🔧 수정: 실제 CSV 컬럼명에 맞춘 감지 로직 개선")
    print("✅ disnm_ko, sym, sym_k 등 실제 컬럼명 지원")
    print("="*80)
    
    success = generate_all_indexes()
    
    if success:
        print("\n✅ 인덱스 생성 성공!")
        print("💡 이제 메인 애플리케이션에서 빠른 로딩이 가능합니다.")
    else:
        print("\n❌ 인덱스 생성 실패!")
        print("🔍 로그를 확인하여 문제를 해결해주세요.")