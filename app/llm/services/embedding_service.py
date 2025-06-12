"""
통합 임베딩 서비스 - 모든 FAISS 인덱스 통합 관리
위치: backend/app/llm/services/embedding_service.py

🎯 목적: 
- 10개 FAISS 인덱스 파일 통합 로드
- KM-BERT 기반 임베딩 생성
- Pickle 클래스 참조 문제 해결
- RAG + 질병 + 의약품 검색 통합

📋 기능:
- UnifiedIndexLoader: 모든 인덱스 통합 로더
- EmbeddingModel: KM-BERT 임베딩 생성
- EnhancedRAGIndexManager: 고급 검색 기능
- CustomUnpickler: Pickle 문제 해결

🚀 기술스택: FastAPI + KM-BERT + FAISS + AWS
"""

import os
import pickle
import json
import logging
import pandas as pd
import numpy as np
import faiss
import torch
import sys
import io
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

# 로깅 설정
logger = logging.getLogger(__name__)

# =============================================================================
# 🔧 Pickle 클래스 참조 문제 해결용 커스텀 Unpickler
# =============================================================================

class CustomUnpickler(pickle.Unpickler):
    """
    Pickle 클래스 참조 문제 해결용 커스텀 Unpickler
    
    🎯 목적: pickle 파일에 저장된 클래스가 다른 모듈 경로에 있을 때 자동 매핑
    📋 해결 방법: 모듈 경로 자동 변환 및 현재 모듈에서 클래스 찾기
    """
    
    def find_class(self, module, name):
        """클래스 찾기 시 경로 매핑"""
        
        # RAG 관련 클래스 특별 처리
        if name in ['RAGDocument', 'RAGContentType']:
            current_module = sys.modules[__name__]
            if hasattr(current_module, name):
                logger.debug(f"🔄 클래스 매핑: {module}.{name} → {__name__}.{name}")
                return getattr(current_module, name)
        
        # 기본 동작 시도
        try:
            return super().find_class(module, name)
        except (AttributeError, ImportError, ModuleNotFoundError) as e:
            logger.debug(f"⚠️ 기본 클래스 찾기 실패: {module}.{name} - {e}")
            
            # 현재 모듈에서 찾기 시도
            current_module = sys.modules[__name__]
            if hasattr(current_module, name):
                logger.info(f"✅ 현재 모듈에서 클래스 발견: {name}")
                return getattr(current_module, name)
            
            # 최후 수단: object 반환
            logger.warning(f"⚠️ 클래스를 찾을 수 없음: {module}.{name}, object로 대체")
            return object

# =============================================================================
# 🔍 RAG 관련 클래스 및 데이터 모델
# =============================================================================

class RAGContentType(Enum):
    """RAG 컨텐츠 타입 정의"""
    QA = "qa"                    # Q&A 형태 데이터
    MEDICAL_DOC = "medical_doc"  # 의료 문서 데이터

@dataclass
class RAGDocument:
    """
    RAG 문서 데이터 클래스
    
    📋 구성요소:
    - doc_id: 문서 고유 ID
    - content: 문서 내용 (검색 대상)
    - metadata: 추가 메타데이터
    - content_type: 컨텐츠 타입 (QA/MEDICAL_DOC)
    - embedding: 벡터 임베딩 (선택사항)
    """
    doc_id: str
    content: str
    metadata: Dict
    content_type: RAGContentType
    embedding: Optional[np.ndarray] = None

# =============================================================================
# 🧠 KM-BERT 기반 임베딩 모델
# =============================================================================

class EmbeddingModel:
    """
    KM-BERT 기반 임베딩 모델
    
    🧠 모델: madatnlp/km-bert (한국어 특화)
    📏 차원: 768차원 벡터
    ⚡ GPU 지원: CUDA 사용 가능 시 자동 활용
    🔧 배치 처리: 대용량 데이터 효율적 처리
    """
    
    def __init__(self, model_name: str = "madatnlp/km-bert"):
        """임베딩 모델 초기화"""
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"🧠 임베딩 모델 로딩 시작: {model_name}")
        logger.info(f"🖥️ 사용 디바이스: {self.device}")
        
        try:
            # 토크나이저와 모델 로드
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # 임베딩 차원 확인
            self.embedding_dim = self.model.config.hidden_size
            
            logger.info(f"✅ 임베딩 모델 로딩 완료 (차원: {self.embedding_dim})")
            
        except Exception as e:
            logger.error(f"❌ 임베딩 모델 로딩 실패: {e}")
            raise e
    
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        텍스트 리스트를 임베딩으로 변환
        
        Args:
            texts: 텍스트 리스트
            batch_size: 배치 크기 (메모리 사용량 조절)
            
        Returns:
            np.ndarray: 임베딩 배열 (n, 768)
        """
        if not texts:
            return np.array([]).reshape(0, self.embedding_dim)
        
        all_embeddings = []
        
        # 배치 단위로 처리 (메모리 효율성)
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            try:
                # 토크나이징
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)
                
                # 임베딩 생성
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # [CLS] 토큰의 임베딩 사용 (문장 표현)
                    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                all_embeddings.append(embeddings)
                
            except Exception as e:
                logger.error(f"❌ 배치 임베딩 생성 실패: {e}")
                # 실패한 배치는 제로 벡터로 대체
                zero_embeddings = np.zeros((len(batch_texts), self.embedding_dim))
                all_embeddings.append(zero_embeddings)
        
        # 모든 배치 결합
        final_embeddings = np.vstack(all_embeddings)
        
        logger.debug(f"🔍 임베딩 생성 완료: {len(texts)}개 텍스트 → {final_embeddings.shape}")
        
        return final_embeddings
    
    def encode_single(self, text: str) -> np.ndarray:
        """단일 텍스트 임베딩 생성"""
        return self.encode([text])[0]
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """코사인 유사도 계산"""
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity([embedding1], [embedding2])[0][0]

# =============================================================================
# 🔧 통합 인덱스 로더 - 모든 FAISS 인덱스 파일 활용
# =============================================================================

class UnifiedIndexLoader:
    """
    통합 FAISS 인덱스 로더 - 모든 10개 파일 활용
    
    🚀 기능:
    - 모든 FAISS 인덱스 한 번에 로드
    - Pickle 클래스 참조 문제 해결
    - 통합 관리 및 모니터링
    - 부분 로드 허용 (일부 파일 누락 시에도 동작)
    
    📁 지원 파일:
    - RAG 인덱스: rag_qa_index.index, rag_medical_index.index
    - 질병 인덱스: disease_key_index.index, disease_full_index.index  
    - 의약품 인덱스: medication_index.index
    - 메타데이터: *.pkl 파일들
    """
    
    INDEX_DIR = "app/integration_test/faiss_indexes"
    
    # 🔍 모든 인덱스 파일 정의
    ALL_INDEX_FILES = {
        # RAG 시스템용
        "rag_qa": "rag_qa_index.index",
        "rag_medical": "rag_medical_index.index",
        
        # 질병 진단용
        "disease_key": "disease_key_index.index",
        "disease_full": "disease_full_index.index",
        
        # 의약품용
        "medication": "medication_index.index"
    }
    
    # 🔍 모든 메타데이터 파일 정의
    ALL_METADATA_FILES = {
        # RAG 메타데이터
        "rag_qa": "rag_qa_documents.pkl",
        "rag_medical": "rag_medical_documents.pkl",
        
        # 질병 메타데이터
        "disease": "disease_metadata.pkl",
        
        # 의약품 메타데이터
        "medication": "medication_metadata.pkl"
    }
    
    CONFIG_FILE = "index_config.json"
    
    def __init__(self):
        """통합 로더 초기화"""
        self.indexes = {}
        self.metadata = {}
        self.config = {}
        self.load_status = {}
        
    def load_all_indexes(self) -> Dict[str, Any]:
        """
        모든 FAISS 인덱스 로드
        
        Returns:
            Dict: 로드된 인덱스들과 상태 정보
        """
        logger.info("🚀 통합 FAISS 인덱스 로더 시작 - 모든 10개 파일 활용")
        
        if not os.path.exists(self.INDEX_DIR):
            logger.error(f"❌ 인덱스 디렉토리가 없습니다: {self.INDEX_DIR}")
            return self._create_empty_result()
        
        # 📊 사용 가능한 파일 확인
        available_files = os.listdir(self.INDEX_DIR)
        logger.info(f"📁 발견된 파일 수: {len(available_files)}개")
        
        # 🔍 1단계: 모든 FAISS 인덱스 로드
        self._load_all_faiss_indexes()
        
        # 🔍 2단계: 모든 메타데이터 로드
        self._load_all_metadata()
        
        # 🔍 3단계: 설정 파일 로드
        self._load_config()
        
        # 📊 4단계: 결과 요약
        result = self._summarize_results()
        
        logger.info("✅ 통합 인덱스 로딩 완료!")
        self._log_summary()
        
        return result
    
    def _load_all_faiss_indexes(self):
        """모든 FAISS 인덱스 파일 로드"""
        logger.info("🔍 FAISS 인덱스 파일들 로드 중...")
        
        for index_name, filename in self.ALL_INDEX_FILES.items():
            file_path = os.path.join(self.INDEX_DIR, filename)
            
            try:
                if os.path.exists(file_path):
                    logger.info(f"📂 {index_name} 인덱스 로드: {filename}")
                    
                    # FAISS 인덱스 로드
                    index = faiss.read_index(file_path)
                    self.indexes[index_name] = index
                    
                    # 상태 기록
                    self.load_status[index_name] = {
                        "status": "success",
                        "file_size": os.path.getsize(file_path),
                        "vector_count": index.ntotal,
                        "dimension": index.d if hasattr(index, 'd') else 'unknown'
                    }
                    
                    logger.info(f"✅ {index_name}: {index.ntotal}개 벡터, {index.d if hasattr(index, 'd') else 'unknown'}차원")
                    
                else:
                    logger.warning(f"⚠️ {index_name} 인덱스 파일이 없습니다: {filename}")
                    self.load_status[index_name] = {"status": "file_not_found"}
                    
            except Exception as e:
                logger.error(f"❌ {index_name} 인덱스 로드 실패: {e}")
                self.load_status[index_name] = {"status": "load_failed", "error": str(e)}
    
    def _load_all_metadata(self):
        """모든 메타데이터 파일 로드 (CustomUnpickler 사용)"""
        logger.info("🔍 메타데이터 파일들 로드 중...")
        
        for metadata_name, filename in self.ALL_METADATA_FILES.items():
            file_path = os.path.join(self.INDEX_DIR, filename)
            
            try:
                if os.path.exists(file_path):
                    logger.info(f"📂 {metadata_name} 메타데이터 로드: {filename}")
                    
                    # CustomUnpickler 사용하여 안전하게 로드
                    metadata = self._safe_pickle_load(file_path)
                    self.metadata[metadata_name] = metadata
                    
                    # 상태 기록
                    self.load_status[f"{metadata_name}_metadata"] = {
                        "status": "success",
                        "file_size": os.path.getsize(file_path),
                        "item_count": len(metadata) if isinstance(metadata, list) else "unknown"
                    }
                    
                    logger.info(f"✅ {metadata_name}: {len(metadata) if isinstance(metadata, list) else 'unknown'}개 항목")
                    
                else:
                    logger.warning(f"⚠️ {metadata_name} 메타데이터 파일이 없습니다: {filename}")
                    self.load_status[f"{metadata_name}_metadata"] = {"status": "file_not_found"}
                    
            except Exception as e:
                logger.error(f"❌ {metadata_name} 메타데이터 로드 실패: {e}")
                self.load_status[f"{metadata_name}_metadata"] = {"status": "load_failed", "error": str(e)}
    
    def _safe_pickle_load(self, file_path: str) -> Any:
        """
        안전한 pickle 로드 (다중 fallback 방식)
        
        🔧 해결 방법:
        1. CustomUnpickler 사용
        2. 모듈 경로 임시 매핑
        3. 글로벌 네임스페이스 주입
        4. 빈 리스트 반환 (최후 수단)
        """
        
        if not os.path.exists(file_path):
            logger.warning(f"⚠️ Pickle 파일이 없습니다: {file_path}")
            return []
        
        file_size = os.path.getsize(file_path)
        logger.debug(f"📊 Pickle 파일 크기: {file_size} bytes")
        
        if file_size < 10:
            logger.warning(f"⚠️ Pickle 파일이 너무 작습니다: {file_size} bytes")
            return []
        
        # 방법 1: CustomUnpickler 사용
        try:
            with open(file_path, 'rb') as f:
                unpickler = CustomUnpickler(f)
                data = unpickler.load()
                
            logger.info(f"✅ CustomUnpickler로 로드 성공: {file_path} ({len(data) if isinstance(data, list) else 'unknown'}개)")
            return data if isinstance(data, list) else []
            
        except Exception as e1:
            logger.warning(f"⚠️ CustomUnpickler 실패: {e1}")
            
            # 방법 2: 모듈 경로 임시 매핑
            try:
                current_module = sys.modules[__name__]
                temp_mappings = {}
                
                old_paths = [
                    'app.llm.main_llm',
                    'app.services.embedding_service',
                    'app.integration_test.embedding_service',
                    'embedding_service',
                    '__main__'
                ]
                
                # 임시 모듈 매핑 설정
                for old_path in old_paths:
                    if old_path not in sys.modules:
                        temp_mappings[old_path] = None
                        sys.modules[old_path] = current_module
                
                # pickle 로드 시도
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                # 임시 매핑 정리
                for old_path in temp_mappings:
                    if temp_mappings[old_path] is None:
                        sys.modules.pop(old_path, None)
                
                logger.info(f"✅ 모듈 매핑으로 로드 성공: {file_path} ({len(data) if isinstance(data, list) else 'unknown'}개)")
                return data if isinstance(data, list) else []
                
            except Exception as e2:
                logger.warning(f"⚠️ 모듈 매핑도 실패: {e2}")
                
                # 방법 3: 글로벌 네임스페이스 주입
                try:
                    import builtins
                    
                    # 현재 모듈의 클래스들을 글로벌에 임시 추가
                    current_module = sys.modules[__name__]
                    temp_globals = {}
                    
                    if hasattr(current_module, 'RAGDocument'):
                        builtins.RAGDocument = getattr(current_module, 'RAGDocument')
                        temp_globals['RAGDocument'] = True
                    if hasattr(current_module, 'RAGContentType'):
                        builtins.RAGContentType = getattr(current_module, 'RAGContentType')
                        temp_globals['RAGContentType'] = True
                    
                    # pickle 로드 시도
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                    
                    # 글로벌 네임스페이스 정리
                    for attr_name in temp_globals:
                        if hasattr(builtins, attr_name):
                            delattr(builtins, attr_name)
                    
                    logger.info(f"✅ 글로벌 주입으로 로드 성공: {file_path} ({len(data) if isinstance(data, list) else 'unknown'}개)")
                    return data if isinstance(data, list) else []
                    
                except Exception as e3:
                    logger.error(f"❌ 모든 pickle 로드 방법 실패: {e3}")
                    logger.debug(f"🔍 최종 오류 상세: {str(e3)}")
                    
                    # 최후 수단: 빈 리스트 반환
                    logger.warning(f"🔄 빈 리스트로 fallback: {file_path}")
                    return []
    
    def _load_config(self):
        """설정 파일 로드"""
        config_path = os.path.join(self.INDEX_DIR, self.CONFIG_FILE)
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                logger.info(f"✅ 설정 파일 로드 완료: {self.CONFIG_FILE}")
                self.load_status["config"] = {"status": "success"}
            except Exception as e:
                logger.error(f"❌ 설정 파일 로드 실패: {e}")
                self.load_status["config"] = {"status": "load_failed", "error": str(e)}
        else:
            logger.info(f"ℹ️ 설정 파일이 없습니다: {self.CONFIG_FILE}")
            self.load_status["config"] = {"status": "file_not_found"}
    
    def _summarize_results(self) -> Dict[str, Any]:
        """결과 요약 생성"""
        total_vectors = sum(
            index.ntotal for index in self.indexes.values() 
            if hasattr(index, 'ntotal')
        )
        
        total_metadata_items = sum(
            len(metadata) for metadata in self.metadata.values()
            if isinstance(metadata, list)
        )
        
        successful_indexes = len([
            status for status in self.load_status.values()
            if status.get("status") == "success"
        ])
        
        return {
            "summary": {
                "total_indexes_loaded": len(self.indexes),
                "total_metadata_loaded": len(self.metadata),
                "total_vectors": total_vectors,
                "total_metadata_items": total_metadata_items,
                "successful_loads": successful_indexes,
                "load_timestamp": datetime.now().isoformat()
            },
            "indexes": self.indexes,
            "metadata": self.metadata,
            "config": self.config,
            "load_status": self.load_status
        }
    
    def _log_summary(self):
        """로딩 결과 요약 로깅"""
        logger.info("📊 === 통합 인덱스 로딩 결과 요약 ===")
        
        # 인덱스별 상태
        for name, index in self.indexes.items():
            if hasattr(index, 'ntotal'):
                logger.info(f"   🔍 {name}: {index.ntotal:,}개 벡터")
        
        # 메타데이터별 상태
        for name, metadata in self.metadata.items():
            if isinstance(metadata, list):
                logger.info(f"   📋 {name}: {len(metadata):,}개 항목")
        
        # 전체 통계
        total_vectors = sum(index.ntotal for index in self.indexes.values() if hasattr(index, 'ntotal'))
        total_items = sum(len(m) for m in self.metadata.values() if isinstance(m, list))
        
        logger.info(f"   📊 총 벡터: {total_vectors:,}개")
        logger.info(f"   📊 총 메타데이터: {total_items:,}개")
        logger.info(f"   📊 로드된 인덱스: {len(self.indexes)}개")
        logger.info("==========================================")
    
    def _create_empty_result(self) -> Dict[str, Any]:
        """빈 결과 생성"""
        return {
            "summary": {
                "total_indexes_loaded": 0,
                "total_metadata_loaded": 0,
                "total_vectors": 0,
                "total_metadata_items": 0,
                "successful_loads": 0,
                "load_timestamp": datetime.now().isoformat()
            },
            "indexes": {},
            "metadata": {},
            "config": {},
            "load_status": {"error": "index_directory_not_found"}
        }

# =============================================================================
# 🔄 개선된 RAG 인덱스 매니저 - 통합 로더 사용
# =============================================================================

class EnhancedRAGIndexManager:
    """
    개선된 RAG 인덱스 매니저 - 모든 인덱스 활용
    
    🚀 기능:
    - 통합 인덱스 로더 사용
    - 모든 FAISS 인덱스 활용 (RAG + 질병 + 의약품)
    - 향상된 검색 성능
    - 멀티 인덱스 통합 검색
    
    📋 지원 검색:
    - RAG Q&A 검색
    - 의료 문서 검색
    - 질병 정보 검색
    - 의약품 정보 검색
    - 통합 검색 (모든 인덱스 동시 검색)
    """
    
    def __init__(self, embedding_model: EmbeddingModel):
        """개선된 RAG 매니저 초기화"""
        self.embedding_model = embedding_model
        
        # 통합 로더 사용
        self.unified_loader = UnifiedIndexLoader()
        
        # 인덱스들
        self.qa_index = None
        self.medical_doc_index = None
        self.disease_key_index = None
        self.disease_full_index = None
        self.medication_index = None
        
        # 메타데이터들
        self.qa_documents = []
        self.medical_documents = []
        self.disease_metadata = []
        self.medication_metadata = []
        
        # 설정
        self.config = {}
        
        logger.info("🔍 개선된 RAG 매니저 초기화: 모든 인덱스 활용")
    
    def load_rag_data(self):
        """모든 RAG 데이터 로드 (통합 방식)"""
        logger.info("🔄 통합 RAG 데이터 로딩 시작...")
        
        # 🚀 통합 로더로 모든 인덱스 로드
        result = self.unified_loader.load_all_indexes()
        
        # 📋 인덱스 할당
        self._assign_indexes(result["indexes"])
        
        # 📋 메타데이터 할당
        self._assign_metadata(result["metadata"])
        
        # 📋 설정 할당
        self.config = result["config"]
        
        # 📊 결과 로깅
        self._log_final_status()
        
        logger.info("✅ 통합 RAG 데이터 로딩 완료!")
    
    def _assign_indexes(self, indexes: Dict[str, Any]):
        """인덱스 할당"""
        self.qa_index = indexes.get("rag_qa")
        self.medical_doc_index = indexes.get("rag_medical")
        self.disease_key_index = indexes.get("disease_key")
        self.disease_full_index = indexes.get("disease_full")
        self.medication_index = indexes.get("medication")
        
        logger.info("✅ 모든 인덱스 할당 완료")
    
    def _assign_metadata(self, metadata: Dict[str, Any]):
        """메타데이터 할당"""
        self.qa_documents = metadata.get("rag_qa", [])
        self.medical_documents = metadata.get("rag_medical", [])
        self.disease_metadata = metadata.get("disease", [])
        self.medication_metadata = metadata.get("medication", [])
        
        logger.info("✅ 모든 메타데이터 할당 완료")
    
    def _log_final_status(self):
        """최종 상태 로깅"""
        logger.info("📊 === 최종 인덱스 상태 ===")
        
        # 각 인덱스 상태
        indexes_info = [
            ("RAG Q&A", self.qa_index, len(self.qa_documents)),
            ("RAG Medical", self.medical_doc_index, len(self.medical_documents)),
            ("Disease Key", self.disease_key_index, len(self.disease_metadata)),
            ("Disease Full", self.disease_full_index, 0),
            ("Medication", self.medication_index, len(self.medication_metadata))
        ]
        
        for name, index, metadata_count in indexes_info:
            if index:
                logger.info(f"   ✅ {name}: {index.ntotal:,}개 벡터, {metadata_count:,}개 메타데이터")
            else:
                logger.info(f"   ❌ {name}: 로드 실패")
        
        # 전체 통계
        total_vectors = sum(
            index.ntotal for index in [
                self.qa_index, self.medical_doc_index, 
                self.disease_key_index, self.disease_full_index, 
                self.medication_index
            ] if index is not None
        )
        
        total_metadata = len(self.qa_documents) + len(self.medical_documents) + \
                        len(self.disease_metadata) + len(self.medication_metadata)
        
        logger.info(f"   📊 전체 벡터: {total_vectors:,}개")
        logger.info(f"   📊 전체 메타데이터: {total_metadata:,}개")
        logger.info("===========================")
    
    # =============================================================================
    # 🔍 기본 검색 메서드들
    # =============================================================================
    
    def search_qa(self, query: str, top_k: int = 5) -> List[Tuple[Any, float]]:
        """RAG Q&A 검색"""
        if not self.qa_index or not self.qa_documents:
            logger.warning("⚠️ RAG Q&A 인덱스가 없습니다")
            return []
        
        try:
            query_embedding = self.embedding_model.encode_single(query)
            scores, indices = self.qa_index.search(
                query_embedding.reshape(1, -1).astype('float32'), top_k
            )
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if 0 <= idx < len(self.qa_documents):
                    results.append((self.qa_documents[idx], float(score)))
            
            return results
            
        except Exception as e:
            logger.error(f"❌ RAG Q&A 검색 실패: {e}")
            return []
    
    def search_medical_docs(self, query: str, top_k: int = 5) -> List[Tuple[Any, float]]:
        """RAG 의료 문서 검색"""
        if not self.medical_doc_index or not self.medical_documents:
            logger.warning("⚠️ RAG Medical 인덱스가 없습니다")
            return []
        
        try:
            query_embedding = self.embedding_model.encode_single(query)
            scores, indices = self.medical_doc_index.search(
                query_embedding.reshape(1, -1).astype('float32'), top_k
            )
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if 0 <= idx < len(self.medical_documents):
                    results.append((self.medical_documents[idx], float(score)))
            
            return results
            
        except Exception as e:
            logger.error(f"❌ RAG Medical 검색 실패: {e}")
            return []
    
    def search_disease_advanced(self, query: str, top_k: int = 5) -> List[Tuple[Any, float]]:
        """고급 질병 검색 (disease 인덱스 활용)"""
        if not self.disease_key_index:
            logger.warning("⚠️ Disease 인덱스가 없습니다")
            return []
        
        try:
            query_embedding = self.embedding_model.encode_single(query)
            scores, indices = self.disease_key_index.search(
                query_embedding.reshape(1, -1).astype('float32'), top_k
            )
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if 0 <= idx < len(self.disease_metadata):
                    results.append((self.disease_metadata[idx], float(score)))
            
            return results
            
        except Exception as e:
            logger.error(f"❌ 고급 질병 검색 실패: {e}")
            return []
    
    def search_medication_advanced(self, query: str, top_k: int = 5) -> List[Tuple[Any, float]]:
        """고급 의약품 검색 (medication 인덱스 활용)"""
        if not self.medication_index:
            logger.warning("⚠️ Medication 인덱스가 없습니다")
            return []
        
        try:
            query_embedding = self.embedding_model.encode_single(query)
            scores, indices = self.medication_index.search(
                query_embedding.reshape(1, -1).astype('float32'), top_k
            )
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if 0 <= idx < len(self.medication_metadata):
                    results.append((self.medication_metadata[idx], float(score)))
            
            return results
            
        except Exception as e:
            logger.error(f"❌ 고급 의약품 검색 실패: {e}")
            return []
    
    # =============================================================================
    # 🔄 통합 검색 메서드들
    # =============================================================================
    
    def search_unified(self, query: str, top_k: int = 3) -> Dict[str, List[Tuple[Any, float]]]:
        """통합 검색 - 모든 인덱스 활용"""
        return {
            "qa_results": self.search_qa(query, top_k),
            "medical_results": self.search_medical_docs(query, top_k),
            "disease_results": self.search_disease_advanced(query, top_k),
            "medication_results": self.search_medication_advanced(query, top_k)
        }
    
    def search_combined(self, query: str, qa_top_k: int = 3, medical_top_k: int = 3) -> Dict[str, List[Tuple[Any, float]]]:
        """기존 호환성을 위한 통합 검색 (Q&A + 의료 문서)"""
        return {
            "qa_results": self.search_qa(query, qa_top_k),
            "medical_results": self.search_medical_docs(query, medical_top_k)
        }
    
    def search_best_match(self, query: str, search_types: List[str] = None) -> Tuple[Any, float, str]:
        """
        최고 점수 결과 반환 (모든 인덱스에서)
        
        Args:
            query: 검색 쿼리
            search_types: 검색할 인덱스 타입 리스트 (None이면 모든 타입)
            
        Returns:
            Tuple[결과, 점수, 인덱스_타입]
        """
        if search_types is None:
            search_types = ["qa", "medical", "disease", "medication"]
        
        all_results = []
        
        # 각 인덱스에서 최고 점수 결과 수집
        if "qa" in search_types:
            qa_results = self.search_qa(query, 1)
            if qa_results:
                all_results.append((qa_results[0][0], qa_results[0][1], "qa"))
        
        if "medical" in search_types:
            medical_results = self.search_medical_docs(query, 1)
            if medical_results:
                all_results.append((medical_results[0][0], medical_results[0][1], "medical"))
        
        if "disease" in search_types:
            disease_results = self.search_disease_advanced(query, 1)
            if disease_results:
                all_results.append((disease_results[0][0], disease_results[0][1], "disease"))
        
        if "medication" in search_types:
            medication_results = self.search_medication_advanced(query, 1)
            if medication_results:
                all_results.append((medication_results[0][0], medication_results[0][1], "medication"))
        
        if not all_results:
            return None, 0.0, "none"
        
        # 최고 점수 결과 반환
        best_result = max(all_results, key=lambda x: x[1])
        return best_result
    
    # =============================================================================
    # 🔧 유틸리티 메서드들
    # =============================================================================
    
    def get_index_status(self) -> Dict[str, Any]:
        """인덱스 상태 정보 반환"""
        return {
            "indexes_loaded": {
                "qa": self.qa_index is not None,
                "medical": self.medical_doc_index is not None,
                "disease_key": self.disease_key_index is not None,
                "disease_full": self.disease_full_index is not None,
                "medication": self.medication_index is not None
            },
            "vector_counts": {
                "qa": self.qa_index.ntotal if self.qa_index else 0,
                "medical": self.medical_doc_index.ntotal if self.medical_doc_index else 0,
                "disease_key": self.disease_key_index.ntotal if self.disease_key_index else 0,
                "disease_full": self.disease_full_index.ntotal if self.disease_full_index else 0,
                "medication": self.medication_index.ntotal if self.medication_index else 0
            },
            "metadata_counts": {
                "qa": len(self.qa_documents),
                "medical": len(self.medical_documents),
                "disease": len(self.disease_metadata),
                "medication": len(self.medication_metadata)
            },
            "config": self.config
        }

# =============================================================================
# 🔄 기존 코드와의 호환성 유지
# =============================================================================

# 기존 RAGIndexManager를 EnhancedRAGIndexManager로 대체하되, 기존 인터페이스 유지
RAGIndexManager = EnhancedRAGIndexManager

# =============================================================================
# 🚀 메인 초기화 함수
# =============================================================================

def initialize_embedding_service() -> Tuple[EmbeddingModel, EnhancedRAGIndexManager]:
    """
    임베딩 서비스 초기화
    
    Returns:
        Tuple[EmbeddingModel, EnhancedRAGIndexManager]: 임베딩 모델과 RAG 매니저
    """
    logger.info("🚀 임베딩 서비스 초기화 시작...")
    
    try:
        # 임베딩 모델 초기화
        embedding_model = EmbeddingModel()
        
        # RAG 매니저 초기화
        rag_manager = EnhancedRAGIndexManager(embedding_model)
        
        # RAG 데이터 로드
        rag_manager.load_rag_data()
        
        logger.info("✅ 임베딩 서비스 초기화 완료!")
        
        return embedding_model, rag_manager
        
    except Exception as e:
        logger.error(f"❌ 임베딩 서비스 초기화 실패: {e}")
        raise e

# =============================================================================
# 🔧 모듈 테스트 코드
# =============================================================================

if __name__ == "__main__":
    """모듈 테스트 실행"""
    print("🧪 임베딩 서비스 모듈 테스트 시작...")
    
    try:
        # 서비스 초기화
        embedding_model, rag_manager = initialize_embedding_service()
        
        # 간단한 검색 테스트
        test_query = "두통이 있어요"
        
        print(f"\n🔍 테스트 쿼리: '{test_query}'")
        
        # 통합 검색 테스트
        unified_results = rag_manager.search_unified(test_query, top_k=2)
        
        for search_type, results in unified_results.items():
            print(f"\n📋 {search_type}: {len(results)}개 결과")
            for i, (doc, score) in enumerate(results[:2]):
                print(f"   {i+1}. 점수: {score:.4f}")
                if hasattr(doc, 'content'):
                    print(f"      내용: {doc.content[:100]}...")
                else:
                    print(f"      내용: {str(doc)[:100]}...")
        
        # 인덱스 상태 확인
        status = rag_manager.get_index_status()
        print(f"\n📊 인덱스 상태:")
        print(f"   로드된 인덱스: {sum(status['indexes_loaded'].values())}개")
        print(f"   총 벡터 수: {sum(status['vector_counts'].values()):,}개")
        print(f"   총 메타데이터: {sum(status['metadata_counts'].values()):,}개")
        
        print("\n✅ 모든 테스트 완료!")
        
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()