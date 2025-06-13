"""
질병 RAG 검색 서비스
위치: ~/backend/app/services/disease_rag_service.py

🎯 목적: clean_ 파일들의 RAG 검색 기능 제공
📋 기능:
   - Q&A RAG 검색 (clean_51004.csv)
   - 의료 문서 RAG 검색 (clean_55588~66149.csv)
   - FAISS 벡터 검색
   - 컨텍스트 문서 반환

⚙️ 의존성: faiss, numpy, logging
"""

import faiss
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .disease_embedding_service import get_embedding_service
from .disease_faiss_loader import get_faiss_loader
from ..utils.disease_exceptions import RagSearchError, RagIndexNotLoadedError

logger = logging.getLogger(__name__)


class RAGContentType(Enum):
    """RAG 컨텐츠 타입"""
    QA = "qa"                    # Q&A 형태 데이터
    MEDICAL_DOC = "medical_doc"  # 의료 문서 데이터


@dataclass
class RAGDocument:
    """RAG 문서 데이터 클래스"""
    doc_id: str
    content: str
    metadata: Dict
    content_type: RAGContentType
    score: float = 0.0


class DiseaseRAGService:
    """질병 RAG 검색 서비스 클래스"""
    
    def __init__(self):
        self.embedding_service = get_embedding_service()
        self.faiss_loader = get_faiss_loader()
        self.is_initialized = False
        
        # RAG 인덱스와 메타데이터
        self.qa_index: Optional[faiss.IndexFlatIP] = None
        self.medical_index: Optional[faiss.IndexFlatIP] = None
        self.qa_metadata: List[Dict] = []
        self.medical_metadata: List[Dict] = []
        
        logger.info("🔍 RAG 검색 서비스 초기화됨")
    
    def initialize(self) -> bool:
        """RAG 서비스 초기화"""
        try:
            logger.info("🔄 RAG 서비스 초기화 중...")
            
            # 임베딩 서비스 확인
            if not self.embedding_service.is_loaded:
                raise RagIndexNotLoadedError("embedding_service")
            
            # FAISS 로더 확인
            if not self.faiss_loader.is_loaded:
                raise RagIndexNotLoadedError("faiss_loader")
            
            # RAG 인덱스와 메타데이터 로드
            self.qa_index, self.medical_index = self.faiss_loader.get_rag_indexes()
            self.qa_metadata, self.medical_metadata = self.faiss_loader.get_rag_metadata()
            
            # 상태 로깅
            self._log_rag_status()
            
            self.is_initialized = True
            logger.info("✅ RAG 서비스 초기화 완료!")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ RAG 서비스 초기화 실패: {e}")
            self.is_initialized = False
            raise RagIndexNotLoadedError(f"RAG 서비스 초기화 실패: {e}")
    
    def _log_rag_status(self):
        """RAG 상태 로깅"""
        logger.info("📊 RAG 데이터 상태:")
        
        if self.qa_index:
            logger.info(f"   - Q&A 인덱스: {self.qa_index.ntotal}개 벡터")
            logger.info(f"   - Q&A 메타데이터: {len(self.qa_metadata)}개 문서")
        else:
            logger.warning("   - Q&A 인덱스: 없음")
        
        if self.medical_index:
            logger.info(f"   - 의료 문서 인덱스: {self.medical_index.ntotal}개 벡터")
            logger.info(f"   - 의료 문서 메타데이터: {len(self.medical_metadata)}개 문서")
        else:
            logger.warning("   - 의료 문서 인덱스: 없음")
    
    def search_qa(self, query: str, top_k: int = 3) -> List[RAGDocument]:
        """Q&A RAG 검색"""
        if not self.is_initialized:
            raise RagSearchError("", "RAG 서비스가 초기화되지 않았습니다.")
        
        if not self.qa_index or not self.qa_metadata:
            logger.warning("⚠️ Q&A 인덱스가 없습니다.")
            return []
        
        try:
            logger.debug(f"🔍 Q&A 검색: '{query}' (top_k={top_k})")
            
            # 쿼리 임베딩 생성
            query_embedding = self.embedding_service.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # FAISS 검색
            scores, indices = self.qa_index.search(query_embedding, top_k)
            
            # 결과 구성
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if 0 <= idx < len(self.qa_metadata):
                    doc_meta = self.qa_metadata[idx]
                    
                    # Q&A 문서 생성
                    question = doc_meta.get('question', '')
                    answer = doc_meta.get('answer', '')
                    content = f"질문: {question}\n답변: {answer}"
                    
                    doc = RAGDocument(
                        doc_id=doc_meta.get('doc_id', f"qa_{idx}"),
                        content=content,
                        metadata=doc_meta,
                        content_type=RAGContentType.QA,
                        score=float(score)
                    )
                    results.append(doc)
            
            logger.debug(f"✅ Q&A 검색 완료: {len(results)}개 결과")
            return results
            
        except Exception as e:
            logger.error(f"❌ Q&A 검색 실패: {e}")
            raise RagSearchError(query, f"Q&A 검색 실패: {e}")
    
    def search_medical_docs(self, query: str, top_k: int = 3) -> List[RAGDocument]:
        """의료 문서 RAG 검색"""
        if not self.is_initialized:
            raise RagSearchError("", "RAG 서비스가 초기화되지 않았습니다.")
        
        if not self.medical_index or not self.medical_metadata:
            logger.warning("⚠️ 의료 문서 인덱스가 없습니다.")
            return []
        
        try:
            logger.debug(f"🔍 의료 문서 검색: '{query}' (top_k={top_k})")
            
            # 쿼리 임베딩 생성
            query_embedding = self.embedding_service.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # FAISS 검색
            scores, indices = self.medical_index.search(query_embedding, top_k)
            
            # 결과 구성
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if 0 <= idx < len(self.medical_metadata):
                    doc_meta = self.medical_metadata[idx]
                    
                    # 의료 문서 생성
                    disease_name = doc_meta.get('disease_name', '')
                    section_title = doc_meta.get('section_title', '')
                    content_text = doc_meta.get('content', '')
                    
                    content_parts = []
                    if disease_name:
                        content_parts.append(f"질병: {disease_name}")
                    if section_title:
                        content_parts.append(f"섹션: {section_title}")
                    if content_text:
                        content_parts.append(f"내용: {content_text}")
                    
                    content = "\n".join(content_parts)
                    
                    doc = RAGDocument(
                        doc_id=doc_meta.get('doc_id', f"doc_{idx}"),
                        content=content,
                        metadata=doc_meta,
                        content_type=RAGContentType.MEDICAL_DOC,
                        score=float(score)
                    )
                    results.append(doc)
            
            logger.debug(f"✅ 의료 문서 검색 완료: {len(results)}개 결과")
            return results
            
        except Exception as e:
            logger.error(f"❌ 의료 문서 검색 실패: {e}")
            raise RagSearchError(query, f"의료 문서 검색 실패: {e}")
    
    def search_combined(self, query: str, qa_top_k: int = 2, doc_top_k: int = 2) -> Tuple[List[RAGDocument], List[RAGDocument]]:
        """Q&A와 의료 문서 통합 검색"""
        try:
            logger.debug(f"🔍 통합 RAG 검색: '{query}'")
            
            # 두 가지 검색 동시 실행
            qa_results = self.search_qa(query, qa_top_k)
            doc_results = self.search_medical_docs(query, doc_top_k)
            
            logger.debug(f"✅ 통합 검색 완료: Q&A {len(qa_results)}개, 문서 {len(doc_results)}개")
            
            return qa_results, doc_results
            
        except Exception as e:
            logger.error(f"❌ 통합 RAG 검색 실패: {e}")
            raise RagSearchError(query, f"통합 RAG 검색 실패: {e}")
    
    def get_rag_context(self, query: str, max_chars: int = 1000) -> str:
        """검색 결과를 컨텍스트 문자열로 변환"""
        try:
            qa_results, doc_results = self.search_combined(query)
            
            context_parts = []
            
            # Q&A 결과 추가
            if qa_results:
                context_parts.append("💬 관련 Q&A:")
                for i, doc in enumerate(qa_results, 1):
                    question = doc.metadata.get('question', '')[:100]
                    answer = doc.metadata.get('answer', '')[:200]
                    context_parts.append(f"{i}. Q: {question}...")
                    context_parts.append(f"   A: {answer}...")
            
            # 의료 문서 결과 추가
            if doc_results:
                context_parts.append("\n📚 관련 의료 문서:")
                for i, doc in enumerate(doc_results, 1):
                    content = doc.content[:150]
                    context_parts.append(f"{i}. {content}...")
            
            # 전체 컨텍스트 조합
            full_context = "\n".join(context_parts)
            
            # 길이 제한
            if len(full_context) > max_chars:
                full_context = full_context[:max_chars] + "..."
            
            return full_context
            
        except Exception as e:
            logger.error(f"❌ RAG 컨텍스트 생성 실패: {e}")
            return ""
    
    def get_service_status(self) -> Dict:
        """RAG 서비스 상태 반환"""
        status = {
            "is_initialized": self.is_initialized,
            "embedding_service_loaded": self.embedding_service.is_loaded if self.embedding_service else False,
            "faiss_loader_loaded": self.faiss_loader.is_loaded if self.faiss_loader else False
        }
        
        if self.is_initialized:
            status.update({
                "qa_index_available": self.qa_index is not None,
                "medical_index_available": self.medical_index is not None,
                "qa_docs_count": len(self.qa_metadata),
                "medical_docs_count": len(self.medical_metadata)
            })
            
            if self.qa_index:
                status["qa_vectors_count"] = self.qa_index.ntotal
            if self.medical_index:
                status["medical_vectors_count"] = self.medical_index.ntotal
        
        return status


# 전역 RAG 서비스 인스턴스 (싱글톤 패턴)
_global_rag_service: Optional[DiseaseRAGService] = None


def get_rag_service() -> DiseaseRAGService:
    """RAG 서비스 싱글톤 인스턴스 반환"""
    global _global_rag_service
    
    if _global_rag_service is None:
        _global_rag_service = DiseaseRAGService()
    
    return _global_rag_service


def initialize_rag_service() -> bool:
    """RAG 서비스 초기화"""
    try:
        service = get_rag_service()
        return service.initialize()
    except Exception as e:
        logger.error(f"❌ RAG 서비스 초기화 실패: {e}")
        raise