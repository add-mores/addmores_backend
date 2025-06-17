# /faiss_manager.py
"""
FAISS 인덱스 저장/로드 매니저
exaone_v6.txt 기반 - 기존 RAGIndexManager 확장
"""

import os
import json
import faiss
import pickle
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class FAISSIndexSaver:
    """
    FAISS 인덱스 저장/로드 관리자
    - 기존 exaone_v6 코드의 모든 인덱스를 디스크에 저장
    - 서버 재시작 시 빠른 로드로 성능 향상
    """
    
    def __init__(self, index_dir: str = "data/indexes"):
        """
        Args:
            index_dir: 인덱스 저장 디렉토리
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # 인덱스 파일 경로
        self.paths = {
            # RAG 인덱스들
            'qa_index': self.index_dir / 'qa_index.faiss',
            'medical_doc_index': self.index_dir / 'medical_doc_index.faiss',
            
            # 질병 인덱스들  
            'disease_key_index': self.index_dir / 'disease_key_index.faiss',
            'disease_full_index': self.index_dir / 'disease_full_index.faiss',
            
            # 의약품 인덱스
            'medication_index': self.index_dir / 'medication_index.faiss',
            
            # 메타데이터
            'qa_documents': self.index_dir / 'qa_documents.pkl',
            'medical_documents': self.index_dir / 'medical_documents.pkl',
            'disease_metadata': self.index_dir / 'disease_metadata.pkl',
            'medication_metadata': self.index_dir / 'medication_metadata.pkl',
            'hospital_data': self.index_dir / 'hospital_data.pkl',
            
            # 시스템 정보
            'index_info': self.index_dir / 'index_info.json'
        }
        
        logger.info(f"📁 FAISS 인덱스 저장소 초기화: {self.index_dir}")

    def save_all_indexes(self, rag_manager, disease_key_index, disease_full_index, 
                        disease_metadata, medication_index, medication_metadata, hospital_data):
        """
        모든 FAISS 인덱스와 메타데이터를 파일로 저장
        exaone_v6.txt의 모든 데이터 구조 지원
        """
        
        try:
            save_start = datetime.now()
            logger.info("💾 전체 인덱스 저장 시작...")
            
            # 1) RAG 인덱스 저장
            self._save_rag_indexes(rag_manager)
            
            # 2) 질병 인덱스 저장
            self._save_disease_indexes(disease_key_index, disease_full_index, disease_metadata)
            
            # 3) 의약품 인덱스 저장
            self._save_medication_index(medication_index, medication_metadata)
            
            # 4) 병원 데이터 저장
            self._save_hospital_data(hospital_data)
            
            # 5) 인덱스 정보 저장
            self._save_index_info()
            
            save_time = (datetime.now() - save_start).total_seconds()
            logger.info(f"✅ 전체 인덱스 저장 완료! ({save_time:.2f}초)")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 인덱스 저장 실패: {e}")
            return False

    def _save_rag_indexes(self, rag_manager):
        """RAG 인덱스 저장 (qa_index, medical_doc_index)"""
        
        # Q&A 인덱스
        if rag_manager.qa_index is not None:
            faiss.write_index(rag_manager.qa_index, str(self.paths['qa_index']))
            logger.info(f"💾 Q&A 인덱스 저장: {self.paths['qa_index']}")
        
        # 의료 문서 인덱스
        if rag_manager.medical_doc_index is not None:
            faiss.write_index(rag_manager.medical_doc_index, str(self.paths['medical_doc_index']))
            logger.info(f"💾 의료문서 인덱스 저장: {self.paths['medical_doc_index']}")
        
        # RAG 문서 메타데이터
        with open(self.paths['qa_documents'], 'wb') as f:
            pickle.dump(rag_manager.qa_documents, f)
        
        with open(self.paths['medical_documents'], 'wb') as f:
            pickle.dump(rag_manager.medical_documents, f)
            
        logger.info(f"💾 RAG 메타데이터 저장: Q&A {len(rag_manager.qa_documents)}개, 의료문서 {len(rag_manager.medical_documents)}개")

    def _save_disease_indexes(self, key_index, full_index, metadata):
        """질병 인덱스 저장 (index_key, index_full, all_docs_meta)"""
        
        # FAISS 인덱스
        if key_index is not None:
            faiss.write_index(key_index, str(self.paths['disease_key_index']))
            logger.info(f"💾 질병 핵심증상 인덱스 저장: {self.paths['disease_key_index']}")
        
        if full_index is not None:
            faiss.write_index(full_index, str(self.paths['disease_full_index']))
            logger.info(f"💾 질병 전체증상 인덱스 저장: {self.paths['disease_full_index']}")
        
        # 메타데이터
        with open(self.paths['disease_metadata'], 'wb') as f:
            pickle.dump(metadata, f)
            
        logger.info(f"💾 질병 메타데이터 저장: {len(metadata)}개")

    def _save_medication_index(self, med_index, med_metadata):
        """의약품 인덱스 저장 (meds_index, meds_meta_list)"""
        
        # FAISS 인덱스
        if med_index is not None:
            faiss.write_index(med_index, str(self.paths['medication_index']))
            logger.info(f"💾 의약품 인덱스 저장: {self.paths['medication_index']}")
        
        # 메타데이터
        with open(self.paths['medication_metadata'], 'wb') as f:
            pickle.dump(med_metadata, f)
            
        logger.info(f"💾 의약품 메타데이터 저장: {len(med_metadata)}개")

    def _save_hospital_data(self, hospital_data):
        """병원 데이터 저장 (df_hosp)"""
        
        with open(self.paths['hospital_data'], 'wb') as f:
            pickle.dump(hospital_data, f)
            
        logger.info(f"💾 병원 데이터 저장: {len(hospital_data)}개")

    def _save_index_info(self):
        """인덱스 생성 정보 저장"""
        
        info = {
            'created_at': datetime.now().isoformat(),
            'version': 'exaone_v6_faiss',
            'indexes': {
                'qa_index': self.paths['qa_index'].exists(),
                'medical_doc_index': self.paths['medical_doc_index'].exists(),
                'disease_key_index': self.paths['disease_key_index'].exists(),
                'disease_full_index': self.paths['disease_full_index'].exists(),
                'medication_index': self.paths['medication_index'].exists()
            },
            'metadata_files': {
                'qa_documents': self.paths['qa_documents'].exists(),
                'medical_documents': self.paths['medical_documents'].exists(),
                'disease_metadata': self.paths['disease_metadata'].exists(),
                'medication_metadata': self.paths['medication_metadata'].exists(),
                'hospital_data': self.paths['hospital_data'].exists()
            }
        }
        
        with open(self.paths['index_info'], 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
            
        logger.info(f"💾 인덱스 정보 저장: {self.paths['index_info']}")

    def load_all_indexes(self):
        """
        저장된 모든 인덱스와 메타데이터 로드
        Returns: (rag_data, disease_data, medication_data, hospital_data)
        """
        
        try:
            load_start = datetime.now()
            logger.info("📂 저장된 인덱스 로드 시작...")
            
            # 인덱스 존재 확인
            if not self._check_indexes_exist():
                logger.warning("⚠️ 일부 인덱스 파일이 없습니다.")
                return None
            
            # 1) RAG 인덱스 로드
            rag_data = self._load_rag_indexes()
            
            # 2) 질병 인덱스 로드
            disease_data = self._load_disease_indexes()
            
            # 3) 의약품 인덱스 로드
            medication_data = self._load_medication_index()
            
            # 4) 병원 데이터 로드
            hospital_data = self._load_hospital_data()
            
            load_time = (datetime.now() - load_start).total_seconds()
            logger.info(f"✅ 전체 인덱스 로드 완료! ({load_time:.2f}초)")
            
            return {
                'rag': rag_data,
                'disease': disease_data,
                'medication': medication_data,
                'hospital': hospital_data
            }
            
        except Exception as e:
            logger.error(f"❌ 인덱스 로드 실패: {e}")
            return None

    def _load_rag_indexes(self):
        """RAG 인덱스 로드"""
        
        rag_data = {}
        
        # FAISS 인덱스 로드
        if self.paths['qa_index'].exists():
            rag_data['qa_index'] = faiss.read_index(str(self.paths['qa_index']))
            
        if self.paths['medical_doc_index'].exists():
            rag_data['medical_doc_index'] = faiss.read_index(str(self.paths['medical_doc_index']))
        
        # 메타데이터 로드
        if self.paths['qa_documents'].exists():
            with open(self.paths['qa_documents'], 'rb') as f:
                rag_data['qa_documents'] = pickle.load(f)
                
        if self.paths['medical_documents'].exists():
            with open(self.paths['medical_documents'], 'rb') as f:
                rag_data['medical_documents'] = pickle.load(f)
        
        logger.info("📂 RAG 인덱스 로드 완료")
        return rag_data

    def _load_disease_indexes(self):
        """질병 인덱스 로드"""
        
        disease_data = {}
        
        # FAISS 인덱스 로드
        if self.paths['disease_key_index'].exists():
            disease_data['key_index'] = faiss.read_index(str(self.paths['disease_key_index']))
            
        if self.paths['disease_full_index'].exists():
            disease_data['full_index'] = faiss.read_index(str(self.paths['disease_full_index']))
        
        # 메타데이터 로드
        if self.paths['disease_metadata'].exists():
            with open(self.paths['disease_metadata'], 'rb') as f:
                disease_data['metadata'] = pickle.load(f)
        
        logger.info("📂 질병 인덱스 로드 완료")
        return disease_data

    def _load_medication_index(self):
        """의약품 인덱스 로드"""
        
        medication_data = {}
        
        # FAISS 인덱스 로드
        if self.paths['medication_index'].exists():
            medication_data['index'] = faiss.read_index(str(self.paths['medication_index']))
        
        # 메타데이터 로드
        if self.paths['medication_metadata'].exists():
            with open(self.paths['medication_metadata'], 'rb') as f:
                medication_data['metadata'] = pickle.load(f)
        
        logger.info("📂 의약품 인덱스 로드 완료")
        return medication_data

    def _load_hospital_data(self):
        """병원 데이터 로드"""
        
        if self.paths['hospital_data'].exists():
            with open(self.paths['hospital_data'], 'rb') as f:
                hospital_data = pickle.load(f)
                
            logger.info("📂 병원 데이터 로드 완료")
            return hospital_data
        
        return None

    def _check_indexes_exist(self):
        """필수 인덱스 파일들이 존재하는지 확인"""
        
        required_files = [
            'qa_index', 'medical_doc_index',
            'disease_key_index', 'disease_full_index', 
            'medication_index'
        ]
        
        missing_files = []
        for file_key in required_files:
            if not self.paths[file_key].exists():
                missing_files.append(file_key)
        
        if missing_files:
            logger.warning(f"⚠️ 누락된 인덱스 파일: {missing_files}")
            return False
            
        return True

    def get_index_info(self):
        """저장된 인덱스 정보 조회"""
        
        if self.paths['index_info'].exists():
            with open(self.paths['index_info'], 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def clear_all_indexes(self):
        """모든 저장된 인덱스 파일 삭제"""
        
        try:
            deleted_count = 0
            for file_path in self.paths.values():
                if file_path.exists():
                    file_path.unlink()
                    deleted_count += 1
            
            logger.info(f"🗑️ 인덱스 파일 {deleted_count}개 삭제 완료")
            return True
            
        except Exception as e:
            logger.error(f"❌ 인덱스 삭제 실패: {e}")
            return False

# =============================================================================
# RAGIndexManager 확장 클래스 - 저장/로드 기능 추가
# =============================================================================

class RAGIndexManagerWithStorage:
    """
    기존 RAGIndexManager에 저장/로드 기능 추가
    exaone_v6.txt 코드와 완전 호환
    """
    
    def __init__(self, embedding_model, index_dir: str = "data/indexes"):
        # 기존 속성들
        self.embedding_model = embedding_model
        self.qa_index = None
        self.medical_doc_index = None
        self.qa_documents = []
        self.medical_documents = []
        
        # 저장 관리자
        self.faiss_saver = FAISSIndexSaver(index_dir)
        
    def load_rag_data(self):
        """
        RAG 데이터 로드 - 저장된 인덱스 우선 사용
        기존 exaone_v6.txt의 load_rag_data() 메서드와 동일한 인터페이스
        """
        
        # 1) 저장된 인덱스가 있으면 로드
        if self._try_load_saved_indexes():
            logger.info("✅ 저장된 RAG 인덱스 로드 완료!")
            return
        
        # 2) 저장된 인덱스가 없으면 새로 구축
        logger.info("📂 저장된 인덱스가 없습니다. 새로 구축합니다...")
        self._load_rag_data_from_csv()
        
        # 3) 새로 구축한 인덱스 저장
        self._save_rag_indexes()

    def _try_load_saved_indexes(self):
        """저장된 RAG 인덱스 로드 시도"""
        
        try:
            rag_data = self.faiss_saver._load_rag_indexes()
            
            if rag_data and 'qa_index' in rag_data and 'medical_doc_index' in rag_data:
                self.qa_index = rag_data['qa_index']
                self.medical_doc_index = rag_data['medical_doc_index']
                self.qa_documents = rag_data.get('qa_documents', [])
                self.medical_documents = rag_data.get('medical_documents', [])
                
                logger.info(f"📂 RAG 인덱스 로드: Q&A {len(self.qa_documents)}개, 의료문서 {len(self.medical_documents)}개")
                return True
                
        except Exception as e:
            logger.warning(f"⚠️ 저장된 인덱스 로드 실패: {e}")
            
        return False

    def _load_rag_data_from_csv(self):
        """
        CSV에서 RAG 데이터 로드 (기존 exaone_v6.txt 로직)
        """
        print("🔄 RAG 데이터 로딩 시작...")
        
        # Q&A 데이터 로드 (clean_51004.csv)
        self._load_qa_data()
        
        # 의료 문서 데이터 로드 (나머지 5개 clean 파일들)
        self._load_medical_documents()
        
        # 인덱스 구축
        self._build_indexes()
        
        print("✅ RAG 데이터 로딩 완료!")

    def _save_rag_indexes(self):
        """현재 RAG 인덱스들을 디스크에 저장"""
        
        try:
            self.faiss_saver._save_rag_indexes(self)
            logger.info("💾 RAG 인덱스 저장 완료")
            
        except Exception as e:
            logger.error(f"❌ RAG 인덱스 저장 실패: {e}")
    
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
    
    def search_qa(self, query: str, top_k: int = 3):
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
    
    def search_medical_docs(self, query: str, top_k: int = 3):
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