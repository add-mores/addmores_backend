"""
질병 FAISS 인덱스 로더 서비스
위치: ~/backend/app/services/disease_faiss_loader.py

🎯 목적: FAISS 인덱스 파일들을 안전하게 로드하고 관리
📋 기능:
   - FAISS 인덱스 파일 로드 (app/api/disease_faiss_indexes/)
   - 메타데이터 로드
   - 인덱스 유효성 검증
   - 메모리 효율적 로드

⚙️ 의존성: faiss, pickle, logging
"""

import os
import faiss
import pickle
import logging
from typing import Optional, Tuple, List, Dict
from pathlib import Path

from ..utils.disease_exceptions import FaissLoadError, FaissFileNotFoundError

logger = logging.getLogger(__name__)


class DiseaseFAISSLoader:
    """질병 FAISS 인덱스 로더 클래스"""
    
    def __init__(self):
        # 🔄 수정: app/api/disease_faiss_indexes/ 경로로 변경
        self.faiss_dir = self._get_faiss_directory()
        self.indexes = {}
        self.metadata = {}
        self.is_loaded = False
        
        logger.info(f"📁 FAISS 디렉토리: {self.faiss_dir}")
    
    def _get_faiss_directory(self) -> Path:
        """FAISS 디렉토리 경로 자동 감지 (app/api/disease_faiss_indexes/)"""
        try:
            # 현재 파일 위치 기반으로 경로 계산
            current_file = Path(__file__).resolve()
            
            # 방법 1: app/services/disease_faiss_loader.py → app/api/disease_faiss_indexes/
            app_dir = current_file.parent.parent  # app/
            api_faiss_dir = app_dir / "api" / "disease_faiss_indexes"
            
            if api_faiss_dir.exists():
                logger.info(f"✅ FAISS 디렉토리 발견: {api_faiss_dir}")
                return api_faiss_dir
            
            # 방법 2: 다양한 상대 경로 시도
            possible_paths = [
                current_file.parent.parent / "api" / "disease_faiss_indexes",  # app/services/ → app/api/
                current_file.parent / "api" / "disease_faiss_indexes",         # app/
                Path.cwd() / "app" / "api" / "disease_faiss_indexes",          # 프로젝트 루트에서
                Path.cwd() / "backend" / "app" / "api" / "disease_faiss_indexes"  # 상위에서
            ]
            
            for path in possible_paths:
                if path.exists():
                    logger.info(f"✅ FAISS 디렉토리 발견: {path}")
                    return path
            
            # 방법 3: 환경변수
            env_path = os.environ.get("DISEASE_FAISS_DIR")
            if env_path:
                env_faiss_dir = Path(env_path)
                if env_faiss_dir.exists():
                    logger.info(f"✅ 환경변수에서 FAISS 디렉토리: {env_faiss_dir}")
                    return env_faiss_dir
            
            # 기본값 반환 (없어도 반환)
            default_path = app_dir / "api" / "disease_faiss_indexes"
            logger.warning(f"⚠️ FAISS 디렉토리를 찾을 수 없습니다. 기본 경로 사용: {default_path}")
            return default_path
            
        except Exception as e:
            logger.error(f"❌ FAISS 디렉토리 경로 설정 오류: {e}")
            # 최후 fallback
            return Path.cwd() / "app" / "api" / "disease_faiss_indexes"
    
    def load_all_indexes(self) -> bool:
        """모든 FAISS 인덱스와 메타데이터 로드"""
        logger.info("🔄 FAISS 인덱스 로딩 시작...")
        
        try:
            # 1단계: 디렉토리 존재 확인
            logger.info("1️⃣ 디렉토리 존재 확인...")
            if not self.faiss_dir.exists():
                raise FaissLoadError(str(self.faiss_dir), f"FAISS 디렉토리가 존재하지 않습니다: {self.faiss_dir}")
            logger.info("✅ 디렉토리 존재 확인 완료")
            
            # 2단계: 필수 파일 확인
            logger.info("2️⃣ 필수 파일 확인...")
            self._validate_required_files()
            
            # 3단계: 질병 인덱스 로드
            logger.info("3️⃣ 질병 인덱스 로드...")
            self._load_disease_indexes()
            
            # 4단계: RAG 인덱스 로드
            logger.info("4️⃣ RAG 인덱스 로드...")
            self._load_rag_indexes()
            
            # 5단계: 메타데이터 로드
            logger.info("5️⃣ 메타데이터 로드...")
            self._load_metadata()
            
            # 6단계: 로드 상태 검증
            logger.info("6️⃣ 로드된 데이터 검증...")
            self._validate_loaded_data()
            
            self.is_loaded = True
            logger.info("✅ 모든 FAISS 인덱스 로딩 완료!")
            self._log_index_info()
            
            return True
            
        except FaissLoadError as e:
            logger.error(f"❌ FAISS 로드 오류: {e}")
            self.is_loaded = False
            raise
        except FileNotFoundError as e:
            logger.error(f"❌ 파일 없음 오류: {e}")
            self.is_loaded = False
            raise FaissLoadError("", f"필요한 파일이 없습니다: {e}")
        except PermissionError as e:
            logger.error(f"❌ 권한 오류: {e}")
            self.is_loaded = False
            raise FaissLoadError("", f"파일 읽기 권한이 없습니다: {e}")
        except Exception as e:
            logger.error(f"❌ 예상치 못한 오류: {e}")
            logger.error(f"   오류 타입: {type(e).__name__}")
            logger.error(f"   오류 메시지: {str(e)}")
            import traceback
            logger.error(f"   스택 트레이스: {traceback.format_exc()}")
            self.is_loaded = False
            raise FaissLoadError("", f"FAISS 인덱스 로딩 실패: {e}")
    
    def _validate_required_files(self):
        """필수 파일들 존재 확인"""
        # 🔄 수정: 실제 파일 확장자 .index로 변경
        required_files = {
            "disease_key": "disease_key_index.index",
            "disease_full": "disease_full_index.index", 
            "disease_metadata": "disease_metadata.pkl"
        }
        
        logger.info("🔍 필수 파일 존재 확인 중...")
        logger.info(f"   📁 기본 디렉토리: {self.faiss_dir}")
        
        # 디렉토리 구조 확인
        logger.info("📂 디렉토리 구조:")
        try:
            for item in self.faiss_dir.iterdir():
                if item.is_dir():
                    logger.info(f"   📁 {item.name}/")
                    for subitem in item.iterdir():
                        logger.info(f"      📄 {subitem.name}")
                else:
                    logger.info(f"   📄 {item.name}")
        except Exception as e:
            logger.error(f"❌ 디렉토리 구조 확인 실패: {e}")
        
        missing_files = []
        existing_files = []
        
        for file_type, file_name in required_files.items():
            file_path = self.faiss_dir / file_name
            logger.info(f"🔍 파일 확인: {file_path}")
            
            if file_path.exists():
                file_size = file_path.stat().st_size
                logger.info(f"   ✅ 존재 ({file_size:,} bytes)")
                existing_files.append(file_name)
            else:
                logger.error(f"   ❌ 없음: {file_path}")
                missing_files.append(file_name)
        
        if missing_files:
            logger.error(f"❌ 누락된 파일들: {missing_files}")
            logger.info(f"✅ 존재하는 파일들: {existing_files}")
            raise FaissLoadError("", f"필수 FAISS 파일이 없습니다: {missing_files}")
        
        logger.info("✅ 필수 파일 확인 완료")
    
    def _load_disease_indexes(self):
        """질병 FAISS 인덱스 로드"""
        logger.info("🏥 질병 FAISS 인덱스 로딩 중...")
        
        try:
            # Disease Key 인덱스 - 확장자 .index로 수정
            key_path = self.faiss_dir / "disease_key_index.index"
            logger.info(f"🔑 Key 인덱스 로드 시도: {key_path}")
            
            if not key_path.exists():
                raise FileNotFoundError(f"Key 인덱스 파일이 없습니다: {key_path}")
            
            logger.info(f"📊 Key 인덱스 파일 크기: {key_path.stat().st_size:,} bytes")
            
            try:
                self.indexes["disease_key"] = faiss.read_index(str(key_path))
                logger.info(f"✅ 질병 Key 인덱스 로드: {self.indexes['disease_key'].ntotal}개 벡터")
            except Exception as e:
                logger.error(f"❌ Key 인덱스 로드 실패: {e}")
                logger.error(f"   파일 경로: {key_path}")
                logger.error(f"   파일 존재: {key_path.exists()}")
                raise
            
            # Disease Full 인덱스 - 확장자 .index로 수정
            full_path = self.faiss_dir / "disease_full_index.index"
            logger.info(f"🔍 Full 인덱스 로드 시도: {full_path}")
            
            if not full_path.exists():
                raise FileNotFoundError(f"Full 인덱스 파일이 없습니다: {full_path}")
            
            logger.info(f"📊 Full 인덱스 파일 크기: {full_path.stat().st_size:,} bytes")
            
            try:
                self.indexes["disease_full"] = faiss.read_index(str(full_path))
                logger.info(f"✅ 질병 Full 인덱스 로드: {self.indexes['disease_full'].ntotal}개 벡터")
            except Exception as e:
                logger.error(f"❌ Full 인덱스 로드 실패: {e}")
                logger.error(f"   파일 경로: {full_path}")
                logger.error(f"   파일 존재: {full_path.exists()}")
                raise
            
        except Exception as e:
            logger.error(f"❌ 질병 인덱스 로드 전체 실패: {e}")
            logger.error(f"   오류 타입: {type(e).__name__}")
            logger.error(f"   오류 메시지: {str(e)}")
            raise FaissLoadError("", f"질병 인덱스 로드 실패: {e}")
    
    def _load_rag_indexes(self):
        """RAG FAISS 인덱스 로드 (선택사항)"""
        logger.info("📚 RAG FAISS 인덱스 로딩 중...")
        
        try:
            # RAG Q&A 인덱스 - 확장자 .index로 수정
            qa_path = self.faiss_dir / "rag_qa_index.index"
            if qa_path.exists():
                try:
                    self.indexes["rag_qa"] = faiss.read_index(str(qa_path))
                    logger.info(f"✅ RAG Q&A 인덱스 로드: {self.indexes['rag_qa'].ntotal}개 벡터")
                except Exception as e:
                    logger.warning(f"⚠️ RAG Q&A 인덱스 로드 실패: {e}")
                    self.indexes["rag_qa"] = None
            else:
                logger.info("ℹ️ RAG Q&A 인덱스 파일 없음 (선택사항)")
                self.indexes["rag_qa"] = None
            
            # RAG Medical 인덱스 - 확장자 .index로 수정
            medical_path = self.faiss_dir / "rag_medical_index.index"
            if medical_path.exists():
                try:
                    self.indexes["rag_medical"] = faiss.read_index(str(medical_path))
                    logger.info(f"✅ RAG Medical 인덱스 로드: {self.indexes['rag_medical'].ntotal}개 벡터")
                except Exception as e:
                    logger.warning(f"⚠️ RAG Medical 인덱스 로드 실패: {e}")
                    self.indexes["rag_medical"] = None
            else:
                logger.info("ℹ️ RAG Medical 인덱스 파일 없음 (선택사항)")
                self.indexes["rag_medical"] = None
                
        except Exception as e:
            logger.warning(f"⚠️ RAG 인덱스 로드 실패 (선택사항): {e}")
            self.indexes["rag_qa"] = None
            self.indexes["rag_medical"] = None
    
    def _load_metadata(self):
        """메타데이터 로드"""
        logger.info("📋 메타데이터 로딩 중...")
        
        try:
            # 질병 메타데이터 (필수) - 서브폴더 없이 바로 파일
            disease_meta_path = self.faiss_dir / "disease_metadata.pkl"
            logger.info(f"📊 메타데이터 로드 시도: {disease_meta_path}")
            
            if not disease_meta_path.exists():
                raise FileNotFoundError(f"메타데이터 파일이 없습니다: {disease_meta_path}")
            
            logger.info(f"📊 메타데이터 파일 크기: {disease_meta_path.stat().st_size:,} bytes")
            
            try:
                with open(disease_meta_path, 'rb') as f:
                    self.metadata["disease"] = pickle.load(f)
                logger.info(f"✅ 질병 메타데이터 로드: {len(self.metadata['disease'])}개")
            except Exception as e:
                logger.error(f"❌ 메타데이터 파일 읽기 실패: {e}")
                logger.error(f"   파일 경로: {disease_meta_path}")
                logger.error(f"   파일 존재: {disease_meta_path.exists()}")
                raise
            
            # RAG Q&A 메타데이터 (선택사항)
            qa_meta_path = self.faiss_dir / "rag_qa_metadata.pkl"
            if qa_meta_path.exists():
                try:
                    with open(qa_meta_path, 'rb') as f:
                        self.metadata["rag_qa"] = pickle.load(f)
                    logger.info(f"✅ RAG Q&A 메타데이터 로드: {len(self.metadata['rag_qa'])}개")
                except Exception as e:
                    logger.warning(f"⚠️ RAG Q&A 메타데이터 로드 실패: {e}")
                    self.metadata["rag_qa"] = []
            else:
                logger.info("ℹ️ RAG Q&A 메타데이터 파일 없음 (선택사항)")
                self.metadata["rag_qa"] = []
            
            # RAG Medical 메타데이터 (선택사항)
            medical_meta_path = self.faiss_dir / "rag_medical_metadata.pkl"
            if medical_meta_path.exists():
                try:
                    with open(medical_meta_path, 'rb') as f:
                        self.metadata["rag_medical"] = pickle.load(f)
                    logger.info(f"✅ RAG Medical 메타데이터 로드: {len(self.metadata['rag_medical'])}개")
                except Exception as e:
                    logger.warning(f"⚠️ RAG Medical 메타데이터 로드 실패: {e}")
                    self.metadata["rag_medical"] = []
            else:
                logger.info("ℹ️ RAG Medical 메타데이터 파일 없음 (선택사항)")
                self.metadata["rag_medical"] = []
                
        except Exception as e:
            logger.error(f"❌ 메타데이터 로드 전체 실패: {e}")
            logger.error(f"   오류 타입: {type(e).__name__}")
            logger.error(f"   오류 메시지: {str(e)}")
            raise FaissLoadError("", f"메타데이터 로드 실패: {e}")
    
    def _validate_loaded_data(self):
        """로드된 데이터 유효성 검증"""
        logger.info("🔍 로드된 데이터 검증 중...")
        
        # 필수 인덱스 확인
        required_indexes = ["disease_key", "disease_full"]
        for index_name in required_indexes:
            if index_name not in self.indexes or self.indexes[index_name] is None:
                raise FaissLoadError("", f"필수 인덱스가 없습니다: {index_name}")
        
        # 인덱스 크기 확인
        disease_key_count = self.indexes["disease_key"].ntotal
        disease_full_count = self.indexes["disease_full"].ntotal
        disease_meta_count = len(self.metadata["disease"])
        
        if disease_key_count != disease_full_count:
            raise FaissLoadError("", f"질병 Key({disease_key_count})와 Full({disease_full_count}) 인덱스 크기가 일치하지 않습니다.")
        
        if disease_key_count != disease_meta_count:
            raise FaissLoadError("", f"질병 인덱스({disease_key_count})와 메타데이터({disease_meta_count}) 크기가 일치하지 않습니다.")
        
        logger.info("✅ 데이터 유효성 검증 완료")
    
    def _log_index_info(self):
        """인덱스 정보 로깅"""
        logger.info("📊 로드된 인덱스 정보:")
        
        for index_name, index in self.indexes.items():
            if index is not None:
                logger.info(f"   - {index_name}: {index.ntotal}개 벡터")
            else:
                logger.info(f"   - {index_name}: 없음")
        
        for meta_name, meta_data in self.metadata.items():
            if isinstance(meta_data, list):
                logger.info(f"   - {meta_name} 메타데이터: {len(meta_data)}개")
    
    def get_disease_indexes(self) -> Tuple[faiss.IndexFlatIP, faiss.IndexFlatIP]:
        """질병 인덱스 반환"""
        if not self.is_loaded:
            raise FaissLoadError("", "FAISS 인덱스가 로드되지 않았습니다.")
        
        return self.indexes["disease_key"], self.indexes["disease_full"]
    
    def get_disease_metadata(self) -> List[Dict]:
        """질병 메타데이터 반환"""
        if not self.is_loaded:
            raise FaissLoadError("", "FAISS 인덱스가 로드되지 않았습니다.")
        
        return self.metadata["disease"]
    
    def get_rag_indexes(self) -> Tuple[Optional[faiss.IndexFlatIP], Optional[faiss.IndexFlatIP]]:
        """RAG 인덱스 반환"""
        if not self.is_loaded:
            raise FaissLoadError("", "FAISS 인덱스가 로드되지 않았습니다.")
        
        return self.indexes.get("rag_qa"), self.indexes.get("rag_medical")
    
    def get_rag_metadata(self) -> Tuple[List[Dict], List[Dict]]:
        """RAG 메타데이터 반환"""
        if not self.is_loaded:
            raise FaissLoadError("", "FAISS 인덱스가 로드되지 않았습니다.")
        
        return self.metadata.get("rag_qa", []), self.metadata.get("rag_medical", [])
    
    def get_memory_usage(self) -> Dict[str, str]:
        """메모리 사용량 정보 반환"""
        if not self.is_loaded:
            return {"status": "not_loaded"}
        
        usage = {"status": "loaded"}
        
        for index_name, index in self.indexes.items():
            if index is not None:
                # 대략적인 메모리 사용량 계산 (바이트)
                vector_count = index.ntotal
                vector_dim = index.d
                memory_bytes = vector_count * vector_dim * 4  # float32 기준
                memory_mb = memory_bytes / (1024 * 1024)
                usage[f"{index_name}_memory_mb"] = f"{memory_mb:.1f}"
        
        return usage
    
    def get_service_status(self) -> Dict:
        """서비스 상태 반환"""
        status = {
            "is_loaded": self.is_loaded,
            "faiss_directory": str(self.faiss_dir),
            "directory_exists": self.faiss_dir.exists()
        }
        
        if self.is_loaded:
            status.update({
                "disease_key_vectors": self.indexes["disease_key"].ntotal if self.indexes.get("disease_key") else 0,
                "disease_full_vectors": self.indexes["disease_full"].ntotal if self.indexes.get("disease_full") else 0,
                "disease_metadata_count": len(self.metadata.get("disease", [])),
                "rag_qa_available": self.indexes.get("rag_qa") is not None,
                "rag_medical_available": self.indexes.get("rag_medical") is not None
            })
        
        return status


# 전역 로더 인스턴스 (싱글톤 패턴)  
_global_loader: Optional[DiseaseFAISSLoader] = None


def get_faiss_loader() -> DiseaseFAISSLoader:
    """FAISS 로더 싱글톤 인스턴스 반환"""
    global _global_loader
    
    if _global_loader is None:
        _global_loader = DiseaseFAISSLoader()
    
    return _global_loader


def initialize_faiss_loader() -> bool:
    """FAISS 로더 초기화"""
    try:
        loader = get_faiss_loader()
        return loader.load_all_indexes()
    except Exception as e:
        logger.error(f"❌ FAISS 로더 초기화 실패: {e}")
        raise