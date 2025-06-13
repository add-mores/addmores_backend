"""
질병 임베딩 서비스
위치: ~/backend/app/services/disease_embedding_service.py

🎯 목적: KM-BERT 모델을 이용한 텍스트 임베딩 생성
📋 기능:
   - KM-BERT 모델 로드 및 관리
   - 텍스트를 벡터로 변환
   - 배치 처리 지원
   - GPU/MPS 지원

⚙️ 의존성: torch, transformers, numpy
"""

import torch
import numpy as np
import logging
from typing import List, Optional, Union
from transformers import AutoTokenizer, AutoModel

from ..utils.disease_constants import EMBEDDING_MODEL_NAME, EMBEDDING_MAX_LENGTH
from ..utils.disease_exceptions import EmbeddingModelLoadError, EmbeddingGenerationError

logger = logging.getLogger(__name__)


class DiseaseEmbeddingService:
    """질병 임베딩 서비스 클래스"""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.device = self._get_best_device()
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModel] = None
        self.is_loaded = False
        
        logger.info(f"🧠 임베딩 서비스 초기화: {model_name}")
        logger.info(f"📱 사용 디바이스: {self.device}")
    
    def _get_best_device(self) -> str:
        """최적의 디바이스 선택"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def load_model(self) -> bool:
        """KM-BERT 모델 로드"""
        if self.is_loaded:
            logger.info("✅ 임베딩 모델이 이미 로드되어 있습니다.")
            return True
        
        try:
            logger.info("🔄 KM-BERT 모델 로딩 중...")
            
            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            logger.info("✅ 토크나이저 로드 완료")
            
            # 모델 로드
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()  # 추론 모드
            logger.info("✅ KM-BERT 모델 로드 완료")
            
            # 모델 테스트
            self._test_model()
            
            self.is_loaded = True
            logger.info("✅ 임베딩 서비스 준비 완료!")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 임베딩 모델 로드 실패: {e}")
            self.is_loaded = False
            raise EmbeddingModelLoadError(self.model_name, str(e))
    
    def _test_model(self):
        """모델 정상 작동 테스트"""
        try:
            test_texts = ["안녕하세요", "테스트"]
            test_embeddings = self._encode_texts(test_texts)
            
            if test_embeddings.shape[0] != 2:
                raise EmbeddingModelLoadError(self.model_name, "모델 테스트 실패: 잘못된 출력 크기")
            
            logger.info(f"✅ 모델 테스트 완료: 임베딩 차원 {test_embeddings.shape[1]}")
            
        except Exception as e:
            raise EmbeddingModelLoadError(self.model_name, f"모델 테스트 실패: {e}")
    
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """텍스트를 벡터로 인코딩"""
        if not self.is_loaded:
            raise EmbeddingModelLoadError(self.model_name, "임베딩 모델이 로드되지 않았습니다. load_model()을 먼저 호출하세요.")
        
        # 단일 텍스트를 리스트로 변환
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            raise EmbeddingGenerationError("", "빈 텍스트 리스트입니다.")
        
        try:
            # 텍스트 전처리
            processed_texts = self._preprocess_texts(texts)
            
            # 임베딩 생성
            embeddings = self._encode_texts(processed_texts)
            
            logger.debug(f"✅ 임베딩 생성 완료: {len(texts)}개 텍스트 -> {embeddings.shape}")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ 임베딩 생성 실패: {e}")
            raise EmbeddingGenerationError(str(texts), f"임베딩 생성 실패: {e}")
    
    def _preprocess_texts(self, texts: List[str]) -> List[str]:
        """텍스트 전처리"""
        processed = []
        
        for text in texts:
            if not isinstance(text, str):
                text = str(text)
            
            # 빈 텍스트 처리
            text = text.strip()
            if not text:
                text = "[빈 텍스트]"
            
            # 길이 제한
            if len(text) > EMBEDDING_MAX_LENGTH * 2:  # 토큰이 아닌 문자 기준으로 대략적 제한
                text = text[:EMBEDDING_MAX_LENGTH * 2]
                logger.warning(f"⚠️ 텍스트가 잘렸습니다: {len(text)}자")
            
            processed.append(text)
        
        return processed
    
    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """실제 임베딩 생성 로직"""
        try:
            # 토크나이징
            encodings = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=EMBEDDING_MAX_LENGTH,
                return_tensors="pt"
            ).to(self.device)
            
            # 모델 추론
            with torch.no_grad():
                outputs = self.model(**encodings)
                last_hidden = outputs.last_hidden_state
                attention_mask = encodings.attention_mask.unsqueeze(-1)
                
                # 마스킹된 평균 풀링
                masked_hidden = last_hidden * attention_mask
                sum_hidden = masked_hidden.sum(dim=1)
                lengths = attention_mask.sum(dim=1)
                sentence_embeddings = sum_hidden / lengths.clamp(min=1e-9)
            
            # CPU로 이동 및 numpy 변환
            embeddings = sentence_embeddings.cpu().numpy()
            
            return embeddings
            
        except Exception as e:
            raise EmbeddingGenerationError("", f"임베딩 생성 중 오류: {e}")
    
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """배치 처리로 대량 텍스트 인코딩"""
        if not texts:
            raise EmbeddingGenerationError("", "빈 텍스트 리스트입니다.")
        
        if len(texts) <= batch_size:
            return self.encode(texts)
        
        logger.info(f"🔄 배치 처리 시작: {len(texts)}개 텍스트, 배치 크기 {batch_size}")
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            logger.debug(f"📦 배치 {i//batch_size + 1}: {len(batch_texts)}개 텍스트 처리 중...")
            
            batch_embeddings = self.encode(batch_texts)
            all_embeddings.append(batch_embeddings)
        
        # 모든 배치 결합
        final_embeddings = np.vstack(all_embeddings)
        logger.info(f"✅ 배치 처리 완료: {final_embeddings.shape}")
        
        return final_embeddings
    
    def get_embedding_dimension(self) -> int:
        """임베딩 차원 반환"""
        if not self.is_loaded:
            raise EmbeddingModelLoadError(self.model_name, "모델이 로드되지 않았습니다.")
        
        # 테스트 임베딩으로 차원 확인
        test_embedding = self.encode(["테스트"])
        return test_embedding.shape[1]
    
    def similarity(self, text1: str, text2: str) -> float:
        """두 텍스트 간 코사인 유사도 계산"""
        embeddings = self.encode([text1, text2])
        
        # 코사인 유사도 계산
        embedding1 = embeddings[0]
        embedding2 = embeddings[1]
        
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
    
    def get_model_info(self) -> dict:
        """모델 정보 반환"""
        info = {
            "model_name": self.model_name,
            "device": self.device,
            "is_loaded": self.is_loaded,
            "max_length": EMBEDDING_MAX_LENGTH
        }
        
        if self.is_loaded:
            info["embedding_dimension"] = self.get_embedding_dimension()
        
        return info
    
    def unload_model(self):
        """모델 언로드 (메모리 해제)"""
        if self.is_loaded:
            self.model = None
            self.tokenizer = None
            
            # GPU 메모리 정리
            if self.device in ["cuda", "mps"]:
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                # MPS는 별도 캐시 정리 함수가 없음
            
            self.is_loaded = False
            logger.info("✅ 임베딩 모델 언로드 완료")


# 전역 임베딩 서비스 인스턴스 (싱글톤 패턴)
_global_embedding_service: Optional[DiseaseEmbeddingService] = None


def get_embedding_service() -> DiseaseEmbeddingService:
    """임베딩 서비스 싱글톤 인스턴스 반환"""
    global _global_embedding_service
    
    if _global_embedding_service is None:
        _global_embedding_service = DiseaseEmbeddingService()
    
    return _global_embedding_service


def initialize_embedding_service() -> bool:
    """임베딩 서비스 초기화"""
    try:
        service = get_embedding_service()
        return service.load_model()
    except Exception as e:
        logger.error(f"❌ 임베딩 서비스 초기화 실패: {e}")
        raise