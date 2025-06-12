# ============================================================================= 
# 1. backend/app/llm/__init__.py
# =============================================================================
"""
LLM 통합 의료 챗봇 API 모듈

🎯 목적: CLI 기반 의료 챗봇을 FastAPI 서비스로 변환
📋 구조:
- main_llm.py: FastAPI 메인 앱
- api/: API 엔드포인트들
- services/: 비즈니스 로직 서비스들
"""

__version__ = "6.0.0"
__author__ = "Medical Chatbot Team"
__description__ = "LLM Integrated Medical Chatbot API"


__all__ = ["llm_app"]