# 🧠 AmedI: 당신의 증상에 딱 맞는 AI 의료 도우미

**AmedI**는 사용자 증상을 자연어로 입력하면, AI가 질병을 예측하고 관련 의약품과 위치 기반 병원을 추천해주는 **지능형 헬스케어 서비스**입니다.

> 본 프로젝트는 **LG Whynot SW 캠프 4기** ‘더해보다’ 팀이 개발하였습니다.
> 
---
## 👥 팀 소개

| 이름     | 역할 |
|----------|------|
| **안재영** | 프로젝트 리더 – 백엔드 (증상·질병) 전체 기획/운영 |
| **강현룡** | 애자일 코치 – 프론트·백엔드 (병원) UI + 협업 관리 |
| **조민규** | 테크 리더 – 프론트 및 AWS·DB 엔지니어 |
| **백지원** | 형상 관리자 – 백엔드 (의약품) Git/GitHub 및 워크플로우 최적화 |



---

## 📌 개요

- 자연어 기반 **증상 입력**
- LLM 기반 **질병 예측**
- 효능 유사도 기반 **의약품 추천**
- 위치 + 진료과 기반 **병원 안내**
- 실시간 **Chatbot** 통합 UI

---

## 🏁 프로젝트 목표

- ✅ 정확한 증상-질병 매칭 시스템 개발
- ✅ 실시간 FastAPI 서버 및 Chatbot 구축
- ✅ 위치 기반 병원 추천 알고리즘 개발
- ✅ Docker + AWS EC2 기반 배포 환경 구축

---

## 🚀 핵심 기능 요약

| 기능                  | 설명 |
|-----------------------|------|
| 질병 예측             | RAG + EXAONE 기반 질병 추론 및 추가 질문 |
| 의약품 추천           | 입력 텍스트 또는 약물명 기반 메타데이터 유사도 매칭 |
| 병원 추천             | 위치 + 진료과 기반 거리 필터 + LLM 사유 생성 |
| Chatbot UI            | 질의응답 기반 챗형 인터페이스 구성 |
| 대시보드 통합         | 전체 예측 결과 및 상태 시각화 지원 |

---

## 🧬 사용 기술 스택

| 범주         | 기술 |
|--------------|------|
| 백엔드       | `Python`, `FastAPI`, `Pandas`, `SQLAlchemy` |
| 프론트엔드   | `Next.js`, `React`, `Tailwind CSS` |
| LLM          | `EXAONE 3.5:7.8B`,  `Langchain` |
| 벡터 검색    | `FAISS`, `HuggingFaceEmbeddings`, `madatnlp/km-bert` |
| 배포         | `Docker`, `AWS EC2`, `.env` 설정 |
| 지도 API     | `Naver Map` |
| 자동화/모니터링 | `Grafana` |

---

## 🧪 주요 LLM API 명세

| 엔드포인트 | 설명 |
|------------|------|
| `POST /llm/hospital` | 질병 기반 병원 추천 (LLM) |
| `POST /llm/medicine` | 의약품 설명 및 추천 |
| `POST /llm/disease` | 증상 기반 질병 예측 |

---

## 🔗 데이터 출처

- **질병 데이터**: 서울대병원, 서울아산병원 (크롤링)
- **의약품 데이터**: e약은요 API
- **병원 데이터**: 건강보험심사평가원 API

---

## 🎯 기대 효과

- 🧑‍⚕️ **맞춤형 의료 의사결정 지원**
- 🕒 **시간 절약형 병원·약 정보 제공**
- 📊 **데이터 기반 건강 인사이트 생성**
- 🌍 **현실 적용 가능한 실전형 서비스**

---

## 📸 시연 이미지 예시

- 병원 추천 지도 UI
- LLM 응답 챗봇 화면
- 대시보드 통합 결과

---

## 🧪 향후 계획 (Suggestions)

- 📱 모바일 전용 UI 개발
- 🔁 사용자 피드백 기반 질의 응답 개선
- 🏥 응급실 실시간 혼잡도 연동 (공공데이터 API)
- 🔬 LLM 비교 (Exaone vs Clova vs GPT 등)

---

## 🛠️ 로컬 실행 가이드

```bash
# 백엔드
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload

# 프론트엔드
cd frontend
npm install
npm run dev
```

환경변수 `.env` 예시:
```env
NEXT_PUBLIC_MAP_CLIENT_ID=your_naver_id
NEXT_PUBLIC_MAP_CLIENT_SECRET=your_naver_secret
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

---

## 📄 라이선스

본 프로젝트는 교육 및 비영리 목적의 사용을 허용하며, 데이터 출처 표기와 AI 응답에 대한 면책 조항을 포함합니다.

---

> 👨‍⚕️ AmedI는 **의료 전문가를 대체하지 않으며**, 정확한 진단 및 치료를 위해 병원 방문을 권장합니다.
