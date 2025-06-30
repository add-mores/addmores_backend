# AmedI: 당신의 증상에 딱 맞는 AI 의료 도우미

**AmedI**는 사용자 증상을 자연어로 입력하면, AI가 질병을 예측하고 관련 의약품과 위치 기반 병원을 추천해주는 **지능형 헬스케어 서비스**입니다.

> 본 프로젝트는 **LG Whynot SW 캠프 4기** ‘더해보다’ 팀이 개발하였습니다.
> 
---
##  팀 소개 

| **안재영** | **강현룡** | **조민규** | **백지원** |
|-----------|-----------|-----------|-----------|
| ![Project&nbsp;Leader](https://img.shields.io/badge/Project%20Leader-2962FF?style=plastic&logoColor=white) | ![Agile&nbsp;Coach](https://img.shields.io/badge/Agile%20Coach-43A047?style=plastic&logoColor=white) | ![Tech&nbsp;Lead](https://img.shields.io/badge/Tech%20Lead-FFB300?style=plastic&logoColor=white) | ![Config&nbsp;Mgr](https://img.shields.io/badge/Config%20Mgr-F57C00?style=plastic&logoColor=white) |
|![AI](https://img.shields.io/badge/AI-7E57C2?style=plastic&logo=openai&logoColor=white) ![BackEnd](https://img.shields.io/badge/BackEnd-3776AB?style=plastic&logo=python&logoColor=white)<br> 증상·질병<br>운영 및 총괄 | ![FrontEnd](https://img.shields.io/badge/FrontEnd-06B6D4?style=plastic&logo=react&logoColor=white) ![BackEnd](https://img.shields.io/badge/BackEnd-3776AB?style=plastic&logo=python&logoColor=white)<br> 병원<br> UI·협업 및 LLM 보조  | ![FrontEnd](https://img.shields.io/badge/FrontEnd-06B6D4?style=plastic&logo=react&logoColor=white)  ![Cloud](https://img.shields.io/badge/Cloud-FF9900?style=plastic&logo=amazonaws&logoColor=white) <br> AWS·DB 엔지니어| ![AI](https://img.shields.io/badge/AI-7E57C2?style=plastic&logo=openai&logoColor=white) ![BackEnd](https://img.shields.io/badge/BackEnd-3776AB?style=plastic&logo=python&logoColor=white)<br> 의약품 <br> GitOps 기획·최적화 |
| [![GitHub](https://img.shields.io/badge/GitHub-181717?style=plastic&logo=github&logoColor=white)](https://github.com/Jacob-53) | [![GitHub](https://img.shields.io/badge/GitHub-181717?style=plastic&logo=github&logoColor=white)](https://github.com/stundrg) | [![GitHub](https://img.shields.io/badge/GitHub-181717?style=plastic&logo=github&logoColor=white)](https://github.com/cho6019) | [![GitHub](https://img.shields.io/badge/GitHub-181717?style=plastic&logo=github&logoColor=white)](https://github.com/jiwon1118) |

---

##  개요

- 자연어 기반 **증상 입력**
- LLM 기반 **질병 예측**
- 효능 유사도 기반 **의약품 추천**
- 위치 + 진료과 기반 **병원 안내**
- 실시간 **Chatbot** 통합 UI

---

##  프로젝트 목표

-  정확한 증상-질병 매칭 시스템 개발
-  실시간 FastAPI 서버 및 Chatbot 구축
-  위치 기반 병원 추천 알고리즘 개발
-  Docker + AWS EC2 기반 배포 환경 구축

---

##  핵심 기능 요약

| 기능                  | 설명 |
|-----------------------|------|
| 질병 예측             | RAG + EXAONE 기반 질병 추론 및 추가 질문 |
| 의약품 추천           | 입력 텍스트 또는 약물명 기반 메타데이터 유사도 매칭 |
| 병원 추천             | 위치 + 진료과 기반 거리 필터  |
| Chatbot UI            | 질의응답 기반 챗형 인터페이스 구성 |
| 대시보드 통합         | 전체 예측 결과 및 상태 시각화 지원 |

---

##  사용 기술 스택

| 범주 | 기술 |
|----------|-------|
| **Backend & API** | ![Python](https://img.shields.io/badge/Python-3776AB?style=plastic&logo=python&logoColor=white) ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=plastic&logo=fastapi&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=plastic&logo=pandas&logoColor=white) ![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-B7312F?style=plastic&logo=sqlalchemy&logoColor=white) |
| **Frontend UI** | ![Next.js](https://img.shields.io/badge/Next.js-000000?style=plastic&logo=next.js&logoColor=white) ![React](https://img.shields.io/badge/React-61DAFB?style=plastic&logo=react&logoColor=000000) ![Tailwind CSS](https://img.shields.io/badge/Tailwind%20CSS-06B6D4?style=plastic&logo=tailwindcss&logoColor=white) |
| **AI & LLM** | ![EXAONE 3.5](https://img.shields.io/badge/EXAONE%203.5:7.8B-00A6E1?style=plastic) ![LangChain](https://img.shields.io/badge/LangChain-000000?style=plastic&logo=langchain&logoColor=white) ![HF Embeddings](https://img.shields.io/badge/HuggingFace%20Embeddings-FCC624?style=plastic&logo=huggingface&logoColor=000000) ![km-bert](https://img.shields.io/badge/km--bert-006400?style=plastic) |
| **Vector Store** | ![FAISS](https://img.shields.io/badge/FAISS-2284CC?style=plastic) |
| **Database** | ![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=plastic&logo=postgresql&logoColor=white) |
| **DevOps & Infra** | ![Docker](https://img.shields.io/badge/Docker-2496ED?style=plastic&logo=docker&logoColor=white) ![AWS EC2](https://img.shields.io/badge/AWS%20EC2-FF9900?style=plastic&logo=amazonaws&logoColor=white) |
| **Maps API** | ![Naver Map](https://img.shields.io/badge/Naver%20Map-03C75A?style=plastic&logo=naver&logoColor=white) |
| **Monitoring** | ![Grafana](https://img.shields.io/badge/Grafana-F46800?style=plastic&logo=grafana&logoColor=white) |


---

##  주요 LLM API 명세

| 엔드포인트 | 설명 |
|------------|------|
| `POST /llm/Amedi` | 자연어 증상 기반 질병 예측 및 의약품 추천 |

---

##  데이터 출처

- **질병 데이터**: 서울대병원, 서울아산병원 (크롤링)
- **의약품 데이터**: e약은요 API
- **병원 데이터**: 건강보험심사평가원 API

---

##  기대 효과

-  **맞춤형 의료 의사결정 지원**
-  **시간 절약형 병원·약 정보 제공**
-  **데이터 기반 건강 인사이트 생성**
-  **현실 적용 가능한 실전형 서비스**

---

##  시연 이미지 예시

- 병원 추천 지도 UI
- LLM 응답 챗봇 화면
- 대시보드 통합 결과

---

##  향후 계획 (Suggestions)

-  모바일 전용 UI 개발
-  사용자 피드백 기반 질의 응답 개선
-  응급실 실시간 혼잡도 연동 (공공데이터 API)
-  LLM 비교 (Exaone vs Hyper Clova X vs GPT-4 등)

---

##  로컬 실행 가이드
<a href="https://github.com/add-mores/frontend" target="_blank">
  <img src="https://img.shields.io/badge/GitHub-Frontend-black?logo=github&style=for-the-badge" alt="GitHub Frontend">
</a>

```bash
# 백엔드
cd backend
pdm install
uvicorn app.main:app --reload

# 프론트엔드
cd frontend
pnpm install
pnpm dev
```

환경변수 `.env` 예시:
```env
NEXT_PUBLIC_MAP_CLIENT_ID=your_naver_id
NEXT_PUBLIC_MAP_CLIENT_SECRET=your_naver_secret
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

---

##  라이선스

본 프로젝트는 교육 및 비영리 목적의 사용을 허용하며, 데이터 출처 표기와 AI 응답에 대한 면책 조항을 포함합니다.

---

>  AmedI는 **의료 전문가를 대체하지 않으며**, 정확한 진단 및 치료를 위해 병원 방문을 권장합니다.
