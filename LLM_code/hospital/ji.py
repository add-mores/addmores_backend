"""
RAG-의료 챗봇  (Geo-Filter + 생활태그 강화 · LangChain 0.2+)

● disease / medicine / hospital → 각기 인덱스 캐싱
● 네이버 지오코딩 + 하버사인 반경 5 km로 ‘근처 병원’ 우선 선별
● medicine 문서에 ‘감기/해열/기침’ 태그 삽입해 약품-매칭률 ↑
● HybridGeoRetriever(BaseRetriever) → RetrievalQA 통과
"""

# ───────── 0. 라이브러리 ─────────
from langchain_ollama                       import OllamaLLM
from langchain_core.documents               import Document
from langchain_core.retrievers              import BaseRetriever
from langchain_community.vectorstores       import FAISS
from langchain_community.embeddings         import HuggingFaceEmbeddings
from langchain.text_splitter                import RecursiveCharacterTextSplitter
from langchain.chains                       import RetrievalQA
from langchain.prompts                      import (
    ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate,
)
from pydantic                                import ConfigDict
import pandas as pd, os, csv, re, math, requests
from datetime                                import datetime

# ───────── 1. 설정 & 유틸 ─────────
# ‼️ 두 이름 모두 읽어 API 키를 찾습니다.
NAVER_MAP_ID = (
    os.getenv("NAVER_MAP_ID")
    or os.getenv("NEXT_PUBLIC_MAP_CLIENT_ID")
)
NAVER_MAP_SECRET = (
    os.getenv("NAVER_MAP_SECRET")
    or os.getenv("NEXT_PUBLIC_MAP_CLIENT_SECRET")
)

LOG = lambda m: print(f"🕒 {datetime.now().strftime('%H:%M:%S')} | {m}")

def pretty(txt: str):
    print(re.sub(r'\n{2,}', '\n\n', txt.replace("\\n", "\n")).strip())

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = map(math.radians, [lat2 - lat1, lon2 - lon1])
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2)
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def normalize_addr(addr: str):
    return re.sub(r'로(\d)', r'로 \1', addr).strip()

def geocode_naver(addr: str):
    if not (NAVER_MAP_ID and NAVER_MAP_SECRET):
        raise RuntimeError("❌ NAVER API 키가 설정되지 않았습니다.")
    url = "https://maps.apigw.ntruss.com/map-geocode/v2/geocode"
    headers = {
        "X-NCP-APIGW-API-KEY-ID": NAVER_MAP_ID,
        "X-NCP-APIGW-API-KEY":    NAVER_MAP_SECRET,
    }
    res = requests.get(url, headers=headers, params={"query": addr}, timeout=10)
    res.raise_for_status()
    items = res.json().get("addresses", [])
    if not items:
        raise RuntimeError("❌ 지오코딩 실패: 결과 없음")
    return float(items[0]["y"]), float(items[0]["x"])   # lat, lon

# ───────── 2. CSV → Documents & 인덱스 ─────────
def load_csv(path: str, label: str):
    df = pd.read_csv(path, quoting=csv.QUOTE_MINIMAL,
                     encoding="utf-8", on_bad_lines="skip")
    docs = []
    for _, row in df.iterrows():
        content = "\n".join(f"{c}: {row[c]}" for c in df.columns)

        # 약품 문서 생활태그
        if label == "medicine":
            lc = content.lower()
            tags = []
            if any(k in lc for k in ["감기", "콧물", "기침"]): tags.append("감기")
            if any(k in lc for k in ["해열", "발열"]):         tags.append("해열")
            if tags:
                content = f"tags: {' '.join(tags)}\n" + content

        docs.append(
            Document(
                page_content=content,
                metadata={
                    "source_file": label,
                    **({"lat": row["lat"], "lon": row["lon"]} if "lat" in df else {})
                },
            )
        )
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs), df

embeds = HuggingFaceEmbeddings(model_name="madatnlp/km-bert")

def build_index(idx_dir, csv_path, label):
    if os.path.isdir(idx_dir):
        return FAISS.load_local(
            idx_dir, embeds, allow_dangerous_deserialization=True
        )
    docs, _ = load_csv(csv_path, label)
    vs = FAISS.from_documents(docs, embeds)
    vs.save_local(idx_dir)
    return vs

disease_vs  = build_index("idx_disease",  "Ragfile/dis.csv",  "disease")
medicine_vs = build_index("idx_medicine", "Ragfile/medi.csv", "medicine")
hospital_docs, hospital_df = load_csv("Ragfile/hospi.csv", "hospital")
hospital_vs = build_index("idx_hospital", "Ragfile/hospi.csv", "hospital")

# ───────── 3. HybridGeoRetriever ─────────
class HybridGeoRetriever(BaseRetriever):
    k: int = 7
    model_config = ConfigDict(extra="allow")

    def __init__(self, k_each=7):
        super().__init__(k=k_each)

    # 좌표 추출
    def _get_coord(self, query: str):
        m = re.search(r'([0-9]+\.[0-9]+)[ ,]+([0-9]+\.[0-9]+)', query)
        if m:
            return float(m.group(1)), float(m.group(2))
        if "로" in query or "길" in query:
            try:
                return geocode_naver(normalize_addr(query))
            except Exception as e:
                LOG(str(e))
        return None, None

    # 동기 검색
    def _get_relevant_documents(self, query, *, run_manager=None, **kw):
        lat, lon = self._get_coord(query)
        docs = []

        # 병원 후보 (좌표 있으면 거리 기반, 없으면 텍스트)
        k_hos = self.k if lat else 2
        if lat:
            sub = hospital_df.copy()
            sub["dist"] = sub.apply(
                lambda r: haversine(lat, lon, r["lat"], r["lon"]), axis=1
            )
            near = sub.nsmallest(k_hos, "dist")
            for _, r in near.iterrows():
                docs.append(
                    Document(
                        page_content="\n".join(f"{c}: {r[c]}" for c in hospital_df.columns),
                        metadata={
                            "source_file": "hospital",
                            "lat": r["lat"],
                            "lon": r["lon"],
                            "dist_km": round(r["dist"], 2),
                        },
                    )
                )
        else:
            docs += hospital_vs.similarity_search(query, k_hos)

        # 질병·약 문서
        docs += disease_vs.similarity_search(query, self.k)
        docs += medicine_vs.similarity_search(query, self.k)
        return docs

    async def _aget_relevant_documents(self, query, *, run_manager=None, **kw):
        return self._get_relevant_documents(query)

retriever = HybridGeoRetriever(k_each=7)

# ───────── 4. RAG 함수 ─────────
def rag_answer(question: str) -> str:
    llm = OllamaLLM(model="exaone3.5:7.8b", temperature=0.1, num_predict=1024)

    sys_msg = (
        "너는 의료 정보 챗봇이야. source_file=disease/medicine/hospital.\n"
        "증상·질병 질문이면 disease/medicine을 우선 요약하고, "
        "병원 질문이면 이미 거리 필터링된 hospital 문서만 이용해 추천해라.\n"
        "병원 출력 시 hos_type은 '병원'으로, 진료과목도 보여줘라.\n"
        "정보가 없으면 '잘 모르겠습니다'라고 답해."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(sys_msg),
            HumanMessagePromptTemplate.from_template(
                "질문: {question}\n\n참고 문서:\n\n{context}"
            ),
        ]
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt},
    )

    res = qa.invoke({"query": question})
    return res["result"] if isinstance(res, dict) else str(res)

# ───────── 5. CLI ─────────
if __name__ == "__main__":
    while True:
        q = input("\n💬 무엇을 도와드릴까요? (exit/종료)\n> ").strip()
        if q.lower() in {"exit", "종료", "quit"}:
            print("👋 감사합니다!"); break
        try:
            pretty(rag_answer(q))
        except Exception as e:
            LOG(f"❌ 오류: {e}")
