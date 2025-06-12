# hospi_llm_exa_api.py
from fastapi import FastAPI, APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Any
import os, json, requests, urllib.parse, re
import numpy as np
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

# ───── 환경 설정 ─────────────────────────────
load_dotenv()
NAVER_MAP_ID     = os.getenv("NEXT_PUBLIC_MAP_CLIENT_ID")
NAVER_MAP_SECRET = os.getenv("NEXT_PUBLIC_MAP_CLIENT_SECRET")

BASE_DIR         = os.path.dirname(__file__)
INDEX_PATH       = os.path.join(BASE_DIR, "hospi_faiss_index")
EMBEDDING_MODEL  = "madatnlp/km-bert"
llm              = OllamaLLM(model="exaone3.5:7.8b", temperature=0.3)
embedding        = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectordb         = FAISS.load_local(INDEX_PATH, embedding, allow_dangerous_deserialization=True)

# ───── FastAPI 설정 ──────────────────────────
router = APIRouter()
app = FastAPI()
app.include_router(router)

# ───── 공통 함수 ─────────────────────────────
def haversine(lat1, lon1, lat2s, lon2s):
    R = 6371
    dlat = np.radians(lat2s - lat1)
    dlon = np.radians(lon2s - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2s)) * np.sin(dlon/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

def geocode_address(query: str) -> Dict[str, Any]:
    url = "https://maps.apigw.ntruss.com/map-geocode/v2/geocode"
    headers = {
        "X-NCP-APIGW-API-KEY-ID": NAVER_MAP_ID,
        "X-NCP-APIGW-API-KEY": NAVER_MAP_SECRET
    }
    r = requests.get(url, headers=headers, params={"query": query}, timeout=10)
    if r.status_code != 200: raise HTTPException(502, r.text)
    arr = r.json().get("addresses", [])
    if not arr: raise HTTPException(404, "주소를 찾을 수 없습니다.")
    a = arr[0]
    return {"lat": float(a["y"]), "lon": float(a["x"]),
            "address_name": a.get("roadAddress") or a.get("jibunAddress")}

def exaone_chat(msgs: List[Dict[str, Any]]) -> str:
    return llm.invoke(msgs)

def make_place_url(name: str, lat: float, lon: float) -> str:
    return f"nmap://place?lat={lat}&lng={lon}&name={urllib.parse.quote(name, safe='')}"

def generate_llm_reason(symptom: str, hospitals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prompt = {
        "symptom": symptom,
        "candidates": hospitals
    }
    msgs = [
        {"role": "system", "content":
         "다음은 증상과 병원 정보입니다. 각 병원이 왜 추천되었는지 간단하고 명확한 추천 사유를 JSON 배열로 생성하세요. "
         "각 항목은 hos_nm과 reason 필드를 포함해야 합니다. 마크다운과 코드블록을 쓰지 마세요."},
        {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}
    ]
    try:
        raw = exaone_chat(msgs).strip()
        print("🧠 LLM 응답 원문:\n", raw)

        # 마크다운 제거
        clean = re.sub(r"```(?:json)?\s*(.*?)\s*```", r"\1", raw, flags=re.S).strip()
        if not clean.startswith("["):
            print("❌ JSON 배열 형식 아님, 무시됨")
            raise ValueError("응답이 배열 형식 아님")

        parsed = json.loads(clean)

        # 병원 이름 매칭해서 필드 보완
        hospi_dict = {c["hos_nm"]: c for c in hospitals}
        for item in parsed:
            h = hospi_dict.get(item.get("hos_nm"))
            if h:
                item.setdefault("add", h["add"])
                item.setdefault("deps", h["deps"])
                item.setdefault("distance", h["distance"])
                item.setdefault("opening_hours", "운영시간 정보 없음")
        return parsed
    except Exception as e:
        print("❌ LLM 요약 실패:", e)
        return [{"hos_nm": h["hos_nm"], "reason": "추천 사유 없음"} for h in hospitals]

# ───── 요청 모델 ─────────────────────────────
class FilterRequest(BaseModel):
    address: str
    symptom: str
    radius: float = 1.0
    
@router.get("/list_departments")
def list_departments():
    """FAISS 인덱스에서 모든 진료과 목록을 추출하여 반환"""
    deps_set = set()
    for doc in vectordb.docstore._dict.values():
        for d in str(doc.metadata.get("treatment", "")).split(","):
            d = d.strip()
            if d:
                deps_set.add(d)
    return sorted(deps_set)

# ───── POST /llm/hospital ────────────────────
@router.post("/llm/hospital")
def recommend(req: FilterRequest):
    # 1) 주소 → 위경도
    geo = geocode_address(req.address)
    lat, lon = geo["lat"], geo["lon"]

    # 2) 진료과 예측
    messages = [
        {"role": "system", "content": "증상에서 예상 진료과를 JSON 배열로만 출력하세요. 마크다운 없이."},
        {"role": "user", "content": req.symptom}
    ]
    try:
        resp = exaone_chat(messages).strip()
        deps = json.loads(resp[resp.find("["):resp.rfind("]")+1])
    except Exception:
        deps = []

    if not deps:
        raise HTTPException(400, "진료과 예측 실패")

    # 3) FAISS 후보 병원 추출
    docs = vectordb.similarity_search(req.symptom, k=vectordb.index.ntotal)
    cands = []
    for d in docs:
        m = d.metadata
        lat2, lon2 = float(m.get("lat", 0)), float(m.get("lon", 0))
        dist = haversine(lat, lon, lat2, lon2)
        dep_list = [x.strip() for x in m.get("treatment", "").split(",") if x.strip()]
        if any(dep in dep_list for dep in deps):
            cands.append({
                "hos_nm": m.get("hospital_name", ""),
                "add": m.get("address", ""),
                "deps": dep_list,
                "distance": round(dist, 2),
                "lat": lat2,
                "lon": lon2,
                "map_url": make_place_url(m.get("hospital_name", ""), lat2, lon2)
            })

    cands = sorted(cands, key=lambda x: x["distance"])[:10]
    if not cands:
        return {"predicted_deps": deps, "recommendations": []}

    # 4) LLM 요약 추천 (5개)
    llm_input = cands[:5]
    summary = generate_llm_reason(req.symptom, llm_input)

    return {
        "predicted_deps": deps,
        "recommendations": cands,
        "llm_summary": summary
    }



# ───── FastAPI 라우터 등록 ────────────────────
app.include_router(router)