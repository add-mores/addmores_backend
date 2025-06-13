from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os, json, requests, urllib.parse, re, ast
import numpy as np
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
import logging

load_dotenv()
# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

NAVER_MAP_ID = os.getenv("NEXT_PUBLIC_MAP_CLIENT_ID")
NAVER_MAP_SECRET = os.getenv("NEXT_PUBLIC_MAP_CLIENT_SECRET")

BASE_DIR = os.path.dirname(__file__)
INDEX_PATH = os.path.join(BASE_DIR, "hospi_faiss_index")
EMBEDDING_MODEL = "madatnlp/km-bert"
llm = OllamaLLM(model="exaone3.5:7.8b", temperature=0.3)
embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectordb = FAISS.load_local(INDEX_PATH, embedding, allow_dangerous_deserialization=True)

router = APIRouter()
app = FastAPI()
app.include_router(router)

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
    return {"lat": float(a["y"]), "lon": float(a["x"]), "address_name": a.get("roadAddress") or a.get("jibunAddress")}

def exaone_chat(msgs: List[Dict[str, Any]]) -> str:
    return llm.invoke(msgs)

def make_web_map_url(name: str) -> str:
    return f"https://map.naver.com/v5/search/{urllib.parse.quote(name)}"

def generate_llm_reason(query: str, hospitals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prompt = {"query": query, "candidates": hospitals}
    msgs = [
        {"role": "system", "content":
         "다음은 증상과 병원 정보입니다. 각 병원이 왜 추천되었는지 간단하고 명확한 추천 사유를 JSON 배열로 생성하세요. "
         "각 항목은 hos_nm과 reason 필드를 포함해야 합니다. 마크다운과 코드블록을 쓰지 마세요."},
        {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}
    ]
    try:
        raw = exaone_chat(msgs).strip()
        # print("🧠 LLM 응답 원문:\n", raw)
        logger.info(f"🧠 LLM 응답 원문:\n{raw}")
        clean = re.sub(r"```(?:json)?\s*(.*?)\s*```", r"\1", raw, flags=re.S).strip()
        if not clean.startswith("["): raise ValueError("응답이 JSON 배열 아님")
        parsed = json.loads(clean)
        hospi_dict = {c["hos_nm"]: c for c in hospitals}
        for item in parsed:
            h = hospi_dict.get(item.get("hos_nm"))
            if h:
                item.setdefault("add", h["add"])
                item.setdefault("deps", h["deps"])
                item.setdefault("distance", h["distance"])
                item.setdefault("opening_hours", "운영시간 정보 없음")
                item.setdefault("map_url", make_web_map_url(item["hos_nm"]))
        return parsed
    except Exception as e:
        # print("❌ LLM 요약 실패:", e)/
        logger.error(f"❌ LLM 요약 실패: {e}")
        return [
            {
                "hos_nm": h["hos_nm"],
                "reason": "추천 사유 없음",
                "add": h.get("add", ""),
                "deps": h.get("deps", []),
                "distance": h.get("distance", 0),
                "opening_hours": h.get("opening_hours", "운영시간 정보 없음"),
                "map_url": make_web_map_url(h["hos_nm"])
            }
            for h in hospitals
        ]

class FilterRequest(BaseModel):
    # 프론트에서 주소(address) 또는 GPS(lat, lon) 중 하나만 보내도 되도록
    address: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    query: str
    radius: float = 1.0

@router.post("/llm/hospital")
def recommend(req: FilterRequest):
    # 수정: lat/​lon이 넘어오면 그대로 쓰고,
    #     address만 넘어오면 geocode_address() 호출
    if req.lat is not None and req.lon is not None:
        lat, lon = req.lat, req.lon
    elif req.address:
        geo = geocode_address(req.address)
        lat, lon = geo["lat"], geo["lon"]
    else:
        raise HTTPException(400, "address 또는 lat, lon 중 하나는 반드시 필요합니다.")

    messages = [
        {"role": "system", "content": "증상에서 예상 진료과를 JSON 배열로만 출력하세요. 각 항목은 {'department': ..., 'score': ...} 형식. 마크다운 없이."},
        {"role": "user", "content": req.query}
    ]
    try:
        resp = exaone_chat(messages).strip()
        # print("🧠 예측 응답:", resp)
        logger.info(f"🧠 예측 응답: {resp}")
        if resp.lstrip().startswith("[{") and "'" in resp:
            preds = ast.literal_eval(resp)
        else:
            preds = json.loads(resp[resp.find("["):resp.rfind("]")+1])
    except Exception as e:
        # print("❌ 진료과 예측 실패:", e)
        logger.error(f"❌ LLM 요약 실패: {e}")
        
        preds = []

    if not preds:
        return {
            "predicted_deps": [],
            "llm_summary": [],
            "message": "입력하신 증상으로는 추천할 수 있는 병원이 없습니다. 다시 시도해 주세요."
        }

    deps = [p["department"].replace(" ", "") for p in preds if "department" in p]
    docs = vectordb.similarity_search(" ".join(deps), k=vectordb.index.ntotal)

    cands = []
    for d in docs:
        m = d.metadata
        lat2, lon2 = float(m.get("lat", 0)), float(m.get("lon", 0))
        dist = haversine(lat, lon, lat2, lon2)
        dep_list = [x.replace(" ", "") for x in m.get("treatment", "").split(",") if x.strip()]
        if any(dep in dep_list for dep in deps):
            cands.append({
                "hos_nm": m.get("hospital_name", ""),
                "add": m.get("address", ""),
                "deps": dep_list,
                "distance": round(dist, 2),
                "lat": lat2,
                "lon": lon2
            })

    cands = sorted(cands, key=lambda x: x["distance"])[:10]

    if not cands:
        all_docs = vectordb.similarity_search("병원", k=vectordb.index.ntotal)
        for d in all_docs:
            m = d.metadata
            lat2, lon2 = float(m.get("lat", 0)), float(m.get("lon", 0))
            dist = haversine(lat, lon, lat2, lon2)
            dep_list = [x.replace(" ", "") for x in m.get("treatment", "").split(",") if x.strip()]
            if any(dep in dep_list for dep in deps):
                cands.append({
                    "hos_nm": m.get("hospital_name", ""),
                    "add": m.get("address", ""),
                    "deps": dep_list,
                    "distance": round(dist, 2),
                    "lat": lat2,
                    "lon": lon2
                })

    summary = generate_llm_reason(req.query, cands)[:3]
    # print(f"✅ 추천 병원 {len(summary)}개 생성 완료")
    logger.info(f"✅ 추천 병원 {len(summary)}개 생성 완료")

    return {
        "predicted_deps": preds,
        "llm_summary": summary
    }

app.include_router(router)
# FastAPI 애플리케이션 설정
