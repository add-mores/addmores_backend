# hospi_llm_api.py
from fastapi import FastAPI, APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from sqlalchemy import create_engine
import pandas as pd, numpy as np, os, json, requests, urllib.parse, re
from dotenv import load_dotenv


load_dotenv()
DATABASE_URL      = os.getenv("DATABASE_URL")
NAVER_MAP_ID      = os.getenv("NEXT_PUBLIC_MAP_CLIENT_ID")
NAVER_MAP_SECRET  = os.getenv("NEXT_PUBLIC_MAP_CLIENT_SECRET")
CLOVA_API_KEY     = os.getenv("CLOVA_API_KEY")
engine = create_engine(DATABASE_URL)

router = APIRouter()

# ─── 공통 함수 ─────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2s, lon2s):
    R = 6371
    dlat = np.radians(lat2s-lat1)
    dlon = np.radians(lon2s-lon1)
    a = (np.sin(dlat/2)**2 +
         np.cos(np.radians(lat1))*np.cos(np.radians(lat2s))*np.sin(dlon/2)**2)
    return R*2*np.arctan2(np.sqrt(a), np.sqrt(1-a))

def make_place_url(name:str, lat:float, lon:float) -> str:
    return f"nmap://place?lat={lat}&lng={lon}&name={urllib.parse.quote(name,safe='')}"

def clova_chat(msgs: List[Dict[str, Any]], model="HCX-005") -> str:
    url = "https://clovastudio.stream.ntruss.com/v1/openai/chat/completions"
    hd  = {"Authorization": f"Bearer {CLOVA_API_KEY}",
           "Content-Type": "application/json"}
    res = requests.post(url, headers=hd,
                        json={"model": model, "messages": msgs,
                              "max_tokens": 1024}, timeout=30)
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"]

def get_valid_deps() -> set[str]:
    df = pd.read_sql("SELECT deps FROM testhosp", engine)
    return {d.strip() for s in df["deps"].dropna() for d in s.split(",")}

def extract_deps(text:str) -> List[str]:
    vd = get_valid_deps()
    return [d for d in vd if d in text]

# ─── 요청 모델 ─────────────────────────────────────────────────────
class FilterRequest(BaseModel):
    lat: float
    lon: float
    radius: float = 1.0
    deps:    Optional[List[str]] = None
    symptom: Optional[str]       = None
    raw_text: Optional[str]      = None   # 자연어 진료과 추출용

# ─── 메인 엔드포인트 ────────────────────────────────────────────────
@router.post("/llm/hospital")
def recommend(req: FilterRequest):
    # 1) 진료과 확보
    if not req.deps and req.raw_text:
        req.deps = extract_deps(req.raw_text)

    if not req.deps and req.symptom:
        msgs = [
            {"role":"system",
             "content":"증상에서 상위 진료과 3개를 쉼표로 출력"},
            {"role":"user","content":req.symptom}
        ]
        try:
            req.deps = [d.strip() for d in clova_chat(msgs).split(",") if d.strip()]
        except Exception:
            req.deps = []

    if not req.deps:
        raise HTTPException(400, "진료과 정보가 없습니다.")

    # 2) DB 필터
    df = pd.read_sql("SELECT hos_nm, add, deps, lat, lon FROM testhosp",
                     engine).dropna(subset=["lat","lon"])
    df["distance"] = haversine(req.lat, req.lon,
                               df["lat"].values, df["lon"].values)
    df = df[(df["distance"] <= req.radius) &
            (df["deps"].apply(lambda s: any(d in s for d in req.deps)))]
    df = df.sort_values("distance").head(30)

    candidates = [
        {"hos_nm":r["hos_nm"], "add":r["add"],
         "deps":[d.strip() for d in r["deps"].split(",")],
         "distance":round(r["distance"],2),
         "lat":r["lat"], "lon":r["lon"],
         "map_url": make_place_url(r["hos_nm"], r["lat"], r["lon"])}
        for _, r in df.iterrows()
    ]
    if not candidates:
        return {"predicted_deps": req.deps, "recommendations": []}

    # 3) LLM 요약 5선
    prompt = {"symptom": req.symptom or "",
              "departments": req.deps,
              "candidates": candidates[:10]}
    msgs = [
        {"role":"system",
         "content":"후보 중 5곳만 JSON 배열로 출력 "
                   "(hos_nm,add,deps,opening_hours,distance,reason 필수)"},
        {"role":"user", "content": json.dumps(prompt, ensure_ascii=False)}
    ]
    try:
        raw = clova_chat(msgs)
        clean = re.sub(r"```.*?```", "", raw, flags=re.S).strip()
        summary = json.loads(clean) if clean.startswith("[") else []
    except Exception:
        summary = []

    return {
        "predicted_deps": req.deps,
        "recommendations": candidates,
        "llm_summary": summary
    }

# ─── 부가 엔드포인트 (진료과·지오코딩) ──────────────────────────────
@router.get("/list_departments")
def list_departments():      return sorted(get_valid_deps())

@router.get("/geocode")
def geocode_address(query:str=Query(...)) -> Dict[str,Any]:
    base="https://maps.apigw.ntruss.com/map-geocode/v2/geocode"
    hd = {"X-NCP-APIGW-API-KEY-ID": NAVER_MAP_ID,
          "X-NCP-APIGW-API-KEY":    NAVER_MAP_SECRET}
    r = requests.get(base, headers=hd, params={"query":query}, timeout=10)
    if r.status_code!=200: raise HTTPException(502,r.text)
    arr=r.json().get("addresses",[])
    if not arr: raise HTTPException(404,"주소 없음")
    a = arr[0]
    return {"lat":float(a["y"]), "lon":float(a["x"]),
            "address_name":a.get("roadAddress") or a.get("jibunAddress")}
app = FastAPI()
app.include_router(router) 