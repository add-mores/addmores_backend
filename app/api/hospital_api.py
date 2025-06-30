from fastapi import APIRouter, Query, HTTPException
from typing   import Dict, Any, List, Optional
from pydantic import BaseModel
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import os, requests
from dotenv import load_dotenv

load_dotenv()
router = APIRouter()

DATABASE_URL     = os.getenv("DATABASE_URL")
NAVER_MAP_ID     = os.getenv("NEXT_PUBLIC_MAP_CLIENT_ID")
NAVER_MAP_SECRET = os.getenv("NEXT_PUBLIC_MAP_CLIENT_SECRET")

engine = create_engine(DATABASE_URL)


# ────────────────────────────── 공통 함수 ──────────────────────────────
def haversine(lat1, lon1, lat2s, lon2s):
    """벡터라이즈된 하버사인 거리(km)"""
    R     = 6371
    dlat  = np.radians(lat2s - lat1)
    dlon  = np.radians(lon2s - lon1)
    a     = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2s)) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# ────────────────────────────── 스키마 ──────────────────────────────
class FilterRequest(BaseModel):
    lat:  float
    lon:  float
    radius: float = 1.0
    deps:  Optional[List[str]] = None
    search_name: Optional[str] = None
    only_er: bool = False          # 🆘 응급실만 보기 체크박스용


# ────────────────────────────── 라우터 ──────────────────────────────
@router.post("/api/hospital")
def filter_hospitals(req: FilterRequest):
    # ① 필요한 컬럼 모두 SELECT
    query = """
        SELECT hos_nm, hos_type, pv, city, add, deps,
               emer, emer_phone,            -- 응급실 정보
               lat, lon
        FROM testhosp
    """
    df = pd.read_sql(query, engine).dropna(subset=["lat", "lon"])

    # ② 거리 계산
    df["distance"] = haversine(req.lat, req.lon, df["lat"].values, df["lon"].values)

    # ③ 진료과 필터
    if req.deps and req.deps != ["string"]:
        df = df[df["deps"].apply(lambda t: any(d.strip() in t.split(",") for d in req.deps if t))]

    # ④ 병원명 검색
    if req.search_name and req.search_name != "string":
        df = df[df["hos_nm"].str.contains(req.search_name, case=False, na=False)]

    # ⑤ 응급실만 보기
    if req.only_er:
        df = df[df["emer"].str.strip() == "있음"]

    # ⑥ 반경·정렬
    df = df[df["distance"] <= req.radius].sort_values("distance")

    # ⑦ 결과 직렬화
    records = []
    for _, row in df.head(30).iterrows():   # 최대 30개
        records.append({
            "hos_nm":      row["hos_nm"],
            "add":         row["add"],
            "deps":        row["deps"],
            "emer":        row["emer"],
            "emer_phone":  row["emer_phone"],
            "lat":         row["lat"],
            "lon":         row["lon"],
            "distance":    round(row["distance"], 2)
        })

    return {"recommendations": records}


@router.get("/list_departments")
def list_departments():
    df = pd.read_sql("SELECT deps FROM testhosp", engine)

    all_depts = set()
    for deps in df["deps"].dropna():
        for d in deps.split(","):
            all_depts.add(d.strip())

    return sorted(all_depts)


@router.get("/geocode")
def geocode_address(query: str = Query(...)) -> Dict[str, Any]:
    base    = "https://maps.apigw.ntruss.com/map-geocode/v2/geocode"
    headers = {
        "X-NCP-APIGW-API-KEY-ID": NAVER_MAP_ID,
        "X-NCP-APIGW-API-KEY":    NAVER_MAP_SECRET
    }
    r = requests.get(base, headers=headers, params={"query": query}, timeout=10)

    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=r.text)

    addresses = r.json().get("addresses", [])
    if not addresses:
        raise HTTPException(status_code=404, detail="주소를 찾을 수 없습니다.")

    a = addresses[0]
    return {
        "lat": float(a["y"]),
        "lon": float(a["x"]),
        "address_name": a.get("roadAddress") or a.get("jibunAddress")
    }
