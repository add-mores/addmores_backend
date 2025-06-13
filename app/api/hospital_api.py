from fastapi import APIRouter, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import os
import requests
from dotenv import load_dotenv

load_dotenv()
router = APIRouter()

DATABASE_URL = os.getenv("DATABASE_URL")
NAVER_MAP_ID     = os.getenv("NEXT_PUBLIC_MAP_CLIENT_ID")
NAVER_MAP_SECRET = os.getenv("NEXT_PUBLIC_MAP_CLIENT_SECRET")
engine = create_engine(DATABASE_URL)

def haversine(lat1, lon1, lat2s, lon2s):
    R = 6371
    dlat = np.radians(lat2s - lat1)
    dlon = np.radians(lon2s - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2s)) * np.sin(dlon/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

class FilterRequest(BaseModel):
    lat: float
    lon: float
    radius: float = 1.0
    deps: Optional[List[str]] = None
    search_name: Optional[str] = None

@router.post("/api/hospital")
def filter_hospitals(req: FilterRequest):
    query = """
    SELECT hos_nm, hos_type, pv, city, add, deps, lat, lon
    FROM testhosp
"""
    df = pd.read_sql(query, engine).dropna(subset=["lat", "lon"])
    df["distance"] = haversine(req.lat, req.lon, df["lat"].values, df["lon"].values)

    # 진료과 필터
    if req.deps and req.deps != ["string"]:
        df = df[df["deps"].apply(lambda t: any(dept.strip() in t.split(",") for dept in req.deps if t))]

    # 병원명 필터
    if req.search_name and req.search_name != "string":
        df = df[df["hos_nm"].str.contains(req.search_name, case=False, na=False)]

    df = df[df["distance"] <= req.radius].sort_values("distance")
    records = []
    for _, row in df.head(30).iterrows():
        records.append({
            "hos_nm": row["hos_nm"],
            "add": row["add"],
            "deps": row["deps"],
            "lat": row["lat"],
            "lon": row["lon"],
            "distance": round(row["distance"], 2)
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
    base = "https://maps.apigw.ntruss.com/map-geocode/v2/geocode"
    headers = {
        "X-NCP-APIGW-API-KEY-ID": NAVER_MAP_ID,
        "X-NCP-APIGW-API-KEY":    NAVER_MAP_SECRET
    }
    r = requests.get(base, headers=headers, params={"query": query}, timeout=10)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=r.text)
    arr = r.json().get("addresses", [])
    if not arr:
        raise HTTPException(status_code=404, detail="주소를 찾을 수 없습니다.")
    a = arr[0]
    return {
        "lat": float(a["y"]),
        "lon": float(a["x"]),
        "address_name": a.get("roadAddress") or a.get("jibunAddress")
    }


