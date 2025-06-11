import os
import json
import re
import requests
from math import radians, sin, cos, sqrt, atan2
from typing import List, Dict, Any
from json import JSONDecodeError
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv

# ─── 0. 환경 설정 ─────────────────────────
load_dotenv()

# ─── 1. 기본 세팅 ──────────────────────────
BASE_DIR   = os.path.dirname(__file__)
INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")
EMBEDDING_MODEL = "madatnlp/km-bert"

llm        = OllamaLLM(model="exaone3.5:7.8b", temperature=0.3)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

NAVER_MAP_ID     = os.getenv("NEXT_PUBLIC_MAP_CLIENT_ID")
NAVER_MAP_SECRET = os.getenv("NEXT_PUBLIC_MAP_CLIENT_SECRET")
if not (NAVER_MAP_ID and NAVER_MAP_SECRET):
    raise ValueError("❌ NAVER 지도 API 키가 .env에 없습니다.")

# ─── 2. FAISS 인덱스 로딩 ──────────────────
if not os.path.exists(INDEX_PATH):
    raise FileNotFoundError("❌ faiss_index 가 없습니다. 먼저 임베딩을 생성하세요.")
vectordb = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

# ─── 3. 거리 계산 함수 ─────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return round(R * 2 * atan2(sqrt(a), sqrt(1-a)), 2)

# ─── 4. 네이버 지오코딩 ─────────────────────
def geocode_naver(addr: str) -> Dict[str,Any]:
    url = "https://maps.apigw.ntruss.com/map-geocode/v2/geocode"
    headers = {
        "X-NCP-APIGW-API-KEY-ID": NAVER_MAP_ID,
        "X-NCP-APIGW-API-KEY":    NAVER_MAP_SECRET
    }
    res = requests.get(url, headers=headers, params={"query": addr}, timeout=10)
    if res.status_code != 200:
        raise RuntimeError(f"❌ 지오코딩 실패: {res.text}")
    data = res.json().get("addresses", [])
    if not data:
        raise RuntimeError("❌ 주소를 찾을 수 없습니다.")
    d = data[0]
    return {"lat": float(d["y"]), "lon": float(d["x"]),
            "address": d.get("roadAddress") or d.get("jibunAddress")}

# ─── 5. LLM 호출 함수 ──────────────────────
def exaone_rank_hospitals(symptom:str, deps:List[str], cands:List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    print(f"🔢 3단계: LLM 전달 후보 수 = {len(cands)}") 
    prompt = {
        "symptom": symptom,
        "departments": deps,
        "candidates": cands
    }
    system_msg = (
        "당신은 의학 전문가입니다. 증상(symptom), 예측 진료과(departments), 후보 병원(candidates)을 바탕으로 위경도 기준 가까운 5개 병원을 추천합니다. "
        "각 병원은 반드시 진료과(deps)를 포함해야 하며, 추천 사유(reason)도 작성해야 합니다. "
        "진료과목은 반드시 후보 병원(candidates)에서 제공하는 진료과목을 사용해야 합니다. "
        "진료과목를 추출할 경우 반드시 모든 진료과를 전부 포함해야 합니다. "
        "**각 병원은 반드시 다음 필드를 포함해야 합니다:** "
        "`hos_nm`, `add`, `deps`, `distance`, `reason`. "
        "`deps`는 문자열 목록입니다. 절대로 `deps` 안에 다른 필드(distance, reason 등)를 넣지 마세요. "
        "**JSON 배열로만** 출력하며, 마크다운·번호·코드펜스를 쓰지 마세요. "
        "후보 목록(candidates) 외 병원은 추가하지 마세요."
        "응답 예시: [{\"hos_nm\": \"병원명\", \"add\": \"주소\", \"deps\": [\"진료과목\", \"distance\": 1.23, \"reason\": \"추천 사유\"}]"

    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}
    ]

    try:
        resp = llm.invoke(messages).strip()
        print("🧠 원본 응답:", resp[:300])  # 응답 디버깅

        json_start = resp.find("[")
        json_end = resp.rfind("]") + 1

        if json_start == -1 or json_end == -1:
            raise ValueError("❌ JSON 배열이 응답에서 감지되지 않았습니다.")

        json_text = resp[json_start:json_end]

        try:
            return json.loads(json_text)
        except json.JSONDecodeError as je:
            print("❌ JSON 디코딩 실패:", je)
            print("⚠️ 잘린 JSON:", json_text)
            raise

    except Exception as e:
        print("❌ LLM 호출 실패 또는 파싱 실패:", e)

        # fallback
        return [
            {
                "hos_nm": c["hos_nm"],
                "add": c["add"],
                "deps": c["deps"],
                "distance": c["distance"],
                "reason": "사유 없음"
            }
            for c in cands[:10]
        ]

# ─── 6. 거리 기반 후보 + LLM 추천 ───────────
def recommend_by_distance(symptom: str, lat: float, lon: float, deps: List[str]) -> List[Dict[str, Any]]:
    total = vectordb.index.ntotal           # 인덱스에 저장된 전체 벡터 개수
    all_docs = vectordb.similarity_search(symptom, k=total)
    print(f"🔢 1단계: 유사도 검색 문서 수 = {len(all_docs)}")
    cands: List[Dict[str, Any]] = []
    for d in all_docs:
        m = d.metadata
        dist = haversine(lat, lon, float(m.get("lat", 0)), float(m.get("lon", 0))) if m.get("lat") and m.get("lon") else 99.9
        deps_list = [x.strip() for x in str(m.get("treatment", "")).split(",") if x.strip()]
        cands.append({
            "hos_nm": m.get("hospital_name", ""),
            "add":    m.get("address", ""),
            "deps":   deps_list,
            "distance": dist,
            "emergency": m.get("emergency", "정보 없음")
        })

    wanted = ["이비인후과"] if not deps else deps
    cands = [c for c in cands if any(w in c["deps"] for w in wanted)]
    print(f"🔢 2단계: ENT 포함 후보 수 = {len(cands)}") 

    cands.sort(key=lambda x: x["distance"])
    print("   ↳ 상위 10개 거리:", [c["distance"] for c in cands[:10]])
    top: List[Dict[str, Any]] = []
    for r in [1, 3, 5, 10, 20, 50]:
        near = [c for c in cands if c["distance"] <= r]
        if len(near) >= 5:
            top = near[:20]
            break
    if not top:
        top = cands[:10]

    return exaone_rank_hospitals(symptom, deps, top)

# ─── 7. 실행부 ─────────────────────────────
if __name__ == "__main__":
    # ① 주소는 한 번만 물어보고 좌표 고정
    raw = input("📍 도로명 주소를 입력하세요: ").strip()
    try:
        geo = geocode_naver(raw)
    except Exception as e:
        print(e); exit(1)

    lat, lon = geo["lat"], geo["lon"]
    print(f"📌 위치: {geo['address']} (위도:{lat}, 경도:{lon})")

    # ② 증상은 반복 입력
    while True:
        symptom = input("\n📝 증상을 입력하세요 (종료: exit): ").strip()
        if symptom.lower() in {"exit", "quit"}:
            print("👋 챗봇을 종료합니다."); break
        if not symptom:
            print("⚠️ 증상을 입력해주세요."); continue

        top5 = recommend_by_distance(symptom, lat, lon, deps=[])
        print("\n🏥 추천 병원:")
        for i, h in enumerate(top5, 1):
            print(f"{i}. {h['hos_nm']}")
            print(f"   📍 주소: {h['add']}")
            print(f"   🏥 진료과: {', '.join(h['deps'])}")
            print(f"   🧭 거리: {h['distance']} km")
            print(f"   💬 이유: {h.get('reason','사유 없음')}\n")