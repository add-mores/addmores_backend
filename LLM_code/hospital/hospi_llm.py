import os
import json
import requests
from math import radians, sin, cos, sqrt, atan2
from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import logging
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv

# ─── 0. 환경 설정 ─────────────────────────
load_dotenv()
logging.getLogger("sentence_transformers.SentenceTransformer").setLevel(logging.ERROR)

# ─── 1. 기본 세팅 ──────────────────────────
BASE_DIR = os.path.dirname(__file__)
INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")
EMBEDDING_MODEL = "madatnlp/km-bert"

llm = OllamaLLM(model="exaone3.5:7.8b", temperature=0.3)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

NAVER_MAP_ID = os.getenv("NEXT_PUBLIC_MAP_CLIENT_ID")
NAVER_MAP_SECRET = os.getenv("NEXT_PUBLIC_MAP_CLIENT_SECRET")
if not (NAVER_MAP_ID and NAVER_MAP_SECRET):
    raise ValueError("❌ NAVER 지도 API 키가 .env에 없습니다.")

if not os.path.exists(INDEX_PATH):
    raise FileNotFoundError("❌ faiss_index 가 없습니다. 먼저 임베딩을 생성하세요.")
vectordb = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

# ─── 2. 거리 계산 함수 ─────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return round(R * 2 * atan2(sqrt(a), sqrt(1 - a)), 2)

# ─── 3. 네이버 지오코딩 ─────────────────────
def geocode_naver(addr: str) -> Dict[str, Any]:
    url = "https://maps.apigw.ntruss.com/map-geocode/v2/geocode"
    headers = {
        "X-NCP-APIGW-API-KEY-ID": NAVER_MAP_ID,
        "X-NCP-APIGW-API-KEY": NAVER_MAP_SECRET
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

# ─── 4. 진료과 예측 LLM 호출 ────────────────
def exaone_predict_departments(question: str) -> List[str]:
    messages = [
        {"role": "system", "content": "당신은 의료 상담 봇입니다. 사용자의 증상 설명에서 예상 진료과를 추출하세요. JSON 배열로만 출력하고, 마크다운을 쓰지 마세요."},
        {"role": "user", "content": question}
    ]
    try:
        resp = llm.invoke(messages).strip()
        json_start, json_end = resp.find("["), resp.rfind("]") + 1
        return json.loads(resp[json_start:json_end])
    except:
        return []

# ─── 5. LLM 기반 병원 랭킹 ──────────────────
def exaone_rank_hospitals(symptom: str, deps: List[str], cands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}
    ]
    try:
        resp = llm.invoke(messages).strip()
        json_start = resp.find("[")
        json_end = resp.rfind("]") + 1
        return json.loads(resp[json_start:json_end])
    except:
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

# ─── 6. 거리 기반 후보 병원 추천 ───────────
def recommend_by_distance(symptom: str, lat: float, lon: float, deps: List[str]) -> List[Dict[str, Any]]:
    total = vectordb.index.ntotal
    all_docs = vectordb.similarity_search(symptom, k=total)
    cands = []
    for d in all_docs:
        m = d.metadata
        dist = haversine(lat, lon, float(m.get("lat", 0)), float(m.get("lon", 0))) if m.get("lat") and m.get("lon") else 99.9
        deps_list = [x.strip() for x in str(m.get("treatment", "")).split(",") if x.strip()]
        cands.append({
            "hos_nm": m.get("hospital_name", ""),
            "add": m.get("address", ""),
            "deps": deps_list,
            "distance": dist,
            "emergency": m.get("emergency", "정보 없음")
        })
    wanted = ["이비인후과"] if not deps else deps
    cands = [c for c in cands if any(w in c["deps"] for w in wanted)]
    cands.sort(key=lambda x: x["distance"])
    for r in [1, 3, 5, 10, 20, 50]:
        near = [c for c in cands if c["distance"] <= r]
        if len(near) >= 5:
            return exaone_rank_hospitals(symptom, deps, near[:20])
    return exaone_rank_hospitals(symptom, deps, cands[:10])

# ─── 7. 실행부 ─────────────────────────────
if __name__ == "__main__":
    # ─ 주소 입력 ─
    while True:
        raw = input("📍 도로명 주소를 입력하세요: ").strip()
        if raw.lower() in {"exit", "quit"}:
            print("👋 챗봇을 종료합니다.")
            exit(0)
        try:
            geo = geocode_naver(raw)
            break
        except Exception as e:
            print(f"❌ 주소를 인식하지 못했습니다: {e}")
            print("🔁 다시 시도해주세요.\n")

    lat, lon = geo["lat"], geo["lon"]
    print(f"📌 위치: {geo['address']} (위도:{lat}, 경도:{lon})")

    # ─ 챗봇 반복 ─
    while True:
        # 질문 입력 및 진료과 추론 반복
        while True:
            print("\n🧠 무엇이 궁금하세요?")
            question = input("   예: '목이 너무 아파요', '근처 가까운 이비인후과 알려줘': ").strip()

            if question.lower() in {"exit", "quit"}:
                print("👋 챗봇을 종료합니다.")
                exit(0)
            if not question:
                print("⚠️ 질문을 입력해주세요.")
                continue

            deps = exaone_predict_departments(question)
            if not deps:
                print("❌ 진료과를 예측하지 못했습니다. 다시 질문해 주세요.")
                continue

            print(f"🩺 예측 진료과: {', '.join(deps)}")
            break  # 성공 시 빠져나옴

        # 병원 추천
        top5 = recommend_by_distance(question, lat, lon, deps)

        print("\n🏥 추천 병원:")
        for i, h in enumerate(top5, 1):
            print(f"{i}. {h['hos_nm']}")
            print(f"   📍 주소: {h['add']}")
            print(f"   🏥 진료과: {', '.join(h['deps'])}")
            print(f"   🧭 거리: {h['distance']} km")
            print(f"   💬 이유: {h.get('reason','사유 없음')}\n")

        again = input("🔁 다른 증상이나 질문을 입력하시겠습니까? (y/n): ").strip().lower()
        if again != 'y':
            print("👋 챗봇을 종료합니다.")
            break


