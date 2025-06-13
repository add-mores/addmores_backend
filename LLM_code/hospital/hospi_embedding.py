# LLM_code/hospital/hospi_embedding.py

#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import torch
from datetime import datetime

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv

load_dotenv()

def log(msg: str):
    print(f"🕒 [{datetime.now().strftime('%H:%M:%S')}] {msg}")

class KmBertEmbeddings:
    def __init__(self, model_name: str = "madatnlp/km-bert", device: str = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModel.from_pretrained(model_name)
        self.device    = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    def _mean_pooling(self, hidden_state, mask):
        mask_expanded = mask.unsqueeze(-1).expand(hidden_state.size()).float()
        summation     = (hidden_state * mask_expanded).sum(dim=1)
        divisor       = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return summation / divisor

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeds = []
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc   = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model(**enc, return_dict=True)
            pooled = self._mean_pooling(out.last_hidden_state, enc["attention_mask"])
            embeds.extend(pooled.cpu().tolist())
        return embeds

    def embed_query(self, text: str) -> list[float]:
        enc = self.tokenizer([text], padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**enc, return_dict=True)
        pooled = self._mean_pooling(out.last_hidden_state, enc["attention_mask"])
        return pooled.cpu().squeeze(0).tolist()

    def __call__(self, text: str) -> list[float]:
        return self.embed_query(text)

def run_embedding(
    csv_path: str = "hospitals_with_emergency_phone.csv",
    index_path: str = "faiss_index",
    model_name: str = "madatnlp/km-bert"
):
    log("1. CSV 로드 중")
    csv_path = os.path.join(os.path.dirname(__file__), "hospitals_with_emergency_phone.csv")
    df = pd.read_csv(csv_path)
    df["treatment"]     = df["treatment"].fillna("")  
    log(f"  → {len(df)}개 레코드 로드")
    
    df = df.rename(columns={
    "hospital_name": "hos_nm",
    "hospital_type": "hos_type",
    "province": "pv",
    "city": "city",
    "address": "add",
    "treatment": "deps",
    "응급의료기관여부": "emergency"  # ✅ 꼭 필요
})

    log("2. Document 생성 중")
    docs = []
    for i, row in df.iterrows():
        content = (
            f"병원이름: {row['hos_nm']}\n"
            f"주소: {row['add']}\n"
            f"진료과목: {row['deps']}\n"
            f"응급의료기관 여부: {row.get('emergency', '정보 없음')}\n"
            f"위도: {row.get('lat')}, 경도: {row.get('lon')}\n"
            f"운영시간: 운영시간 정보 없음"
        )
        docs.append(Document(
            page_content=content,
            metadata={
                "index": i,
                "hospital_name": row["hos_nm"],
                "address": row["add"],
                "treatment": row["deps"],
                "lat": row.get("lat"),
                "lon": row.get("lon"),
                "emergency": row.get("emergency", "정보 없음"),
                # "opening_hours": row["opening_hours"],
            }
        ))
    log(f"  → {len(docs)}개 Document 생성")
    # ✅ 누락 여부 확인
    if len(docs) != len(df):
        log(f"⚠️ 병원 수 불일치! CSV: {len(df)}개, Document: {len(docs)}개")
    else:
        log("✅ CSV 병원 수와 Document 수 일치")

    # log("3. 텍스트 분할 시작")
    # splitter = RecursiveCharacterTextSplitter
    # split_docs = splitter.split_documents(docs)
    # log(f"  → {len(split_docs)}개 chunk 생성")

    log("3. 임베딩 로드")
    embedder = KmBertEmbeddings(model_name)

    if os.path.exists(index_path):
        log("✅ 기존 FAISS 인덱스 로드")
        vectordb = FAISS.load_local(index_path, embedder, allow_dangerous_deserialization=True)
    else:
        log("🛠️ FAISS 인덱스 생성 중...")

    # 텍스트 분할 생략하고 바로 임베딩 생성
        vectordb = FAISS.from_documents(docs, embedder)

    vectordb.save_local(index_path)
    log("💾 인덱스 저장 완료")

    log("임베딩 완료")

if __name__ == "__main__":
    run_embedding()
