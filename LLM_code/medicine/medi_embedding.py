import pandas as pd
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# 1) CSV 경로
csv_path = "./testmed.csv"
print(f"CSV 파일 경로: {csv_path}")

# 2) CSV 불러오기
df = pd.read_csv(csv_path)

# 3) row → 효능 중심 자연어 변환
def row_to_efficacy_text(row):
    return (
        f"약 이름: {row['ph_nm_c']}\n"
        f"{row['ph_nm_c']}은(는) 다음과 같은 효능이 있습니다:\n"
        f"{row['ph_effect']}\n"
        f"관련 키워드: {row['ph_effect_c']}"
    )

# 4) 문서 생성 (분할 없이)
docs = [
    Document(
        page_content=row_to_efficacy_text(row),
        metadata={
            "item_name": row["ph_nm_c"],
            "efficacy": row.get("ph_effect", ""),
            "keywords": row.get("ph_effect_c", ""),
            "dosage": row.get("ph_use", ""),
            "interaction": row.get("ph_anti_warn", ""),
            "precautions": row.get("ph_warn", ""),
            "side_effects": row.get("ph_s_effect", ""),
            "storage": row.get("ph_stor", "")
        }
    )
    for _, row in df.iterrows()
]
print(f"총 문서 개수: {len(docs)}")

# 5) 임베딩 모델 지정
embedding_model_name = "madatnlp/km-bert"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

# 6) FAISS 벡터 DB 생성
print("임베딩 및 FAISS 인덱스 생성 중...")
vectordb = FAISS.from_documents(docs, embeddings)

# 7) 인덱스 저장
faiss_index_path = "./faiss_index"
vectordb.save_local(faiss_index_path)
print(f"FAISS 인덱스가 '{faiss_index_path}' 폴더에 저장되었습니다.")

# 8) 검색기 준비
retriever = vectordb.as_retriever()
print("✅ 임베딩 및 벡터 저장소 생성 완료.")
