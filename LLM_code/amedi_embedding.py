from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
import os

def load_csv_as_documents(file_path, file_label, chunk_size=500, chunk_overlap=50):
    df = pd.read_csv(file_path)
    docs = []
    for idx, row in df.iterrows():
        content = "\n".join([f"{col}: {row[col]}" for col in df.columns])
        doc = Document(
            page_content=content,
            metadata={"source_file": file_label, "row_index": str(idx)}
        )
        docs.append(doc)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(docs)

def build_vectorstore():
    # 데이터 로드 및 문서화
    dis_docs = load_csv_as_documents("LLM_code/dis_prototype.csv", "질병")
    med_docs = load_csv_as_documents("LLM_code/testmed.csv", "의약품")
    all_docs = dis_docs + med_docs

    # 임베딩 및 벡터스토어 생성
    embedding_model_name = "madatnlp/km-bert"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    vectorstore = FAISS.from_documents(all_docs, embeddings)

    # 저장 경로: app/api/faiss_store
    save_path = "app/api/faiss_store"
    os.makedirs(save_path, exist_ok=True)
    vectorstore.save_local(save_path)
    print(f"✅ FAISS 벡터스토어가 '{save_path}' 경로에 저장되었습니다.")

if __name__ == "__main__":
    build_vectorstore()
