import os
import pandas as pd
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from dotenv import load_dotenv

# ───── 0. 환경 설정 ─────
load_dotenv()
embedding_model = "madatnlp/km-bert"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
llm = OllamaLLM(model="exaone3.5:7.8b", temperature=0.3)

# ───── 1. CSV 로드 → Document 생성 ─────
def load_csv_as_documents(path: str, label: str):
    df = pd.read_csv(path)
    docs = []
    for _, row in df.iterrows():
        content = " ".join([str(cell) for cell in row.values])
        doc = Document(page_content=content, metadata={"source_file": label})
        docs.append(doc)
    return docs

# ───── 2. 문서 로딩 및 임베딩 (청크 없음) ─────
base_dir = os.path.join(os.path.dirname(__file__), "Ragfile")
all_docs = []
all_docs += load_csv_as_documents(os.path.join(base_dir, "dis.csv"), "dis.csv")
all_docs += load_csv_as_documents(os.path.join(base_dir, "medi.csv"), "medi.csv")
all_docs += load_csv_as_documents(os.path.join(base_dir, "hospi.csv"), "hospi.csv")

vectordb = FAISS.from_documents(all_docs, embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 10})

# ───── 3. LLM 질의 처리 ─────
def run_query(question: str):
    system_template = (
        "너는 의료 정보에 정통한 챗봇이야.\n"
        "질병 정보는 dis.csv, 약품 정보는 medi.csv, 병원 정보는 hospi.csv에서만 참고해야 해.\n"
        "문서의 메타데이터에 source_file 항목이 있으니 어떤 파일 출신인지 구분할 수 있어.\n"
        "문서 내용에 없거나 불확실한 정보는 '잘 모르겠습니다'라고 정직하게 답변해.\n"
        "답변 마지막에는 출처 파일명을 명시해줘."
    )
    human_template = (
        "질문: {question}\n\n"
        "참고할 문서:\n\n{context}"
    )
    system_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": chat_prompt}
    )

    return qa.invoke({"query": question})

# ───── 4. 실행부 ─────
if __name__ == "__main__":
    while True:
        query = input("\n🩺 무엇이 궁금하신가요? (종료하려면 'exit')\n> ").strip()
        if query.lower() in {"exit", "quit"}:
            print("👋 챗봇을 종료합니다.")
            break
        try:
            answer = run_query(query)
            print("\n🧠 답변:\n", answer)
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
