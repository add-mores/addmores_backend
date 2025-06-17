from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import pandas as pd

# CSV 파일 로드
dis_df = pd.read_csv("Ragfile/dis.csv")
med_df = pd.read_csv("Ragfile/medi.csv")
hos_df = pd.read_csv("Ragfile/hospi.csv")

# 행을 텍스트로 변환

def load_csv_as_documents(file_path, file_label, chunk_size=500, chunk_overlap=50):
    df = pd.read_csv(file_path)
    docs = []

    # 각 행(row)을 Document로 변환
    for _, row in df.iterrows():
        content = "\n".join([f"{col}: {row[col]}" for col in df.columns])
        doc = Document(
            page_content=content,
            metadata={"source_file": file_label}
        )
        docs.append(doc)

    # ✅ 청크 분할 (LangChain 내장 Splitter 사용)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunked_docs = text_splitter.split_documents(docs)

    return chunked_docs

# 세 CSV 파일을 각각 문서로 로드
hos_docs = load_csv_as_documents("Ragfile/hospi.csv", "병원")
dis_docs = load_csv_as_documents("Ragfile/dis.csv", "질병")
med_docs = load_csv_as_documents("Ragfile/medi.csv", "의약품")

all_docs = hos_docs + dis_docs + med_docs

# FAISS 벡터 저장소 생성
embedding_model_name = "madatnlp/km-bert"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

vectorstore = FAISS.from_documents(all_docs, embeddings)

# RAG 기반 질의 함수 정의
def rag_answer(question):
    retriever = vectorstore.as_retriever()

    llm = Ollama(model="exaone3.5:7.8b", temperature=0.1, num_predict=1024)

    # System 프롬프트 (카테고리 구분 및 출처 필터링 지시)
    system_template = (
        "너는 전문 의료 정보 비서야.\n"
        "사용자의 질문을 읽고, 아래 세 가지 중 어떤 카테고리에 해당하는지 판단해:\n"
        "- '질병': 질병 이름, 증상, 원인, 치료 방법, 관련 진료과 등에 대한 정보\n"
        "- '의약품': 약 이름, 효능, 주의 사항,부작용, 복용법 등에 대한 정보\n"
        "- '병원': 병원 이름, 위치(주소), 진료 과목, 응급실 여부 등에 대한 정보\n\n"
        "문서의 출처(source_file)를 확인해서 질문과 관련 없는 문서는 무시하고,\n"
        "관련 있는 문서만 참고해서 답변해.\n"
        "정보가 명확하지 않거나 문서에 없다면 '모르겠습니다'라고 답변해."
    )
    system_prompt = SystemMessagePromptTemplate.from_template(system_template)

    # 사용자 질문 + 문서 context 프롬프트
    human_template = (
        "질문: {question}\n\n"
        "참고 가능한 문서 리스트 (내용과 출처 포함):\n\n{context}"
    )
    human_prompt = HumanMessagePromptTemplate.from_template(human_template)

    # 통합 프롬프트 템플릿
    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

    # RetrievalQA 구성
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": chat_prompt}
    )

    result = qa.invoke({"query": question}) 
    return result["result"]


# 5. 사용자 입력 및 응답 출력
while True:
    query = input("\n무엇을 도와드릴까요? (예: 증상, 감기에 좋은 약, 강남 병원 추천 등)\n(종료하려면 'exit' 또는 '종료' 입력)\n")
    if query.lower() in ["exit", "종료", "quit"]:
        print("👋 이용해주셔서 감사합니다.")
        break

    response = rag_answer(query)
    print("\n🤖 답변:\n", response)