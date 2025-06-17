from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import RetrievalQA

# FastAPI 라우터
router = APIRouter()

# 사용자 입력 스키마
class QueryRequest(BaseModel):
    question: str

# 임베딩 모델 및 벡터스토어 로딩 (앱 실행 시 1회)
embedding_model_name = "madatnlp/km-bert"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
vectorstore = FAISS.load_local("./app/api/faiss_store", embeddings, allow_dangerous_deserialization=True)

# RAG 함수
def rag_answer(question: str) -> str:
    retriever = vectorstore.as_retriever()
    llm = Ollama(model="exaone3.5:7.8b", temperature=0.1, num_predict=1024)

    system_template = (
        "너는 전문 의료 정보 비서야.\n"
        "사용자의 질문을 보고, 질병 / 의약품 중 어떤 카테고리에 해당하는지 판단하고,\n"
        "관련 있는 문서만 참고해서 답변해. 문서의 출처(source_file)를 확인해서 불필요한 문서는 무시해야 해.\n"
        "정보가 명확하지 않거나 문서에 없다면 '모르겠습니다'라고 답변해."
    )
    human_template = (
        "질문: {question}\n\n"
        "참고 가능한 문서 리스트 (내용과 출처 포함):\n\n{context}"
    )
    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template)
    ])

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": chat_prompt}
    )

    result = qa.invoke({"query": question})
    return result["result"]

# POST /rag 엔드포인트
@router.post("/llm/amedi")
async def get_rag_response(query: QueryRequest):
    try:
        answer = rag_answer(query.question)
        return {"question": query.question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
