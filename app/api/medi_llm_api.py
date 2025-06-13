from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
from langchain_community.llms import Ollama
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import difflib
import re

router = APIRouter()

# 모델 및 벡터 DB 로딩
embedding_model_name = "madatnlp/km-bert"
faiss_index_path = "./app/api/medi_faiss_index"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
vectordb = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})
llm = Ollama(model="exaone3.5:7.8b", temperature=0.1, num_predict=1024)

all_item_names = list({doc.metadata["item_name"] for doc in vectordb.docstore._dict.values()})

class RecommendationRequest(BaseModel):
    query: str
    age_group: Optional[str] = None 
    is_pregnant: Optional[bool] = False
    chronic_conditions: Optional[List[str]] = []

WARNING_MSG = "\n---\n\u26a0\ufe0f 이 답변은 참고용입니다. 실제 복용 전에는 반드시 병원 진료를 받거나 약사와 상담을 통해 처방받으시기 바랍니다.\n---"


def create_context(docs):
    items_list_str = "\n".join([f"- {doc.metadata['item_name']}" for doc in docs])
    context = "\n\n".join([
        f"약 이름: {doc.metadata.get('item_name', 'N/A')}\n"
        f"효능: {doc.metadata.get('efficacy', 'N/A')}\n"
        f"복용법: {doc.metadata.get('dosage', 'N/A')}\n"
        f"주의사항: {doc.metadata.get('precautions', 'N/A')}\n"
        f"부작용: {doc.metadata.get('side_effects', 'N/A')}"
        for doc in docs
    ])
    return items_list_str, context


def find_mentioned_item_name(query, item_names):
    sorted_names = sorted(item_names, key=len, reverse=True)
    for name in sorted_names:
        if name in query:
            return name
    return None


def find_similar_item_name(query, item_names, cutoff=0.6):
    words = re.findall(r'\w+', query)
    all_matches = []
    postpositions_to_remove = ["이랑", "랑", "처럼", "하고"]
    for word in words:
        for postposition in postpositions_to_remove:
            if word.endswith(postposition):
                word = word[:-len(postposition)]
                break
        matches = difflib.get_close_matches(word, item_names, n=5, cutoff=cutoff)
        all_matches.extend(matches)
    return list(dict.fromkeys(all_matches))


def condition_filter(doc, age_group, is_pregnant, chronic_conditions):
    text = f"{doc.metadata.get('precautions', '')} {doc.metadata.get('side_effects', '')}"
    age_keywords = {
        '소아': ['소아', '어린이', '유아', '영아', '아동'],
        '청소년': ['청소년', '10대', '10세', '십대'],
        '성인': ['성인'],
        '노인': ['노인', '고령자']
    }
    if age_group and age_group != '성인':
        for keyword in age_keywords.get(age_group, [age_group]):
            if keyword in text:
                return False
    if is_pregnant and any(word in text for word in ['임산부', '임신', '임부']):
        return False
    for disease in chronic_conditions:
        if disease and disease in text:
            return False
    return True


@router.post("/llm/medicine")
def recommend_medicine(req: RecommendationRequest):
    query = req.query.strip()
    
    # 1) 증상 기반 추천 : 쿼리가 "증상은 ..." 으로 시작하면 증상 추천 로직 실행
    if query.startswith("증상은"):
        # 1-1) 연령대, 임신, 기저질환 정보가 모두 없으면 안내 메시지
        if (not req.age_group or req.age_group.strip() == "") and not req.is_pregnant and not req.chronic_conditions:
            return {
                "message": "추가 정보가 필요합니다. 연령대, 임신 여부, 기저질환 정보를 입력해주세요.",
                "need_conditions": True
            }

        symptom_text = query[len("증상은"):].strip()
        # 먼저 5개만 검색
        docs = vectordb.similarity_search(symptom_text, k=5)
        filtered_docs = [doc for doc in docs if condition_filter(doc, req.age_group, req.is_pregnant, req.chronic_conditions)]
        
        # 조건에 맞는 약이 없으면 50개로 다시 검색
        if not filtered_docs:
            docs = vectordb.similarity_search(symptom_text, k=50)
            filtered_docs = [doc for doc in docs if condition_filter(doc, req.age_group, req.is_pregnant, req.chronic_conditions)]
        
        if not filtered_docs:
            raise HTTPException(status_code=404, detail="조건에 맞는 의약품을 찾을 수 없습니다.")
				
        items_list_str, context = create_context(filtered_docs)
        prompt = f"""
당신은 숙련된 약사이며, 일반의약품에 대한 전문가입니다.

사용자의 증상에 대해 아래 약 목록 중에서 적절한 약을 최대 3개까지 추천하십시오.
반드시 아래 약 목록에 있는 약만 추천해야 합니다.

사용자 정보:
- 연령대: {req.age_group}
- 임신 여부: {'예' if req.is_pregnant else '아니오'}
- 기저질환: {', '.join(req.chronic_conditions) if req.chronic_conditions else '없음'}

---

약 목록:
{items_list_str}

약 정보:
{context}

사용자 증상: {req.query}
"""
        return {"result": llm.invoke(prompt) + WARNING_MSG}
        
    # 2) 약 이름 기반 추천 (증상기반 아니면 이쪽)
    item_name = find_mentioned_item_name(query, all_item_names)
    if not item_name:
        close_matches = find_similar_item_name(query, all_item_names)
        if close_matches:
            # 유사 약 제안: 유사약 리스트를 리턴해서 프론트에 보여줄 수도 있음
            return {
                "message": "입력하신 약과 비슷한 약을 찾았습니다. 아래 중 선택해주세요.",
                "similar_medicines": close_matches
            }
        else:
            # 약 이름, 유사약 없으면 증상기반 추천도 같이 안내할 수도 있음
            return {
                "message": "해당 약 이름을 찾을 수 없습니다. 증상 기반 검색을 원하시면 '증상은 ...' 으로 시작하여 검색해주세요."
            }

    # item_name 이 확정되면 해당 약 정보 가져오기
    base_docs = [doc for doc in vectordb.docstore._dict.values() if doc.metadata.get("item_name") == item_name]
    if not base_docs:
        raise HTTPException(status_code=404, detail="해당 약 정보를 찾을 수 없습니다.")

    base_doc = base_docs[0]

    # 3) 유사 약 추천 요청 처리 ("유사", "비슷", "대체" 단어 포함 시)
    if any(k in query for k in ["유사", "비슷", "대체"]):
        base_vector = embeddings.embed_query(base_doc.page_content)
        similar_docs = vectordb.similarity_search_by_vector(base_vector, k=10)
        filtered_similar_docs = [d for d in similar_docs if condition_filter(d, req.age_group, req.is_pregnant, req.chronic_conditions) and d.metadata.get("item_name") != item_name]
        similar_items = [d.metadata.get("item_name") for d in filtered_similar_docs][:3]
        similar_metas = [d.metadata for d in filtered_similar_docs][:3]
				
        if not similar_items:
            return {"message": "조건에 맞는 유사 약을 찾을 수 없습니다."}
         

        prompt = f"""
당신은 전문 약사입니다.

사용자가 관심 있는 약 '{item_name}'과(와) 유사한 일반의약품 최대 3개를 아래 목록에서 추천해주세요.

💊 기준 약: {item_name}
- 주요 효능: {base_doc.metadata.get('efficacy', '정보 없음')}
- 주의사항: {base_doc.metadata.get('precautions', '정보 없음')}

💊 유사 약 목록:
"""
        for i, meta in enumerate(similar_metas):
            prompt += f"""
{i+1}. 약 이름: {similar_items[i]}
- 주요 효능: {meta.get('efficacy', '정보 없음')}
- 주의사항: {meta.get('precautions', '정보 없음')}
"""

        prompt += """
위 정보를 참고하여 각 유사 약이 기준 약과 어떤 점에서 유사한지 자연스럽고 구체적으로 설명해주세요.
"""
        return {"result": llm.invoke(prompt) + WARNING_MSG}

    # 4) 일반 약 정보 설명
    _, context = create_context(base_docs)
    prompt = f"""
당신은 전문 약사입니다.

아래는 일반의약품 '{item_name}' 에 대한 설명 요청입니다.
사용자 질문: {req.query}

[참고 문서]
{context}
"""
    return {"result": llm.invoke(prompt) + WARNING_MSG}
