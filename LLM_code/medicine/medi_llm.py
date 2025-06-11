from langchain_community.llms import Ollama
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import difflib
import re

# 모델 및 벡터 DB 로딩
embedding_model_name = "madatnlp/km-bert"
faiss_index_path = "./faiss_index"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
vectordb = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})
llm = Ollama(model="exaone3.5:7.8b", temperature=0.1, num_predict=1024)

# 약 이름 리스트 확보
all_item_names = list({doc.metadata["item_name"] for doc in vectordb.docstore._dict.values()})
print(f"총 등록된 약 이름 수: {len(all_item_names)}")

# context 생성 함수
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

# 약 이름 포함 여부
def find_mentioned_item_name(query, item_names):
    sorted_names = sorted(item_names, key=len, reverse=True)
    for name in sorted_names:
        if name in query:
            return name
    return None

# 유사 약 이름 찾기
def find_similar_item_name(query, item_names, cutoff=0.6):
    words = re.findall(r'\w+', query)
    all_matches = []
    # 최소한으로 제거할 조사 리스트
    postpositions_to_remove = ["이랑", "랑", "처럼", "하고"]
    for word in words:
        # 조사 제거: 조사 리스트에 있는 조사만 뒤에 붙으면 제거
        for postposition in postpositions_to_remove:
            if word.endswith(postposition):
                word = word[:-len(postposition)]
                break  # 하나만 제거
        
        matches = difflib.get_close_matches(word, item_names, n=5, cutoff=cutoff)
        all_matches.extend(matches)
    return list(dict.fromkeys(all_matches))

# 조건 필터 함수
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

# 항상 마지막에 붙는 참고용 워닝 메시지
WARNING_MSG = "\n---\n⚠️ 이 답변은 참고용입니다. 실제 복용 전에는 반드시 병원 진료를 받거나 약사와 상담을 통해 처방받으시기 바랍니다.\n---"

# 메인 루프
while True:
    query = input("\n💊 증상이나 약 이름을 입력하세요. 종료하려면 'q'를 입력하세요.\n\n📝 입력: ")
    if query.lower() in ["q", "quit", "exit"]:
        print("종료합니다.")
        break

    item_name = find_mentioned_item_name(query, all_item_names)
    if not item_name:
        close_matches = find_similar_item_name(query, all_item_names)
        found_match = False
        if close_matches:
            for match in close_matches:
                yn = input(f"🤔 혹시 '{match}' 약을 말씀하신 건가요? (네/아니요): ").strip()
                if yn == "네":
                    item_name = match
                    found_match = True
                    break

        if not found_match:
            print("⚠️ 입력하신 약 이름과 정확히 일치하거나 유사한 약을 찾지 못했습니다.")
            print("💡 입력하신 내용을 바탕으로 증상 기반 일반 의약품 추천을 도와드릴게요 (다시 의약품 검색을 하고 싶으시면 q).\n")

            # 사용자 정보 수집
            symptom_query = query
            age_group = input("연령대가 어떻게 되시나요? (소아, 청소년, 성인, 노인): ").strip()
            if age_group.lower() == 'q':
                continue
            pregnant_input = input("현재 임신 중이신가요? (예/아니요): ").strip()
            if pregnant_input.lower() == 'q':
                continue
            chronic_input = input("앓고 있는 기저질환이 있다면 알려주세요 (쉼표로 구분, 없으면 엔터): ").strip()
            if chronic_input.lower() == 'q':
                continue

            is_pregnant = pregnant_input == "예"
            chronic_conditions = [c.strip() for c in chronic_input.split(",")] if chronic_input else []

            symptom_docs = vectordb.similarity_search(symptom_query, k=5)
            filtered_docs = [doc for doc in symptom_docs if condition_filter(doc, age_group, is_pregnant, chronic_conditions)]

            if not filtered_docs:
                print("⚠️ 조건에 맞는 약이 없어 더 많은 후보를 확인합니다...")
                extended_docs = vectordb.similarity_search(symptom_query, k=70)
                filtered_docs = [doc for doc in extended_docs if condition_filter(doc, age_group, is_pregnant, chronic_conditions)]

            if not filtered_docs:
                print("❌ 사용자 조건에 맞는 적절한 일반의약품을 찾을 수 없습니다.")
                continue

            items_list_str, context = create_context(filtered_docs)

            prompt = f"""
당신은 숙련된 약사이며, 일반의약품에 대한 전문가입니다.

사용자의 증상에 대해 아래 약 목록 중에서 적절한 약을 최대 3개까지 추천하십시오.  
반드시 아래 약 목록에 있는 약만 추천해야 합니다.  
주의사항과 부작용은 사용자 안전을 위해 자세히 설명하십시오.

사용자 정보:
- 연령대: {age_group}
- 임신 여부: {"예" if is_pregnant else "아니오"}
- 기저질환: {', '.join(chronic_conditions) if chronic_conditions else '없음'}

각 추천 약에 대해 다음 정보를 포함하십시오:
1. 약 이름  
2. 추천 이유  
3. 복용법  
4. 주의사항  
5. 부작용  

---

약 목록:  
{items_list_str}

약 정보:  
{context}

사용자 증상: {symptom_query}
"""
            print("\n💡 추천 결과:\n")
            print(llm.invoke(prompt) + WARNING_MSG) 
            print("\n" + "=" * 80 + "\n")
            continue

    # 약 이름이 명확히 있는 경우
    # 1) 기준 문서 하나만 찾기 (item_name과 정확히 일치하는 문서)
    base_docs = [doc for doc in vectordb.docstore._dict.values() if doc.metadata.get("item_name") == item_name]

    if not base_docs:
        print(f"⚠️ '{item_name}' 약 정보가 벡터 DB에 없습니다.")
        continue

    base_doc = base_docs[0]

    if any(keyword in query for keyword in ["비슷한", "유사한", "대체"]):
        # 2) 기준 문서 벡터 추출
        base_vector = embeddings.embed_query(base_doc.page_content)

        # 3) 벡터 기반 유사 문서 검색 (기준 약 제외)
        similar_docs = vectordb.similarity_search_by_vector(base_vector, k=10)
        similar_items = []
        for d in similar_docs:
            name = d.metadata.get("item_name")
            if name != item_name and name not in similar_items:
                similar_items.append(name)
            if len(similar_items) >= 3:
                break

        # 4) 유사 약 메타데이터 수집
        similar_metas = []
        for name in similar_items:
            doc = next((d for d in vectordb.docstore._dict.values() if d.metadata.get("item_name") == name), None)
            if doc:
                similar_metas.append(doc.metadata)

        # 5) 기준 약 메타데이터
        base_meta = base_doc.metadata

        # 6) 프롬프트 생성
        prompt = f"""
당신은 전문 약사입니다.

사용자가 관심 있는 약 '{item_name}'과(와) 유사한 일반의약품 최대 3개를 아래 목록에서 추천해주세요.

💊 기준 약: {item_name}
- 주요 효능: {base_meta.get('efficacy', '정보 없음')}
- 주의사항: {base_meta.get('precautions', '정보 없음')}

💊 유사 약 목록:
"""
        for i, name in enumerate(similar_items):
            meta = similar_metas[i]
            prompt += f"""
    {i+1}. 약 이름: {name}
- 주요 효능: {meta.get('efficacy', '정보 없음')}
- 주의사항: {meta.get('precautions', '정보 없음')}
"""

        prompt += """
위 정보를 참고하여 각 유사 약이 기준 약과 어떤 점에서 유사한지 자연스럽고 구체적으로 설명해주세요.
"""

        print("\n🔍 유사 약 추천 및 설명:\n")
        print(llm.invoke(prompt) + WARNING_MSG) 

    else:
        _, context = create_context(base_docs)
        prompt = f"""
당신은 전문 약사입니다.

아래는 일반의약품 **'{item_name}'** 에 대한 설명 요청입니다.
사용자가 자연어로 질문한 문장이 포함되어 있지만, 답변은 반드시 '{item_name}'을 기준으로 제공해주세요.

- 사용자의 질문 표현과 약 이름이 정확히 일치하지 않더라도, '{item_name}'을 중심으로 설명하세요.
- '{item_name}'이 문서에 없을 경우, 유사한 약 이름을 찾아 참고하고, 이 사실을 명확히 안내해주세요.

[참고 문서]
{context}

[사용자 질문]
{query}
"""
        print(f"\n💊 약 정보 설명: 사용자님, 질문하신 '{item_name}'에 대해 설명드리겠습니다.\n")
        print(llm.invoke(prompt) + WARNING_MSG) 
        print("\n" + "=" * 80 + "\n")
