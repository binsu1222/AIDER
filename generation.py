import os
from openai import OpenAI
from typing import List
from langchain_core.documents import Document

# 환경 변수에서 Hugging Face API 토큰을 불러오기
if "HF_TOKEN" not in os.environ:
    print("[ERROR] HF_TOKEN 환경 변수가 설정되지 않았습니다.")
    print("Hugging Face API 토큰을 환경 변수에 설정해주세요.")

client = OpenAI(
    base_url="https://router.huggingface.co/v1", # API 접속 주소
    api_key=os.environ.get("HF_TOKEN", "dummy_key"), # 토큰이 없으면 더미 '키' 사용 (실행 실패)
    timeout=60.0 # 응답 대기 시간 설정 (초)
)

MODEL_NAME = "openai/gpt-oss-20b:groq" 

#검색된 청크(문맥)와 사용자 질문을 통합하여 LLM에 전달할 최종 프롬프트를 구성합니다.
def make_rag_prompt(query: str, retrieval_results: List[Document]) -> str:
    print("\n[Generation] RAG 프롬프트 구성 !")
    
    # 검색된 청크들을 하나의 문자열로 결합
    context_text = "\n\n---\n\n".join([doc.page_content for doc in retrieval_results])
    
    # LLM에게 역할과 제약 조건을 부여하는 시스템 프롬프트 구성
    PROMPT_TEMPLATE = """
    당신은 사용자에게 맞춤형 투자 조언을 제공하는 **최고의 주식 전문가 AI**입니다.
    
    **[역할 및 지시 사항]**
    1. **전문성 유지**: 당신의 모든 답변은 전문적인 주식 시장 분석가 및 투자 전략가의 관점에서 제공되어야 합니다.
    2. **문맥 기반 조언**: 제공된 문맥(Context)에 포함된 **매매 전략**에 기반하여 사용자 질문(Question)에 대해 구체적이고 자세하게 조언을 생성해야 합니다.
    3. **데이터 가정**: 질문에 '사용자의 매매 기록'과 '종목의 종가 데이터'가 포함되어 있다고 가정하고, 이를 활용하여 조언을 생성합니다.
    4. **정보 부족 시**: 만약 문맥에 답변할 충분한 매매 전략 정보가 없다면, 조언을 생성하지 말고 반드시 아래 문구를 사용하세요.
       "제공된 문맥으로는 해당 전략에 기반한 조언을 생성할 수 없습니다. 유튜브 영상 분석 결과를 추가해주세요."
    
    **[출력 형식]**
    당신의 답변은 **반드시** 아래와 같은 **JSON 형식**이어야 합니다. 다른 형식의 텍스트는 일절 포함하지 마세요.
    
    ```json
    {{
        "텍스트": "[여기에 전문가의 투자 조언 내용을 상세히 작성합니다.]"
    }}
    ```
    
    Context:
    {context}
    
    Question: {question}
    
    Assistant:
    """
    
    final_prompt = PROMPT_TEMPLATE.format(context=context_text, question=query)
    
    print(f"  - 검색된 청크 개수: {len(retrieval_results)}개")
    # print(f"  - 최종 프롬프트 (일부):\n{final_prompt[:500]}...")

    return final_prompt

#검색 결과와 질문을 바탕으로 LLM을 호출하여 최종 답변을 생성합니다.
def generate_answer(query: str, retrieval_results: List[Document]) -> str:
    rag_prompt = make_rag_prompt(query, retrieval_results)
    
    print(f"[Generation] LLM ({MODEL_NAME}) 호출 시작!")

    try:
        # LLM 호출
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": rag_prompt}
            ],
            temperature=0.1,
            max_tokens=2048,
            response_format={"type": "json_object"}
        )
        
        # 응답 추출
        if completion.choices:
            final_answer = completion.choices[0].message.content.strip()
            print("[Result] 최종 답변 !")
            print("-" * 50)
            print(final_answer)
            print("-" * 50)
            return final_answer
        else:
            return "LLM으로부터 유효한 응답을 받지 못했습니다."

    except Exception as e:
        error_message = f"[Error] LLM 호출 실패: {e}"
        print(error_message)
        # Hugging Face Router를 사용하는 경우 모델 이름 또는 토큰 오류가 가장 흔합니다.
        if "API key" in str(e) or "Unauthorized" in str(e):
            print("  -> HF_TOKEN이 유효한지, 환경 변수에 정확히 설정되었는지 확인하세요.")
        if "404" in str(e) or "Invalid model" in str(e):
            print(f"  -> 모델 이름 ({MODEL_NAME})이 Hugging Face Router에서 유효한지 확인하세요.")
        
        return "LLM 응답 생성 중 오류 발생. 로그를 확인하세요."

# --- 테스트 코드 (main.py에서 이 함수를 사용할 예정) ---
if __name__ == "__main__":
    # 이 부분은 main.py에서 vector_store.py를 통해 검색 결과를 받은 후 호출됩니다.
    
    # 모의 검색 결과 (실제로는 vector_store.py에서 가져옴)
    mock_results = [
        Document(page_content="주요 매매 전략은 저가 매수 후 고가 매도이며, 장기적인 가치 투자를 지향한다."),
        Document(page_content="분할 매수와 분산 투자는 위험 관리를 위한 핵심 원칙으로 동영상에서 강조되었다."),
        Document(page_content="이 전략은 특히 변동성이 큰 시장에서 안정적인 수익률을 목표로 한다.")
    ]
    
    test_query = "이 동영상에서 주요 매매 전략은 무엇인가요?"
    
    # 환경 변수 HF_TOKEN이 설정되어 있다면 실제 LLM 호출을 시도합니다.
    if os.environ.get("HF_TOKEN"):
        generate_answer(test_query, mock_results)
    else:
        print("\n[Info] HF_TOKEN이 설정되지 않아 실제 LLM 호출을 건너뛰고 프롬프트 구성만 테스트합니다.")
        mock_prompt = make_rag_prompt(test_query, mock_results)
        print("\n--- 구성된 프롬프트 ---")
        print(mock_prompt)