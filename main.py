import os
from transcript import transcript
from vector_store import create_vector_db, search_strategy, reset_db
from generation import generate_answer

def main():
    reset_db()  # (선택사항) DB 초기화가 필요할 때 사용하세요
    
    print("YouTube 동영상 ID를 입력하세요 (예: EMfQuZkzmuc):")
    video_id = input().strip()
    print(f"YouTube 동영상 ID: {video_id}")

    # 자막 가져오기
    transcript_text = transcript(video_id)
    if transcript_text is None:
        print("자막을 가져오지 못했습니다. 종료합니다.")
        return
    
    vector_db = create_vector_db(transcript_text)
    query = "이 동영상에서 주요 매매 전략은 무엇인가요?"
    # 저장된 ChromaDB에서 질문과 유사한 내용을 검색
    retrieval_result = search_strategy(query)

    # 검색된 청크(문맥)와 사용자 질문을 통합하여 LLM에 전달할 최종 답변
    final_answer = generate_answer(query, retrieval_result)

if __name__ == "__main__":
    main()